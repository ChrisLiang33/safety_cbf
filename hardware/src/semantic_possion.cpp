#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <random>
#include <cmath>
#include <queue>
#include <map>
#include <array>
#include <unistd.h>
#include <cstring>
#include <cfloat>
#include <set>
#include <shared_mutex>
#include <cuda_runtime.h>
#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "clound_merger.h"
#include "poisson/human_tracker.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"
#include <message_filter/subscriber.h>
#include <message_filter/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace ss {

enum class PipelineStage{
  OcupancyPreprocess,
  SemanticFusion,
  GeometryShaping,
  GuidanceField,
  SafetyFieldSolve,
  DhdhUpdate,
  PredictiveControl,
  RealtimeFilter,
  CommandDispatch
};

struct TimingSample{
  double occupency_preprocess_ms{0.0};
  double semantic_fusion_ms{0.0};
  double geometry_shaping_ms{0.0};

  double guidance_boundry_setup_ms{0.0};
  double guidance_social_expansion_ms{0.0};
  double guidance_laplace_ms{0.0};
  double guidance_copyout_ms{0.0};

  double safety_feild_solve_ms{0.0};
  double dhdt_update_ms{0.0};
  double predictive_control_ms{0.0};
  double realtime_filter_ms{0.0};
  double command_dispatch_ms{0.0};
  double field_data_age_ms{0.0};
  double end_to_end_grid_ms{0.0};
};

struct ConnectedComponentsData{
  cv::Mat binary;
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  int num_labels{0}
};

struct SemanticStageOutput{
  bool tight_area{false};
  std::vector<HumanTrack> active_tracks;
};

struct GuidanceStageOutput{
  float* bounded_Guidance{nullptr}
};

class ScopedTimer{
public:
  explicit ScopedTimer(double& target_ms) : target_ms_(target_ms), t0_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer(){
    const auto t1 = std::chrono:;steady_clock::now();
    target_ms_ = std::chrono::duration<double, std::milli>(t1 -t0_).count();
  }

private:
  double& target_ms_;
  std:chrono::steady_clock::time_point t0_;
};

class PoissonControllerNode : public rclcpp::Node{
public:
  PoissonControllerNode() : Node("poisson_control", sport_req(this)){
    declare_and_load_parameters();
    initialize_clocks_and_flags();
    initialize_static_grids();
    allocate_persostent_buffers();
    initialize_robot_kernels();
    initialize_mpc();
    initialize_ros_interfaces();
    startup_robot();
  }

  ~PoissonControllerNode() override {
    if (hgrid1) cudaFreeHost(hgrid1);
    if (hgrid0) cudaFreeHost(hgrid0);
    if (bound) cudaFreeHost(bound);
    if (force) cudaFreeHost(force);

    if(dhdt_grid) std::free(dhdt_grid);
    if(guidance_x_grid) std::free(guidance_x_grid);
    if(guidance_y_grid) std::free(guidance_y_grid);

    if(hgrid_temp_) std::free(hgrid_temp_);
    if(guidance_x_temp_) std::free(guidance_x_temp_);
    if(guidance_y_temp_) std::free(guidance_y_temp_);
    if(forceing_zero_temp_) std::free(forcing_zero_temp_);
    if(bound_guidance_temp_) std::free(bound_guidance_temp_);
    if(class_map_temp_expanded_) std::free(class_map_temp_expanded_);
    if(boundry_temp_) std::free(bountry_temp_);
    if(inflated_bound_temp_) std::free(inflated_bound_temp_);
    if(inflated_class_temp_) std::free(inflated_class_temp_);

    if(robot_kernel_human) std::free(robot_kernel_human);
    if(robot_kernel_obstacle) std::free(robot_kernel_obstacle);

    if(outFileCSV.is_open()) outFileCSV.close();
    if(outFileBIN.is_open()) outFileBIN.close();
  }

private:
  // 1. ros orchestration
  void teleop_callback(geometry_msgs::msg::Twist::uniquePtr msg){
    handle_teleop_input(*msg);
  }

  void keyboard_callback(std_msgs::msg::Int32::UniquePtr msg){
    handle_keyboard_input(*msg)
  }

  void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){
    handle_occupancy_update(*msg)
  }

  void class_map_callback(nav_msgs::msg:OccupancyGrid::UniquePtr msg){
    if(msg->data.size() != IMAX * JMAX){
      RCLCPP_WARN(
        this->get_logger(),
        "class_map size mismathc: got %zu expected %d",
        msg->data.size(),
        IMAX*JMAX
      );
      return;
    }

    for(int n = 0; n < IMAX *JMAX; ++n){
      class_map[n] = msg->data[n]
    }
  }

  void visibility_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){
    if(msg->data.size() != IMAX*JMAX){
      RCLCPP_WARN(
        this->get_logger(),
        "visibility_map size mismatch: got %zu expected %d",
        msg->data.size(),
        IMAX*JMAX
      );
      return;
    }

    for(int n = 0; n <IMAX*JMAX; ++n){
      visibility_map[n] = msg->data[n];
    }
  }

  void state_update_callback(const nav_msgs::msg:Odometry::SharedPtr data){
    handle_state_update(*data);
  }

  void mpc_callback(){
    handle_mpc_callback();
  }

  //2. high level handlers

  void handle_teleop_input(const geometry_msgs::msg::Twist& msg){
    const std::vector<float> vtb = {
      static_cast<float>(msg.linear.x),
      static_cast<float>(msg.lienar.y),
      static_cast<float>(msg.angular.z)
    };

    vt = {
      std::cos(x[2]) * vtb[0] - std::sin(x[2]) * vtb[1],
      std::sin(x[2]) * vtb[0] + std::cos(x[2]) * vtv[1],
      vtb[2]
    };

    xd[0] += 0.01f * vt[0]
    xd[1] += 0.01f * vt[1]
    xd[2] += 0.01f * vt[2]

    if(!start_flag){
      xd = x;
      vt = {0.0f, 0.0f, 0.0f};
    }
  }

  void handle_keyboard_input(const std_msgs::msg::Int32& msg){
    if(!save_flag) t_start = std::chrono::steadu_clock.now();
    else t_ms = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_start).count() * 1.0e3f 

    char param = ' ';
    const int ch = msg.data;
    switch(ch){
      case ' ':
        space_counter++;
        if (space_counter >= 1) save_flag = true;
        if (space_counter >= 3) start_flag = true;
        if (space_counter >= 6) stop_flag = true;
        break;
      case 'r': realtime_sf_flag = ! realtime_sf_flag; break;
      case 'p': predictive_sf_flag = !predictive_sf_flag; break;
      case 'd': 
        param = current_parameter_deck.back();
        current_paramether_deck.pop_back();
        if(current_parameter_deck.empty()){
          current_parameter_deck = sorted_parameter_deck;
          std::shuffle(current_parameter_deck.begin(), current_parameter_deck.end(), gen);
        }
        break;
      default break;
    }

    apply_parameter_deck_selection(param, ch);
    maybe_write_experiment_data();
  }

  void handle_occupancy_update(const nav_msgs::msg::OccupancyGrid& msg){
    const auto grid_start = std:;chrono::steady_clock.now()

    if(!update_grid_metadata_from_message(msg)){
      return;
    }

    preprocess_occupancy();
    auto semantic_output = run_semantic_fusion();
    build_inflated_boundaries(semantic_output.tight_area);
    auto guidance_output = build_guidance_field(semantic_output.active_tracks);
    h_flag = solve_safety_field(guidance_output);

    if(start_flag&& dhdt_flag){
      ScopedTimer timer(timing_.dhdt_update_ms);
      update_temporal_field_derivative();
    }

    latest_field_time_step_ = std:;chrono::stready_clock::now();
    timing_.end_toend_grid_ms = std:;chrono::duration<double, std::milli>(std::chrono::steady_clock.now() - grid_start).count();

    if(enable_display) render_visialization();
    publish_timing_data()
  }

  void handle_state_update(const nav_msgs::msg::Odometry& data){
    update_robot_state(data);

    std::vector<float> v_input_body = form_nomial_body_command();
    timing_.field_data_age_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock.now() - latest_field_timestamp_).count();
    {
      ScopedTimer timer(timing_.realtime_filter_ms);
      if(h_flag) compute_realtime_safe_control(v_input_body);
      else v = v_input_body;
    }

    postprocess_command();

    {
      Scopedtimer timer(timing_.command_dispatch_ms);
      dispatch_robot_command();
    }
  }

  void handle_mpc_update(){
    if(!(predictive_sf_flag && h_flag && mpc-mutex.trylock())) return;
    std::lock_guard<std::mutex> lock(mpc_mutex, std::adopt_lock);
    ScopedTimer timer(timing_.predictive_control_ms);
    compute_predictive_control();
  }

  //3. pipeline: occupancy /semantics/ geometry

  void preprocess_occupancy(){
    ScopedTimer timer(timing_.occupency_preprocess_ms);
    build_occ_map(occ1, occ0, conf);

    std::memmcp(
      hgrid_temp_,
      hgrid1,
      IMAX * JMAX *QMAX *sizeof(float)
    );

    find_boundary(hgrid_temp, occ1, false);
  }

  SemanticStageOutput run_semantic_fusion(){
    ScopedTimer timer(timing_.semantic_fusion_ms);
    SemanticStageOutput out;
    label_human_clusters(occ1);
    out.active_tracks = human_tracker_->get_active_tracks();
    out.tight_area = is_tight_area();
    return out;
  }

  void vuild_inflated_boundriers(bool tight_area){
    ScopedTimer timer(timing_.geometry_shaping_ms);

    float* bound_q0 = bound;
    std::memcpy(bound_q0, occ1, IMAX*JAMX*sizeof(float));
    inflated_occupancy_grid(bound_q0, class_map_expanded);

    #pragma omp parallell for num_threads(4)
    for (int q=0; q <QMAX; q++){
      float* bounded_slice = bound + q *IMAX*JMAX;
      float* hgrid_slice = hgrid_temp_ + q *IMAX*JMAX;
      if(q!=0){
        std::memcpy(bound_slice, occ1, IMAX*JMAX* sizeof(float));
        inflate_occupancy_grid(bound_slice, class_map_expanded);
      }
      find_boundray(hgrid_slice, bound_slice, true, tight_are, class_map_expanded);
    }
  }

  //4 pipeline: guidance/safety field

  GuidanceStageOutput build_guidance_field(const std::vector<HumanTrack>& active_tracks){
    GuidanceStageOutput out;
    out.bound_guidance = bound;

    std::memset(guidance_x_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
    std::memset(guidance_y_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
    std::memset(forcing_zero_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
    std::memset(tangent_layer_display, 0, IMAX * JMAX * sizeof(int8_t));

    const float c_yaw = std::cos(x[2]);
    const float s_yaw = std::sin(x[2]);
    const float vn_body_x = c_yaw * vn[0] + s_yaw * vn[1];
    const float vn_body_y = -s_yaw * vn[0] + c_yaw * vn[1];
    {
      ScopedTimer timer(timing_.guidance_boundary_setup_ms);
      compute_boundary_gradients(guidance_x_temp_, guidance_y_temp_, bound, class_map_expanded, x[0], x[1], vn_body_x, vn_body_y, true);

      #pregma omp parallel for num_threads(4)
      for (int q = 1; q < QMAX; q ++){
        float* bound_slice = bound + q * IMAX *JMAX;
        float* gx = guidance_x_temp_ + q * IMAX *JMAX;
        float* gy = guidance_y_temp + q * IMAX *JMAX;
        compute_boundary_gradients(gx, gy, bound_slice, class_map_expanded, x[0], x[1], vn_body_x, vn_body_y, false);
      }
    }
    {
      ScopedTimer timer(timing_.guidance_social_expansion_ms);
      if(enabl_social_navigation_ && social_tangetn_layers_ > 0 && !human_boundary_info_.empty()){
        out.bound_guidance = bound_guidance_temp_;
        const float sign = compute_tangent_diraction(active_tracks, 0.0f, 0.0f, vn_body_x, vn_body_y);
        for(int q = 0; q<QMAX; q++){
          expand_human_obstacles_for_guidance(
            bound_guidance_temp_ + q * IMAX *JMAX,
            guidance_x_temp_ + q * IMAX *JMAX,
            guidance_y_temp_ + q * IMAX *JMAX,
            bound + q * IMAX*JMAX;
            social_tangent_layers_,
            social_layer_thickness_,
            social_tangent_bias_,
            sign
          );
        }
      }
    }
    {
      ScopedTimer timer(timing_.guidance_laplace_ms);
      solve_guidance_laplace(out.bound_guidance)
    }

    compute_guidance_forcing(out.bound_guidance);

    {
      ScopedTimer timer(timing_.guidnace_copyout_ms);
      std::memcpy(guidance_x_display, guidance_x_temp_, IMAX*JMAX*sizeof(float));
      std::memcpy(guidance_y_display, guidance_y_temp_, IMAX *JMAX *sizeof(float));
      std::memcpy(buid_display, bound, IMAX * JMAX * sizeof(float));
      std::memcp(guidance_x_grid, guidance_x_temp_, IMAX*JMAX*QMAX*sizeof(float))
      std::memcp(guidance_y_grid, guidance_y_temp_, IMAX*JMAX*QMAX*sizeof(float))
    }
    return out;
  }

  void solve_guidance_laplace(float* bounded_guidance){
    const float v_RelTol = 1.0e-4f;
    const int N_guidance = IMAX/5;
    const float w_SOR_guidance = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N_guidnace + 1)));
    (void)Kernell::possonSolve(guidnace_x_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
    (void)Kernell::possonSolve(guidnace_y_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
  }

  void compute_guidnace_forcing(const float* bound_guidnace){
    #pragma omp parallel for num_threads(4)
    for(int q=0; q<QMAX; q++){
      float* force_slice = force + q *IMAX*JMAX;
      const float* bound_slice = bound_guidnace + q *IMAX*JMAX;
      float* gx = guidance_x_temp_ + q * IMAX * JMAX;
      float* gy = guidance_y_temp + q * IMAX*JMAX;
      compute_optimal_forcing_function(force_slice, gx, gy, bound_slice);
      for(int n=0; n<IMAX*JMAX;n++) force_slice[n] *= DS*DS;
    }
  }

  bool solve_safety_field(const GuidanceStageOutput& guidance){
    ScopedTimer timer(timing_.safety_field_ms);

    const float relTol = 1.0e-4f;
    const int N = IMAX/5;
    const float w_Sor = 2.0f/(1.0f + std::sin(M_PI / static_cast<float>(N+1)));

    bool success = true;
    if(!hgrid_temp_ || !force || !guidance.bound_guidance){
      success = false;
    }else{
      (void)Kernel::poissonSolve(hgrid_temp_, force, guidance.bound_guidance, relTol, w_Sor);
    }

    std::memcpy(occ0, occ1, IMAX*JMAX*sizeof(float));
    std::memcpy(hgrid0, hgrid1, IMAX*JMAX*QMAX*sizeof(float));
    std::memcpy(hgrid1, hgrid_temp, IMAX*JAMX*QMAX*sizeof(float));

    if(success){
      dhdt_flag = true;
    }
    return success
  }

  void update_temporal_field_derivative(){
    const float wc= 10.0f;
    const float kc = 1.0f - std::exp(-wc * dt_grid);
    for(int=0; i<IMAX;i++){
      for int(j=0;j<JMAX;j++){
        for(int q=0;q<QMAX; q++){
          const float i0 = static_cast<float>(i) + dx[1] /DS;
          const float j0 = static_cast<float>(j) + dx[0] /DS;
          const bool in_grid = (i0 >= 0.0f) && (i0 <= static_cast<float>(IMAX-1)) && (j0 >=0.0f) && (j0 <=static_cast<float>(JMAX-1));
          float dhdt_ij = 0.0f;
          if(in_grid){
            const float h0v = trilinear_interpolation(hgrid0, i0, j0, q);
            const float h1v = trilinear_interpolation(hgrid1, i, j,q);
            dhdt_ij = (h1v - h0v) / dt_grid;
          }
          dhdt_grid[q*IMAX*JMAX + i* JMAX+j] *= 1.0f - kc;
          dhdt_grid[q*IMAX*JAMX + i*JMAX+j] += kc *=dhdt_ij
        }
      }
    }
  }

  //5 control
  void compute_predictive_control(){
    std::vector<float> x_body_link = {0.0f, 0.0f, x[2]};
    for(int i=0; i< MAX_SQP_ITERS; i++){
      const float c = std::cos(x[2]);
      const float s = std::sin(x[2]);
      std::vector<float> vn_body = {c * vn[0 + s * vn[1], -s *vn[0]] + c* vn[1], vn[2]};
      mpc3d_controller.update_cost(vn_body);
      mpc3d_controller.update_constraints(hgrid1, dhdt_grid, guidance_x_grid, guidance_y_grid, x_body_link, xc, grid_age, wn, issf, cbf_sigma_epsilon, cbf_sigma_keppa_);
      mpc3d_controller.solve();
      if(mpc3d_controller.update_residual() < 1.0f) break;
    }
    mpc3d_controller.set_input(vd);
  }

  std::vector<float> from_nominal_body_command(){
    vn = vt;
    if(predictive_sf_flag)return vd;
    const float c = std::cos(x[2]);
    const float s = std::sin(x[2]);
    return {c* vn[0] + s * vn[1], -s * vn[0] + c * vn[1], vn[2]};
  }

  void compute_realtime_safe_control(const std::vector<float>& v_input_body){
    safety_filter(v_input_body);
  }

  void postprocess_command(){
    const std::vector<float> vb_new = v;
    if(std:;abs(vb[0]) > 10.0f || std::abs(vb[1]) > 10.0f || std::abs(vb[2]) > 10.0f) sit_flag = true;
    vb[0] = std::clamp(vb[0], -vel_max_x_bwd, vel_max_x_fwd);
    vb[1] = std::clamp(vb[1], -vel_max_y_, vel_max_y_);
    vb[2] = std::clamp(vb[2], -vel_max_yaw_, vel_max_yaw_);
  }

  void dispatch_robot_command(){
    if(stop_flag){
      sport_req.StopMove(req);
      sport_req.StandDown(req);
      rclcpp::shutdown();
    }else if(sit_flag){
      sport_req.StopMove(req);
      sport_req.StandDown(req);
    }else if (start_flag){
      sport_req.Move(req, vb[0], vb[1], vb[2]);
    }
  }

  //6. visualizatipn/ logging/ experiment support
  void redner_visualization(){

  }

  bool should_publish_logging_now(){

  }

  void publish_profiling_data(){

  }

  void maybe_write_experiment_data(){

  }

  void apply+parameter_deck_selection(char param, int ch){

  }

  //7 helpers initialization

  void initialize_mpc()[
    mpc3d_controller.set_velocity_bounds(
      vel_max_x_fwd_,
      vel_max_x_bwd_,
      vel_max_y_,
      vel_max_yaw_,
    )
    mpc3d_controller.setup_QP();
    mpc3d_controller.solve();
  ]

  void update_robot_state(const nav_msgs::msg::Odometry& data){
    dt_state = std::chrono::duration<float>(std::chrono::steady_clock.now() - t_state).count();
    t_state = std::chrono::steady_clock::now();
    grid_age += dt_state;

    x[0] = data.pose.pose.position.x;
    x[1] = data.pose.pose.position.y;

    const auto& q = data.pose.pose.orientation;
    const float sin_yaw = 2.0f * (q.w * q.z + q.x * q.y);
    const float cos_yaw = 1.0f - 2.0f * (q.y * q,y + q.z * q.z);
    x[2] = std::atan2(sin_yaw, cos_yaw);
  }

  bool update_grid_metadata_from_message(const nav_msgs::msg::OcupancyGrid& msg){
    if(msg.data.size() != IMAX*JMAX){
      RCLCPP_WARN(
        this->get_logger();
        "occupancy_grid size mismatch got %zu expected %d",
        msg.data.size(),
        IMAX*JMAX
      );
      return false;
    }
    dt_grid = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_grid).count();
    t_grid = std::chrono::steady_clock::now();
    grid_age = dt_grid;

    dx[0] = msg.info.origin.position.x - xc[0];
    dx[1] = msg.info.origin.position.y - xc[1];
    xc[0] = msg.info.origin.position.x;
    xy[1] = msg.info.origin.position.y;

    for(int n=0; n<IMAX*JMAX; n++){
      conf[n] = msg.data[n];
    }
    return true;
  }

  void startup_robot(){
    sport_req.RecoveryStand(req);
    sleep(1);
    sport_req.SpeedLevel(req, 1);
    sleep(1);
  }

  void initialize_robot_kernels(){
    robot_kernel_obstacle = nullptr;
    robot_kernel_human = nullptr;

    robot_kernel_dim_obstacle = initialize_robot_kernel(robot_kernel_obstacle, robot_MOS_obstacle);
    robot_kernel_dim_human = initialize_robot_kernel(robot_kernel_human, robot_MOS_human);
  }

  int initialize_robot_kernel(float*& kernel, float mos){

  }

  void initialize_static_grids(){
    for(int n=0; n<IMAX*JMAX;n++){
      occ1[n] = 1.0f;
      occ0[n] = 1.0f;
      conf[n] = 0;
      grid_temp[n] = 0.0f;
      class_map[n] = 0;
      visibility_map[n] = 0;
      class_map_expanded[n] = 0;
      guidance_x_display[n] = 0.0f;
      guidance_y_display[n] = 0.0f;
      bound_display[n] = 0.0f;
      tangent_layer_display[n] = 0;
    }
  }

  void initialize_clock_and_flags(){
    gen.seed(rd());
    current_parameter_deck = sorted_paramether_deck;
    std:shuffle(current_parameter_deck.begin(), current_paramether_deck.end(), gen);

    t_start = std::chrono::steady_clock::now();
    t_grid = std::chrono::steady_clock::now();
    t_state = std::chrono::steady_clock::now();
    latest_field_timestamp = std::chrono::steady_clock::now();
    latest_logging_publush_time = std::chrono::steady_clock::now();
  }

  void initialize_logging_outputs(){

  }

  void declare_and_load_parameters(){

  }

  void allocate_persistent_buffers(){

  }

  void initialize_ros_interfaces(){

  }

  void publish_logging_data(){
    if(!logging_data_pub_) return;
    std::msgs::msg::Float32MultiArray msg;
    msg.data = {
      t_ms,
      static_cast<float>(space_counter),
      x[0], x[1], x[2],
      v[0], v[1], v[2],
      vt[0], vt[1], vt[2],
      h, dhdx, dhdy, dhdq, dhdt,
      wn,
      static_cast<float>(realtime_sf_flag | predictive_sf_flag) 
    }
    logging_data_pub_->publish(msg);
  }

  //8 existing lowlevel methods to keep/move verbatim

  //good
  void build_occ_map(float* occ_map, const float* occ_map_old, const int8_t* conf){
    const int8_t T_hi = 85;
    const int8_t T_lo = 64;
    
    for(int i=0; i<IMAX; i++){
      for(int j=0; j<JMAX; j++){
        const int i0 = i + static_cast<int>(std::round(dx[1] / DS));
        const int j0 = j + static_cast<int>(std::round(dx[0] / DS));

        const bool in_grid = (i0 >= 0) && (i0 <IMAX) && (j0 >=0) && (j0 <JMAX);
        const bool strong = conf(i * JMAX + j) >= T_hi;
        const bool weak = conf(i * JMAX + j) >=T_lo;

        if(strong){
          occ_map[i* JMAX + j] = -1.0f;
        }else if (weak && in_grid){
          occ_map[i * JAMX + j] = occ_map_old[i0 * JMAX +j0];
        }else{
          occ_map[i * JMAX + j] = 1.0f;
        }
      }
    }
  }

  void find_boundary(float* grid, float* bound, bool fix_flag, bool tight_area, const int8_t* class_map){

  }

  void fill_eliptical_robot_kernel(float* kernel, float yawq, int dim, float expo, float mos){

  }

  void inflate_occupancy_grid(float* bound, int8_t* class_map){

  }

  //good
  bool is_tight_area(){
    auto tracks = human_tracker_->get_active_tracks();
    if(tracks.empty()) return false;

    float min_human_dist = FLT_MAX;
    for(const auto& track: tracks){
      const float d = std::sqrt(std::pow(track.x - x[0], 2) + std::pow(track.y - x[1], 2));
      min_human_dist = std::main(min_human_dist, d);
    }

    const float ic = y_to_i(0.0f, xc[1]);
    const float jc = x_ti_j(0.0f, xc[0]);
    const float qc = yaw_to_q(x[2], xc[2]);

    const float ic_clamped = std::clamp(ic, 0.0f, static_cast<float>(IMAX - 1));
    const float jc_clamped = std::clamp(jc, 0.0f, static_cast<float>(JMAX - 1));
    
    const float h_at_robot = trilinear_interpolation(hgrid1, ic_clamped, jc_clamped, qc);

    const bool tight = (min_human_dist < tight_area_human_threshold) && (h_at_robot < tight_area_h_threshold_);
    return tight;
  }

  void compute_boundary_gradients(float* guidance_x, float* guidance_y, float* bound, const int8_t* class_map, float /*rx*/, float /*ry*/, float        /*vn_x*/, float /*vn_y*/, bool populate_human_info){

  }

  float compute_tangent_direction(const std::vector<HumanTrack>& active_tracks, float /*rx*/, float /*ry*/, float /*vn_x*/, float /*vn_y*/) {

  }

  void expand_human_obstacles_for_guidance(float* bound_guidance, float* guidance_x,float* guidance_y,const float* bound_original,int num_layers,int layer_thickness, float bias_strength, float sign) {

  }

  void compute_optimal_forcing_function(float* force, const float* guidance_x, const float* guidance_y, const float* bound){

  }

  ConnectedComponentsData compute_connected_components(const float* occ_true){

  }

  std::vector<ClusterInfo> extract_lidar_clusters(const ConnectedComponentsData& cc){

  }

  void label_human_clusters(const float* occ_true){

  }

  void safety_filter(const std::vector<float>& vd){

  }

  // 9 state

  TimingSample timing_{};
  std::chrono::steady_lock::time_point latest_field_timestamp_{};

  std::mutex mpc_mutex;
  MPC3D mpc3d_controller;
  mutable std::shared_mutex field_mutex_;

  const float h0 = 0.0f;
  const float dh0 = 1.0f;
  float wn = 1.0f;
  float issf = 50.0f;

  bool flag = false;
  bool dhdt_flag = false;
  bool save_flag = false;
  bool start_flag = false;
  bool enable_display = false;
  bool sit_flag = false;
  bool stop_flag = false;
  bool predictive_sf_flag = false;
  bool realtime_sf_flag = false;  
  int space_counter = 0;
  int poisson_save_counter = 0;

  const std::vector<char> sorted_paramter_deck = {'1', '2', '3', '4', '5', '6', '0', '0'};
  std::random_device rd;
  std::mt19937 gen;
  std::vector<char> current_parameter_deck;

  std::vector<float> x = {0.0f, 0.0f, 0.0f};
  std::vector<float> xd = {0.0f, 0.0f, 0.0f};
  std::vector<float> xc = {-2.0f, -2.0f, 0.0f};
  std::vector<float> dx = {0.0f, 0.0f, 0.0f};

  std::chrono::steady_clock::time_point t_grid, t_state, t_start;
  float grid_age = 0.0f;
  float dt_grid = 1.0e10f;
  float dt_state = 1.0e10f;
  float t_ms = 0.0f;

  std::vector<float> vt = {0.0f, 0.0f, 0.0f};
  std::vector<float> vn = {0.0f, 0.0f, 0.0f};
  std::vector<float> vd = {0.0f, 0.0f, 0.0f};
  std::vector<float> v = {0.0f, 0.0f, 0.0f};
  std::vector<float> vb = {0.0f, 0.0f, 0.0f};
  float h{} dhdt{}; dhdx{}; dhdy{}; dhdq{};

  float occ1[IMAX *JAMX];
  float occ0[IMAX*JMAX];
  int8_t conf[IMAX *JMAX];
  float grid_temp[IMAX * JMAX];
  float* hgrid1 {};
  float* hgrid0 {};
  float* bound {};
  float* force {};
  float* dhdt_grid {};
  float* robot_kernel_human {};
  float* robot_kernel_obstacle {};
  float* guidance_x_grid {};
  float* guidance_y_grid {};

  float guidance_x_display[IMAX *JMAX];
  float guidance_y_display[IMAX *JMAX];
  float bounded_display[IMAX *JAMX];
  int8_t tangent_layer_display[IMAX *JMAX];

  float robot_length{}, robot_width{};
  float robot_MOS_human{}, robot_MOS_obstacle{};
  int robot_kernel_dim_human{}, robot_kernel_dim_obstacle{};

  rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
  rclcpp::TimerBase::SharedPtr mpc_timer;
  rclcpp::Subscription<std_msgs::msg::Int32>::sharedPtr key_suber_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr occ_grid_suber;
  rclcpp::Substription<nav_msgs::msg::OccupancyGrid>::SharedPtr class_map_suber;
  rclcpp::Substription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibility_map_suber_;
  rclcpp::Substription<nav_msgs::msg::Odometry>::SharedPtr pose_suber_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PiintClound2>> cloud_sub_;

  int8_t class_map[IMAX*JMAX];
  int8_t visibility_map[IMAX*JMAX];
  int8_t class_map_expanded[IMAX*JMAX];
  std::unique_ptr<HumanTracker> human_tracker_;
  int min_yolo_cells_ = 5;
  bool enable_human_tracker_dilation_ = true;
  float dh0_human = 1.0f;
  float dh0_obstacle = 0.3f;
  bool enable_social_navigation_ = false;
  float social_tangent_bias_ = 0.5f;
  int social_tengnet_layers_ = 3;
  int social_layer_thickness_ = 1;
  float current_tangent_direction_ = 1.0f;
  float human_direction_threshold_ = 0.15f;
  std::map<int, std::pair<float, float>> prev_human_distances_;
  std::vector<std::tuple<int, int, float, float, float>> human_boundrary_info_;

  float tight_area_human_threshold = 2.0f;
  float tight_area_h_threshold_ = 0.3f;
  float tight_area_wall_slack = -0.1f;

  float cbf_sigma_epsilon_ = 0.1f;
  float cbf_sigma_kappa = 5.0f;
  float vel_max_x_fwd = 0.9f;
  float vel_max_x_bwd = 0.9f;
  float vel_max_y = 0.9f;
  float vel_max_yaw = 0.8f;

  unitree_api::msg::Request req;
  SportClient sport_req;
  std::ofstream outFileCSV;
  std::ofstream outFileBIN;
  rclcpp::Publisher<<sensor_msgs::msg::Image>::SharedPtr poisson_image_hub_;
  rclcpp::Publisher<std_msgs::msg::float32MultiArray>::SharedPtr logging_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr profiling_data_pub_;
  double logging_publish_hz_ = 10.0;
  double logging_publish_period_ = 0.1;
  std::chrono::steady_clock::time_point last_logging_publish_time_;
  bool enable_data_logging_to_file_ = false;
};

}

int main(int argc, char* argv[]){
  rclcpp::init(argc, argv);

  rclcpp::executors::MultiTreadedExecutor executor;

  auto poissonNide = std::make_shared<ss::PoissonControllerNode>();

  const float min_z = poissonNode->get_paramether("min_z").as_double();
  const float max_z = poissonNode->get_parameter('max_z').as_double();

  RCLCPP_INFO{
    poissonNode->get_logger(),
    "passing min_z%.2f, max_z=%.2f to cloudmergerNode",
    min_z, max_z
  }

  auto mappingNode = std::make_shared<cloundMergerNode>(min_z, max_z);

  executor.add_node(mappingNode);
  executor.add_node(poissonNode);

  try{
    executor.spin();
    throw('terminated')
  }catch(const char* msg){
    rclcpp::shutdown();
    std::cout << msg <, std::endl;
  }
  return 0
}