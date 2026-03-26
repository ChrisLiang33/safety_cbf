import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import pandas as pd
import shutil

from env import AdaptiveCBFEnv 

if __name__ == "__main__":
    total_runs = 2  # Run the entire sequence this many times to test seed variance
    timesteps_to_test = [ 150000, 200000, 250000, 300000, 
        400000, 500000, 600000, 700000, 800000, 900000, 1000000
    ]
    
    log_dir = "./cbf_logs/"
    model_dir = "./model/"
    eval_dir = "./eval_plots/"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    scenarios = [
        {"name": "Scenario 1: Center Park", "obs_pos": np.array([4.0, 0.1]), "target_pos": np.array([9.0, 0.0]), "target_radius": 1.0},
        {"name": "Scenario 2: High Offset", "obs_pos": np.array([5.0, -0.5]), "target_pos": np.array([8.0, 3.0]), "target_radius": 1.5},
        {"name": "Scenario 3: Tight Low Corner", "obs_pos": np.array([3.0, 0.5]), "target_pos": np.array([9.0, -2.5]), "target_radius": 0.8},
        {"name": "Scenario 4: Dead Center Block", "obs_pos": np.array([5.0, 0.0]), "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
        {"name": "Scenario 5: Early Dodge", "obs_pos": np.array([2.0, 0.0]), "target_pos": np.array([8.0, 4.0]), "target_radius": 1.5}
    ]

    for run_id in range(3, total_runs + 3):
        print("\n" + "#"*60)
        print(f"🔥 STARTING MASTER RUN {run_id} OF {total_runs} 🔥")
        print("#"*60)

        for ts in timesteps_to_test:
            print("\n" + "="*50)
            print(f"🚀 RUN {run_id} | EXPERIMENT: {ts} TIMESTEPS")
            print("="*50)

            # ---------------------------------------------------------
            # PHASE 1: TRAINING FROM SCRATCH
            # ---------------------------------------------------------
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)

            n_envs = 8
            vec_env = make_vec_env(AdaptiveCBFEnv, n_envs=n_envs, vec_env_cls=DummyVecEnv, monitor_dir=log_dir)

            print(f"[Run {run_id}] Training new model for {ts} steps...")
            model = PPO("MlpPolicy", vec_env, verbose=0, device="cuda") 
            model.learn(total_timesteps=ts) 
            
            # Save format: run1_150000_model.zip
            model_path = os.path.join(model_dir, f"run{run_id}_{ts}_model")
            model.save(model_path)
            print(f"Model saved to {model_path}.zip")

            # ---------------------------------------------------------
            # PHASE 2: PLOT REWARD CURVE
            # ---------------------------------------------------------
            dataframes = []
            for file in os.listdir(log_dir):
                if file.endswith("monitor.csv"):
                    df_part = pd.read_csv(os.path.join(log_dir, file), skiprows=1)
                    df_part['timestep'] = df_part['l'].cumsum()
                    dataframes.append(df_part)
                    
            if dataframes:
                df = pd.concat(dataframes).sort_values(by='timestep').reset_index(drop=True)
                
                fig_reward = plt.figure(figsize=(10, 4))
                window_size = max(10, int(len(df) * 0.05)) 
                plt.plot(df['timestep'], df['r'].rolling(window=window_size).mean(), label='Rolling Average Reward', color='green')
                
                plt.title(f"Reward Curve (Run {run_id} - {ts} Timesteps)")
                plt.xlabel("Total Training Timesteps (Per Core)")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid()
                
                reward_save_path = os.path.join(eval_dir, f"run{run_id}_{ts}_reward_curve.png")
                plt.savefig(reward_save_path, bbox_inches='tight')
                plt.close(fig_reward) 

            # ---------------------------------------------------------
            # PHASE 3: MULTI-SCENARIO EVALUATION
            # ---------------------------------------------------------
            print(f"[Run {run_id}] Running scenario evaluations for {ts}_model...")
            eval_env = AdaptiveCBFEnv()
            
            fig_eval, axs = plt.subplots(5, 2, figsize=(16, 25))
            fig_eval.suptitle(f"Adaptive CBF Robustness: Dynamic " + r"$\alpha$" + f" (Run {run_id} | {ts} Steps)", fontsize=18)
            
            for i, scen in enumerate(scenarios):
                obs, info = eval_env.reset()
                
                eval_env.robot_pos = np.array([0.0, 0.0])
                eval_env.obstacle_pos = scen["obs_pos"]
                eval_env.target_pos = scen["target_pos"]
                eval_env.target_radius = scen["target_radius"]
                eval_env.prev_dist2target = np.linalg.norm(eval_env.robot_pos - eval_env.target_pos)
                
                obs = eval_env._get_obs()
                
                traj_x, traj_y = [], []
                alphas, distances = [], []
                total_reward = 0.0
                
                for step in range(150):
                    traj_x.append(eval_env.robot_pos[0])
                    traj_y.append(eval_env.robot_pos[1])
                    
                    obs_pos_2d = np.array([eval_env.obstacle_pos[0], eval_env.obstacle_pos[1]])
                    robot_pos_2d = np.array([eval_env.robot_pos[0], eval_env.robot_pos[1]])
                    dist = np.linalg.norm(robot_pos_2d - obs_pos_2d) - eval_env.obstacle_radius
                    distances.append(dist)
                    
                    action, _ = model.predict(obs, deterministic=True)
                    alphas.append(action[0])
                    
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        traj_x.append(eval_env.robot_pos[0])
                        traj_y.append(eval_env.robot_pos[1])
                        break

                ax_traj = axs[i, 0]
                ax_alpha = axs[i, 1]
                
                ax_traj.set_title(f"[Run {run_id} - {ts} Steps] {scen['name']} | Steps: {step + 1} | Reward: {total_reward:.1f}")
                ax_traj.set_xlabel("X Position")
                ax_traj.set_ylabel("Y Position")
                
                obs_circle = plt.Circle((eval_env.obstacle_pos[0], eval_env.obstacle_pos[1]), eval_env.obstacle_radius, color='red', alpha=0.5)
                ax_traj.add_patch(obs_circle)
                
                target_circle = plt.Circle((eval_env.target_pos[0], eval_env.target_pos[1]), eval_env.target_radius, color='green', alpha=0.3)
                ax_traj.add_patch(target_circle)
                
                ax_traj.plot(traj_x, traj_y, color='gray', linestyle='--', alpha=0.5)
                sc = ax_traj.scatter(traj_x[:len(alphas)], traj_y[:len(alphas)], c=alphas, cmap='coolwarm', vmin=0.1, vmax=5.0, s=20, zorder=5)
                
                time_interval = 10
                for t in range(0, len(traj_x)-1, time_interval):
                    ax_traj.plot(traj_x[t], traj_y[t], marker='s', color='black', markersize=4, zorder=6)
                    ax_traj.text(traj_x[t], traj_y[t] + 0.3, f"t={t}", fontsize=8, color='black', ha='center', zorder=7)
                
                ax_traj.set_xlim(-1, 10)
                ax_traj.set_ylim(-5, 5)
                ax_traj.set_aspect('equal', adjustable='box')
                ax_traj.grid(True)
                
                cbar = plt.colorbar(sc, ax=ax_traj, fraction=0.046, pad=0.04)
                cbar.set_label(r'$\alpha$ Value')

                ax_alpha.set_title(f"[Run {run_id} - {ts} Steps] {scen['name']} - Dynamic " + r"$\alpha$ & Distance")
                ax_alpha.set_xlabel("Time Step")
                ax_alpha.set_ylabel(r"$\alpha$ Value", color='purple')
                ax_alpha.plot(range(len(alphas)), alphas, color='purple', linewidth=2, label=r'$\alpha$ Value')
                ax_alpha.tick_params(axis='y', labelcolor='purple')
                ax_alpha.set_ylim(0, 5.5)
                
                ax_dist = ax_alpha.twinx()
                ax_dist.set_ylabel('Distance to Obstacle Surface (m)', color='darkorange')
                ax_dist.plot(range(len(distances)), distances, color='darkorange', linestyle='-.', linewidth=2, label='Distance to Obstacle')
                ax_dist.tick_params(axis='y', labelcolor='darkorange')
                ax_dist.set_ylim(0, max(distances) + 1)
                
                lines_1, labels_1 = ax_alpha.get_legend_handles_labels()
                lines_2, labels_2 = ax_dist.get_legend_handles_labels()
                if i == 0: 
                    ax_alpha.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left")
                ax_alpha.grid(True)

            plt.tight_layout()
            eval_save_path = os.path.join(eval_dir, f"run{run_id}_{ts}_dt0.05_multi_scenario_dashboard.png")
            plt.savefig(eval_save_path, bbox_inches='tight')
            plt.close(fig_eval) 
            
            print(f"✅ Finished Run {run_id} for {ts} steps! Plots saved.")

    print("\n🎉 ALL DOUBLE-RUN EXPERIMENTS COMPLETE! 🎉")