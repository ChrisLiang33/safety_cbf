import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env import AdaptiveCBFEnv 

if __name__ == "__main__":
    # --- CONFIGURE YOUR RUN HERE ---
    model_name = "150000_model"  # Change this to whatever model you want to test
    # -------------------------------

    print(f"Loading Adaptive CBF AI from {model_name}...")
    model = PPO.load(f"./model/{model_name}") 
    env = AdaptiveCBFEnv()
    
    # 5 distinct scenarios to prove generalization
    scenarios = [
        {
            "name": "Scenario 1: Center Park",
            "obs_pos": np.array([4.0, 0.1]),
            "target_pos": np.array([9.0, 0.0]),
            "target_radius": 1.0
        },
        {
            "name": "Scenario 2: High Offset",
            "obs_pos": np.array([5.0, -0.5]),
            "target_pos": np.array([8.0, 3.0]),
            "target_radius": 1.5
        },
        {
            "name": "Scenario 3: Tight Low Corner",
            "obs_pos": np.array([3.0, 0.5]),
            "target_pos": np.array([9.0, -2.5]),
            "target_radius": 0.8
        },
        {
            "name": "Scenario 4: Dead Center Block",
            "obs_pos": np.array([5.0, 0.0]),
            "target_pos": np.array([10.0, 0.0]),
            "target_radius": 1.0
        },
        {
            "name": "Scenario 5: Early Dodge",
            "obs_pos": np.array([2.0, 0.0]),
            "target_pos": np.array([8.0, 4.0]),
            "target_radius": 1.5
        }
    ]
    
    # Created a 5x2 grid of subplots (Height increased to 25 to fit 5 rows cleanly)
    fig, axs = plt.subplots(5, 2, figsize=(16, 25))
    fig.suptitle(f"Adaptive CBF Robustness: Dynamic " + r"$\alpha$" + f" ({model_name})", fontsize=18)
    
    for i, scen in enumerate(scenarios):
        print(f"Running {scen['name']}...")
        obs, info = env.reset()
        
        env.robot_pos = np.array([0.0, 0.0])
        env.obstacle_pos = scen["obs_pos"]
        env.target_pos = scen["target_pos"]
        env.target_radius = scen["target_radius"]
        
        env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
        
        obs = env._get_obs()
        
        traj_x, traj_y = [], []
        alphas = []
        distances = []
        total_reward = 0.0
        
        for step in range(150):
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            
            obs_pos_2d = np.array([env.obstacle_pos[0], env.obstacle_pos[1]])
            robot_pos_2d = np.array([env.robot_pos[0], env.robot_pos[1]])
            dist = np.linalg.norm(robot_pos_2d - obs_pos_2d) - env.obstacle_radius
            distances.append(dist)
            
            action, _ = model.predict(obs, deterministic=True)
            alphas.append(action[0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                traj_x.append(env.robot_pos[0])
                traj_y.append(env.robot_pos[1])
                print(f"  Finished at step {step + 1}! Total Reward: {total_reward:.1f}")
                break

        # --- PLOTTING ---
        ax_traj = axs[i, 0]
        ax_alpha = axs[i, 1]
        
        ax_traj.set_title(f"{scen['name']} | Steps: {step + 1} | Reward: {total_reward:.1f}")
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")
        
        obs_circle = plt.Circle((env.obstacle_pos[0], env.obstacle_pos[1]), env.obstacle_radius, color='red', alpha=0.5)
        ax_traj.add_patch(obs_circle)
        
        target_circle = plt.Circle((env.target_pos[0], env.target_pos[1]), env.target_radius, color='green', alpha=0.3)
        ax_traj.add_patch(target_circle)
        
        ax_traj.plot(traj_x, traj_y, color='gray', linestyle='--', alpha=0.5)
        
        sc = ax_traj.scatter(traj_x[:-1], traj_y[:-1], c=alphas, cmap='coolwarm', vmin=0.1, vmax=5.0, s=20, zorder=5)
        
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

        ax_alpha.set_title(f"{scen['name']} - Dynamic " + r"$\alpha$ & Distance")
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
    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Dynamic save path based on the model name
    save_path = os.path.join(eval_dir, f"{model_name}_multi_scenario_dashboard.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Done! Timestamped multi-scenario graph saved to {save_path}")