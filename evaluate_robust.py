import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

# Import your updated environment with the 4-variable action space
from env import AdaptiveCBFEnv 

if __name__ == "__main__":
    print("Loading Robust Adaptive CBF AI...")
    # Update this filename if you saved your new model under a different name
    model = PPO.load("robust_adaptive_cbf_model") 
    env = AdaptiveCBFEnv()
    
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
        }
    ]
    
    # Create a 3x3 grid (Trajectory, Alpha, Epsilon for each scenario)
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(r"Robust CBF Dashboard: Dynamic $\alpha$ and $\epsilon$ Tuning", fontsize=18)
    
    for i, scen in enumerate(scenarios):
        print(f"Running {scen['name']}...")
        obs, info = env.reset()
        
        # Override environment variables for the test
        env.robot_pos = np.array([0.0, 0.0])
        env.obstacle_pos = scen["obs_pos"]
        env.target_pos = scen["target_pos"]
        env.target_radius = scen["target_radius"]
        obs = env._get_obs()
        
        traj_x, traj_y = [], []
        alphas = []
        epsilons = []
        
        for step in range(150):
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            
            action, _ = model.predict(obs, deterministic=True)
            
            # Unpack the new action space
            alpha_val = action[0]
            epsilon_val = action[1]
            
            alphas.append(alpha_val)
            epsilons.append(epsilon_val)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                traj_x.append(env.robot_pos[0])
                traj_y.append(env.robot_pos[1])
                print(f"  Finished at step {step}!")
                break

        # --- PLOTTING ---
        ax_traj = axs[i, 0]
        ax_alpha = axs[i, 1]
        ax_eps = axs[i, 2]
        
        # Panel 1: Trajectory
        ax_traj.set_title(f"{scen['name']} - Trajectory")
        ax_traj.set_xlabel("X Position")
        ax_traj.set_ylabel("Y Position")
        obs_circle = plt.Circle((env.obstacle_pos[0], env.obstacle_pos[1]), env.obstacle_radius, color='red', alpha=0.5)
        ax_traj.add_patch(obs_circle)
        target_circle = plt.Circle((env.target_pos[0], env.target_pos[1]), env.target_radius, color='green', alpha=0.3)
        ax_traj.add_patch(target_circle)
        ax_traj.plot(traj_x, traj_y, color='blue', marker='o', markersize=3)
        ax_traj.set_xlim(-1, 10)
        ax_traj.set_ylim(-5, 5)
        ax_traj.set_aspect('equal', adjustable='box')
        ax_traj.grid(True)

        # Panel 2: Alpha
        ax_alpha.set_title(f"{scen['name']} - Dynamic " + r"$\alpha$")
        ax_alpha.set_xlabel("Time Step")
        ax_alpha.set_ylabel(r"$\alpha$ Value")
        ax_alpha.plot(range(len(alphas)), alphas, color='purple', linewidth=2)
        ax_alpha.axhline(5.0, color='gray', linestyle=':', label='Max (Aggressive)')
        ax_alpha.axhline(0.1, color='gray', linestyle='--', label='Min (Cautious)')
        ax_alpha.set_ylim(0, 5.5)
        ax_alpha.grid(True)
        if i == 0: ax_alpha.legend(loc="lower left")

        # Panel 3: Epsilon
        ax_eps.set_title(f"{scen['name']} - Dynamic " + r"$\epsilon$")
        ax_eps.set_xlabel("Time Step")
        ax_eps.set_ylabel(r"$\epsilon$ Value")
        ax_eps.plot(range(len(epsilons)), epsilons, color='orange', linewidth=2)
        ax_eps.axhline(50.0, color='gray', linestyle=':', label='Max (Strict)')
        ax_eps.axhline(0.1, color='gray', linestyle='--', label='Min (Relaxed)')
        ax_eps.set_ylim(0, 55.0)
        ax_eps.grid(True)
        if i == 0: ax_eps.legend(loc="lower right")

    plt.tight_layout()
    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)
    save_path = os.path.join(eval_dir, "robust_cbf_dashboard.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Done! Multi-scenario graph saved to {save_path}")