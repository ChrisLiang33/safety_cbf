import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

# Import your actual environment so we don't have to rewrite the math
from env import AdaptiveCBFEnv 

if __name__ == "__main__":
    print("Loading Adaptive CBF AI...")
    # Make sure this matches whatever you name your new saved model
    model = PPO.load("adaptive_cbf_model") 
    
    env = AdaptiveCBFEnv()
    obs, info = env.reset()
    
    # 1. Force a strict scenario for a clean evaluation graph
    env.robot_pos = np.array([0.0, 0.0])
    env.obstacle_pos = np.array([4.0, 0.1])
    env.target_pos = np.array([9.0, 0.0])
    env.target_radius = 1.0
    
    # Update the observation array with the forced coordinates
    obs = env._get_obs()
    
    # Tracking arrays
    traj_x, traj_y = [], []
    alphas = []
    
    print("Running dynamic trajectory...")
    for step in range(150):
        # Record current position
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        
        # Ask the AI for its 3-part action
        action, _ = model.predict(obs, deterministic=True)
        alpha_choice = action[0]
        alphas.append(alpha_choice)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Record final position
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            print(f"Episode finished at step {step}!")
            break

    # 2. PLOT THE DYNAMIC DASHBOARD
    print("Generating Dynamic Alpha Dashboard...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(r"Adaptive CBF Dashboard: Dynamic $\alpha$ Tuning", fontsize=16)

    # Panel 1: Trajectory
    ax1 = axs[0]
    ax1.set_title("Robot Trajectory")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    
    # Draw Obstacle
    obs_circle = plt.Circle((env.obstacle_pos[0], env.obstacle_pos[1]), env.obstacle_radius, color='red', alpha=0.5, label='Obstacle')
    ax1.add_patch(obs_circle)
    
    # Draw Target
    target_circle = plt.Circle((env.target_pos[0], env.target_pos[1]), env.target_radius, color='green', alpha=0.3, label='Target Zone')
    ax1.add_patch(target_circle)
    
    ax1.plot(traj_x, traj_y, color='blue', marker='o', markersize=4, label='Robot Path')
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Panel 2: Alpha Brain Waves
    ax2 = axs[1]
    ax2.set_title(r"AI Chosen $\alpha$ over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel(r"$\alpha$ Value")
    ax2.plot(range(len(alphas)), alphas, color='purple', linewidth=2, label=r'Dynamic $\alpha$')
    
    # Show the action space bounds for context
    ax2.axhline(5.0, color='gray', linestyle=':', label='Max Alpha (Aggressive)')
    ax2.axhline(0.1, color='gray', linestyle='--', label='Min Alpha (Cautious)')
    
    ax2.set_ylim(0, 5.5)
    ax2.legend(loc="lower left")
    ax2.grid(True)

    plt.tight_layout()
    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)
    plt.savefig(os.path.join(eval_dir, "dynamic_alpha_dashboard.png"), bbox_inches='tight')
    plt.show()
    print("Done! Graph saved to eval_plots/dynamic_alpha_dashboard.png")