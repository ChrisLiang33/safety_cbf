from env import AdaptiveCBFEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import pandas as pd

if __name__ == "__main__":
    log_dir = "./cbf_logs/"
    os.makedirs(log_dir, exist_ok=True)

    n_envs = 8
    print(f"Spinning up {n_envs} parallel environments...")

    # Create the parallel vectorized environment
    # monitor_dir automatically wraps each core in a Monitor to log rewards
    vec_env = make_vec_env(
        AdaptiveCBFEnv, 
        n_envs=n_envs, 
        vec_env_cls=DummyVecEnv, 
        monitor_dir=log_dir
    )

    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")  

    print("Starting training on multiple CPU cores...")
    model.learn(total_timesteps=1000000) 
    print("Training finished!")

    print("Plotting Learning Curve...")
    dataframes = []
    
    for file in os.listdir(log_dir):
        if file.endswith("monitor.csv"):
            df_part = pd.read_csv(os.path.join(log_dir, file), skiprows=1)
            
            df_part['timestep'] = df_part['l'].cumsum()
            dataframes.append(df_part)
            
    if dataframes:
        df = pd.concat(dataframes)
        
        df = df.sort_values(by='timestep').reset_index(drop=True)
        
        plt.figure(figsize=(10, 4))
        
        # Because we have 500,000 steps of data, we increase the rolling window to 500
        # to get a beautifully smooth, rising trend line.
        plt.plot(df['timestep'], df['r'].rolling(window=500).mean(), label='Rolling Average Reward', color='green')
        
        plt.title("RL Training: Chronological Reward Learning Curve")
        plt.xlabel("Total Training Timesteps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.show()

    # 5. Test the trained Agent in 3 Randomized Scenarios
    print('testing pure rl agent')
    test_env = AdaptiveCBFEnv()

    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)

    for episode in range(3):
        obs, info = test_env.reset()

        test_env.robot_pos = np.array([0.0,0.0])
        if episode == 0:
            test_env.obstacle_pos = np.array([5.0, 5.0])
        else:
            test_env.obstacle_pos = np.array([5.0, 0.0])
        
        obs = test_env._get_obs()

        robot_xs, robot_ys = [], []
        k_xs, k_ys = [], []
        obs_x, obs_y = test_env.obstacle_pos[0], test_env.obstacle_pos[1]
        obs_r = test_env.obstacle_radius

        for i in range(150):
            action, _states = model.predict(obs, deterministic=True)
            k_x, k_y = action[0], action[1]

            k_xs.append(k_x)
            k_ys.append(k_y)
            robot_xs.append(obs[0])
            robot_ys.append(obs[1])

            obs, reward, terminated, truncated, info = test_env.step(action)

            if terminated or truncated:
                print(f"Scenario {episode + 1} finished at step {i+1}!")
                break

        # 6. PLOT THE 2-PANEL PURE RL DASHBOARD
        print(f"Saving Dashboard for Scenario {episode + 1}...")
        steps = range(len(k_xs))
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Pure RL Dashboard: Scenario {episode + 1}", fontsize=16)

        # Panel 1: Trajectory
        ax1 = axs[0]
        ax1.set_title("Robot Trajectory")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        circle = plt.Circle((obs_x, obs_y), obs_r, color='red', alpha=0.5)
        ax1.add_patch(circle)
        ax1.plot(robot_xs, robot_ys, color='blue', marker='o', markersize=4)
        ax1.set_xlim(-1, 10)
        ax1.set_ylim(-5, 5)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True)

        # Panel 2: Velocity Actions
        ax3 = axs[1]
        ax3.set_title("Agent Requested Velocity")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Velocity (m/s)")
        ax3.plot(steps, k_xs, label='Forward Vel ($k_x$)', color='green')
        ax3.plot(steps, k_ys, label='Lateral Vel ($k_y$)', color='orange')
        ax3.axhline(2.0, color='gray', linestyle=':', label='Max Speed')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        save_path = os.path.join(eval_dir, f"pure_rl_scenario_{episode + 1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    print(f"All dashboards successfully saved to {eval_dir}!")

    print("Training finished!")
    # ADD THIS LINE:
    model.save("adaptive_cbf_model")