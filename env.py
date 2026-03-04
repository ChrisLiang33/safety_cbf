# add a energy function to the action 
# what is the alpha and epsilon changes 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
import os
import pandas as pd

class AdaptiveCBFEnv(gym.Env):
    """
    Custom Environment for tuning CBF parameters on a 2D single integrator.
    """
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        
        # ---------------------------------------------------------
        # ACTION SPACE: z = [alpha, epsilon, k_x, k_y] (4 values)
        # Note: We set the lower bound of epsilon to 0.001 to avoid dividing by zero
        # ---------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.5, -2.0], dtype=np.float32),
            high=np.array([5.0, 50.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # OBSERVATION SPACE: [robot_x, robot_y, obs_x, obs_y, obs_radius]
        # ---------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, -10.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0, 10.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        self.robot_pos = np.zeros(2)
        self.obstacle_pos = np.zeros(2)
        self.obstacle_radius = 1.0

    def reset(self, seed=None, options=None):
        """Resets the environment and randomizes the obstacle."""
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])

        # Domain Randomization: Place obstacle randomly in front of the robot
        self.obstacle_pos = np.array([
            self.np_random.uniform(2.0, 8.0), 
            self.np_random.uniform(-2.0, 2.0) 
        ])
        info = {}
        return self._get_obs(), info

    def step(self, action):
        """Applies the RL action to tune the CBF, solves the QP, and moves the robot."""

        alpha, epsilon, k_x, k_y = action
        k_nom = np.array([k_x, k_y])

        # 2. Calculate State Variables for the CBF
        x_diff = self.robot_pos - self.obstacle_pos
        h_x = np.sum(x_diff**2) - (self.obstacle_radius**2)
        
        # 3. Calculate Lie Derivatives for Single Integrator
        L_g_h = 2 * x_diff
        norm_L_g_h_sq = np.sum(L_g_h**2)
        # 4. Set up and solve the Quadratic Program (TISSf-QP)
        u = cp.Variable(2)
        # The Constraint: L_g_h * u >= -alpha * h_x + (||L_g_h||^2 / epsilon)
        A = L_g_h
        b = -alpha * h_x + (norm_L_g_h_sq / epsilon)
        
        # The Cost: Minimize 0.5 * ||u - k(x)||^2
        cost = cp.Minimize(0.5 * cp.sum_squares(u - k_nom))
        constraints = [A @ u >= b]
        prob = cp.Problem(cost, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            safe_u = u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0]) # Fallback if math is infeasible
        except Exception:
            safe_u = np.array([0.0, 0.0]) # Fallback if solver crashes

        safe_u = np.clip(safe_u, -2.0, 2.0)
        # 5. Apply physics using the safe, filtered velocity
        self.robot_pos += safe_u * self.dt
        
        # 6. Calculate Reward & Check Failures
        reward = 0.0
        terminated = False
        truncated = False
        
        # Recalculate distance after moving to check for crashes
        dist_sq = np.sum((self.robot_pos - self.obstacle_pos)**2)
        new_hx = dist_sq - (self.obstacle_radius**2)

        if self.robot_pos[1] > 10.0 or self.robot_pos[1] < -10.0 or self.robot_pos[0] < -5.0:
            reward = -50.0
            terminated = True
        
        elif new_hx < 0:
            reward = -100.0 # Crash penalty
            terminated = True
        else:
            reward = safe_u[0] * self.dt # Reward for moving right
            epsilon_pentalty = 0.05 * (1.0 / epsilon)
            reward -= epsilon_pentalty

        if self.robot_pos[0] > 9.0:
            reward += 50.0 # Success bonus for crossing the scene
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.obstacle_pos[0], self.obstacle_pos[1],
            self.obstacle_radius
        ], dtype=np.float32)


# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================

# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # 1. Setup a log directory to save the training data
    log_dir = "./cbf_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Check CPU cores and setup Multiprocessing
    total_cores = os.cpu_count()
    print(f"Detected {total_cores} CPU cores on this machine.")
    n_envs = max(1, total_cores - 2) 
    print(f"Spinning up {n_envs} parallel environments...")

    # Create the parallel vectorized environment
    # monitor_dir automatically wraps each core in a Monitor to log rewards
    vec_env = make_vec_env(
        AdaptiveCBFEnv, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv, 
        monitor_dir=log_dir
    )

    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")  

    # 3. Train the Agent
    print("Starting training on multiple CPU cores...")
    model.learn(total_timesteps=500000) 
    print("Training finished!")

    # 4. Extract and Plot the Learning Curve
    print("Plotting Learning Curve...")
    dataframes = []
    
    # Loop through all the core logs and combine them
    for file in os.listdir(log_dir):
        if file.endswith("monitor.csv"):
            df_part = pd.read_csv(os.path.join(log_dir, file), skiprows=1)
            
            # The 'l' column is the length of the episode. 
            # cumsum() calculates the exact timestep this episode finished.
            df_part['timestep'] = df_part['l'].cumsum()
            dataframes.append(df_part)
            
    if dataframes:
        # Combine all the cores into one dataset
        df = pd.concat(dataframes)
        
        # Sort every episode from all 22 cores chronologically by the new timestep column
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
    print("Testing the trained model in 3 randomized scenarios...")
    test_env = AdaptiveCBFEnv()

    # Create a new folder to hold your evaluation graphs
    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)

    # Loop 3 times to generate 3 separate test runs
    for episode in range(3):
        obs, info = test_env.reset()

        # OVERRIDE: Randomize the physical geometry of the test
        test_env.robot_pos = np.array([np.random.uniform(0.0, 2.0), np.random.uniform(-3.0, 3.0)])
        test_env.obstacle_pos = np.array([np.random.uniform(3.0, 7.0), np.random.uniform(-2.0, 2.0)])
        
        # Update the observation array
        obs = test_env._get_obs()

        alphas = []
        epsilons = []
        distances = []

        for i in range(150): 
            action, _states = model.predict(obs, deterministic=True)
            alpha, epsilon, k_x, k_y = action
            
            alphas.append(alpha)
            epsilons.append(epsilon)
            
            robot_pos = np.array([obs[0], obs[1]])
            obs_pos = np.array([obs[2], obs[3]])
            dist = np.linalg.norm(robot_pos - obs_pos) - obs[4]
            distances.append(dist)

            obs, reward, terminated, truncated, info = test_env.step(action)
            
            if terminated or truncated:
                print(f"Scenario {episode + 1} finished at step {i+1}!")
                break

        # 6. Plot and SAVE the Dynamic Parameter Tuning
        print(f"Saving Scenario {episode + 1} graph...")
        steps = range(len(alphas))
        
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Distance to Obstacle', color='black')
        ax1.plot(steps, distances, color='black', linestyle='--', label='Distance')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.axhline(0, color='red', linewidth=1, linestyle=':')

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Parameter Value')
        ax2.plot(steps, alphas, color='blue', label='Alpha (Aggressiveness)')
        ax2.plot(steps, epsilons, color='orange', label='Epsilon (Tolerance)')
        
        plt.title(f"Dynamic Tuning: Randomized Scenario {episode + 1}")
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))
        plt.grid()
        
        # Save the figure to your new folder instead of showing it
        save_path = os.path.join(eval_dir, f"scenario_{episode + 1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        
        # Close the plot in the background so it doesn't eat up your RAM
        plt.close()
        
    print(f"All graphs successfully saved to {eval_dir}!")