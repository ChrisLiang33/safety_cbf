# add a energy function to the action 
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
        
        # ACTION SPACE: z = [alpha, epsilon, k_x, k_y] (4 values)
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.5, -2.0], dtype=np.float32),
            high=np.array([5.0, 50.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        # OBSERVATION SPACE: [robot_x, robot_y, obs_x, obs_y, obs_radius]
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
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])
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

        return self._get_obs(), reward, terminated, truncated, {"safe_u": safe_u, "h_x": new_hx}

    def _get_obs(self):
        return np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.obstacle_pos[0], self.obstacle_pos[1],
            self.obstacle_radius
        ], dtype=np.float32)

if __name__ == "__main__":
    log_dir = "./cbf_logs/"
    os.makedirs(log_dir, exist_ok=True)

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

    print("Starting training on multiple CPU cores...")
    model.learn(total_timesteps=500000) 
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
    print("Testing the trained model in 3 randomized scenarios...")
    test_env = AdaptiveCBFEnv()

    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)

    for episode in range(3):
        obs, info = test_env.reset()

        # Randomize the physical geometry of the test
        test_env.robot_pos = np.array([np.random.uniform(0.0, 2.0), np.random.uniform(-3.0, 3.0)])
        test_env.obstacle_pos = np.array([np.random.uniform(3.0, 7.0), np.random.uniform(-2.0, 2.0)])
        obs = test_env._get_obs()

        # Clipboards for our new dashboard
        alphas, epsilons, distances = [], [], []
        robot_xs, robot_ys = [], []
        k_nom_xs, k_nom_ys = [], []
        u_safe_xs, u_safe_ys = [], []
        h_xs = []

        obs_x, obs_y = test_env.obstacle_pos[0], test_env.obstacle_pos[1]
        obs_r = test_env.obstacle_radius

        for i in range(150): 
            action, _states = model.predict(obs, deterministic=True)
            alpha, epsilon, k_x, k_y = action
            
            alphas.append(alpha)
            epsilons.append(epsilon)
            k_nom_xs.append(k_x)
            k_nom_ys.append(k_y)
            robot_xs.append(obs[0])
            robot_ys.append(obs[1])
            
            dist = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array([obs[2], obs[3]])) - obs[4]
            distances.append(dist)

            obs, reward, terminated, truncated, info = test_env.step(action)
            
            safe_u = info.get("safe_u", np.array([0.0, 0.0]))
            u_safe_xs.append(safe_u[0])
            u_safe_ys.append(safe_u[1])
            h_xs.append(info.get("h_x", 0.0))
            
            if terminated or truncated:
                print(f"Scenario {episode + 1} finished at step {i+1}!")
                break

        # 6. PLOT THE 4-PANEL DASHBOARD
        print(f"Saving Dashboard for Scenario {episode + 1}...")
        steps = range(len(alphas))
        
        # Create a massive 2x2 grid for our plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Diagnostic Dashboard: Randomized Scenario {episode + 1}", fontsize=16)

        # --- Panel 1: Top Down Trajectory (Color-Mapped by Alpha) ---
        ax1 = axs[0, 0]
        ax1.set_title("Robot Trajectory (Color = Alpha Value)")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        
        # Draw the obstacle
        circle = plt.Circle((obs_x, obs_y), obs_r, color='red', alpha=0.5, label='Obstacle')
        ax1.add_patch(circle)
        
        # Plot the trajectory with a colormap
        scatter = ax1.scatter(robot_xs, robot_ys, c=alphas, cmap='coolwarm', s=20, edgecolor='black', linewidth=0.5)
        fig.colorbar(scatter, ax=ax1, label='Alpha (Aggressiveness)')
        ax1.plot(robot_xs, robot_ys, color='black', linewidth=0.5, alpha=0.5) # connecting line
        ax1.set_xlim(-1, 10)
        ax1.set_ylim(-5, 5)
        ax1.grid(True)

        # --- Panel 2: Dynamic Parameter Tuning ---
        ax2_dist = axs[0, 1]
        ax2_dist.set_title("Parameter Tuning vs. Distance")
        ax2_dist.set_xlabel('Time Step')
        ax2_dist.set_ylabel('Distance to Obstacle edge', color='black')
        ax2_dist.plot(steps, distances, color='black', linestyle='--', label='Distance')
        ax2_dist.axhline(0, color='red', linewidth=1, linestyle=':')
        
        ax2_params = ax2_dist.twinx()
        ax2_params.set_ylabel('Parameter Value')
        ax2_params.plot(steps, alphas, color='blue', label='Alpha')
        ax2_params.plot(steps, epsilons, color='orange', label='Epsilon')
        ax2_dist.grid(True)

        lines_1, labels_1 = ax2_dist.get_legend_handles_labels()
        lines_2, labels_2 = ax2_params.get_legend_handles_labels()
        ax2_params.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

        # --- Panel 3: Nominal vs. Safe Action (X-Axis Velocity) ---
        ax3 = axs[1, 0]
        ax3.set_title("Control Input Override: Forward Velocity (X)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Velocity (m/s)")
        ax3.plot(steps, k_nom_xs, label='Requested Velocity ($k_{nom}$)', color='gray', linestyle='--')
        ax3.plot(steps, u_safe_xs, label='Safe Velocity ($u_{safe}$)', color='green')
        ax3.legend()
        ax3.grid(True)

        # --- Panel 4: The Barrier Function ---
        ax4 = axs[1, 1]
        ax4.set_title("Control Barrier Function Value ($h(x)$)")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("h(x) value")
        ax4.plot(steps, h_xs, label='Safety Boundary $h(x)$', color='purple')
        ax4.axhline(0, color='red', linewidth=1, linestyle=':', label='Crash Line ($h(x) = 0$)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(eval_dir, f"scenario_{episode + 1}_dashboard.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    print(f"All dashboards successfully saved to {eval_dir}!")