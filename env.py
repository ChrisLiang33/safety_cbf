# add a energy function to the action 
# what is the alpha and epsilon changes 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class AdaptiveCBFEnv(gym.Env):
    """
    Custom Environment for tuning CBF parameters on a 2D single integrator.
    """
    def __init__(self):
        # Modern Python 3 super() call
        super().__init__()
        
        self.dt = 0.1 # Time step for the single integrator physics
        
        # ---------------------------------------------------------
        # ACTION SPACE: z = [alpha, epsilon, k_x, k_y] (4 values)
        # Note: We set the lower bound of epsilon to 0.001 to avoid dividing by zero
        # ---------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([0.1, 5.0, 0.5, -5.0], dtype=np.float32),
            high=np.array([10.0, 50.0, 5.0, 5.0], dtype=np.float32),
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

        # Initialize state variables
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
        
        return self._get_obs(), {}

    def step(self, action):
        """Applies the RL action to tune the CBF, solves the QP, and moves the robot."""
        # 1. Unpack the RL action
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

        if self.robot_pos[1] > 10.0 or self.robot_pos[1] < -10.0:
            reward = -50.0
            terminated = True
        
        if new_hx < 0:
            reward = -100.0 # Crash penalty
            terminated = True
        else:
            reward = safe_u[0] * self.dt # Reward for moving right

        if self.robot_pos[0] > 9.0:
            reward += 50.0 # Success bonus for crossing the scene
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Constructs the observation array."""
        return np.array([
            self.robot_pos[0], self.robot_pos[1],
            self.obstacle_pos[0], self.obstacle_pos[1],
            self.obstacle_radius
        ], dtype=np.float32)


# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from stable_baselines3.common.monitor import Monitor

    # 1. Setup a log directory to save the training data
    log_dir = "./cbf_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Wrap the environment in a Monitor to track learning progress
    env = AdaptiveCBFEnv()
    env = Monitor(env, log_dir)
    check_env(env)
    print("Environment check passed! It is ready for SB3.")

    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=0) 

    # 3. Train the Agent
    print("Starting training (this will take 15-30 seconds)...")
    model.learn(total_timesteps=50000) 
    print("Training finished!")

    # 4. Extract and Plot the Learning Curve
    print("Plotting Learning Curve...")
    # Monitor saves a CSV with a header row we need to skip
    df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
    
    plt.figure(figsize=(10, 4))
    # We use a rolling average to smooth out the noisy RL reward graph
    plt.plot(df['r'].rolling(window=50).mean(), label='Rolling Average Reward (50 eps)', color='green')
    plt.title("RL Training: Reward Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.show()

    # 5. Test the trained Agent and Record Parameters
    print("Testing the trained model to track alpha and epsilon...")
    obs, info = env.reset()

    # Lists to store the data for our final plot
    alphas = []
    epsilons = []
    distances = []

    for i in range(150): 
        action, _states = model.predict(obs, deterministic=True)
        
        # Save the agent's chosen parameters and the current distance
        alpha, epsilon, k_x, k_y = action
        alphas.append(alpha)
        epsilons.append(epsilon)
        
        # Calculate distance to obstacle using observation array
        robot_pos = np.array([obs[0], obs[1]])
        obs_pos = np.array([obs[2], obs[3]])
        dist = np.linalg.norm(robot_pos - obs_pos) - obs[4] # Distance to edge of circle
        distances.append(dist)

        # Step the physics environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished at step {i+1}!")
            break

    # 6. Plot the Dynamic Parameter Tuning
    print("Plotting Alpha and Epsilon Tuning...")
    steps = range(len(alphas))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Distance on the primary Y-axis
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Distance to Obstacle', color='black')
    ax1.plot(steps, distances, color='black', linestyle='--', label='Distance')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.axhline(0, color='red', linewidth=1, linestyle=':') # The crash line

    # Create a secondary Y-axis for the tuning parameters
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Parameter Value')
    ax2.plot(steps, alphas, color='blue', label='Alpha (Aggressiveness)')
    ax2.plot(steps, epsilons, color='orange', label='Epsilon (Tolerance)')
    
    plt.title("Dynamic Parameter Tuning During Obstacle Avoidance")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.grid()
    plt.show()