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

        # ACTION SPACE: [alpha, epsilon, k_x, k_y] (4 values)
        # alpha bounds: [0.1, 5.0]
        # epsilon bounds: [1.0, 50.0] (Keep strictly > 0 to avoid division by zero)
        self.action_space = spaces.Box(
            low=np.array([0.1, 1.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([5.0, 50.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32    
        )

        # OBSERVATION: [robot_x, robot_y, obs_x, obs_y, obs_radius, target_x, target_y, target_radius]
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, -10.0, 0.0, -10.0, -10.0, 0.1], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 3.0], dtype=np.float32),
            dtype=np.float32
        )

        self.target_pos = np.zeros(2)
        self.target_radius = 1.0
        self.prev_dist2target = 0.0
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
        # Randomize target Y position and target radius
        target_y = self.np_random.uniform(-4.0, 4.0)
        self.target_pos = np.array([9.0, target_y])
        self.target_radius = self.np_random.uniform(0.5, 2.0)
        
        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        info = {}
        return self._get_obs(), info

    def step(self, action):
        """Applies the RL action to tune the CBF, solves the QP, and moves the robot."""
        alpha, epsilon, k_x, k_y = action
        k_nom = np.array([k_x, k_y])

        # 2. Calculate CBF constraints based on current position
        robot_pos_2d = np.array([self.robot_pos[0], self.robot_pos[1]])
        obs_pos_2d = np.array([self.obstacle_pos[0], self.obstacle_pos[1]])
        
        # x_diff = robot_pos_2d - obs_pos_2d
        # h_x = np.sum(x_diff**2) - (self.obstacle_radius**2)
        # L_g_h = 2 * x_diff
        # norm_L_g_h_sq = np.sum(L_g_h**2)

        #new
        x_diff = robot_pos_2d - obs_pos_2d
        dist = np.linalg.norm(x_diff)
        h_x = dist - self.obstacle_radius
        # The derivative of linear distance is the unit vector (length exactly 1.0)
        if dist > 0.001: 
            L_g_h = x_diff / dist
        else:
            L_g_h = np.array([1.0, 0.0])
        norm_L_g_h_sq = np.sum(L_g_h**2)
        #^^^^


        u = cp.Variable(2)
        # The Cost: Minimize 0.5 * ||u - k(x)||^2
        cost = cp.Minimize(0.5 * cp.sum_squares(u - k_nom))

        constraints = [
            L_g_h @ u >= -alpha * h_x + (norm_L_g_h_sq / epsilon)
        ]
        
        # constraints = [A @ u >= b]
        prob = cp.Problem(cost, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            safe_u = u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])
            
        safe_u = np.clip(safe_u, -2.0, 2.0)

        # 4. Apply physics using the SAFE velocity, not the requested velocity
        self.robot_pos += safe_u * self.dt

        # 5. Calculate distances for the reward function after moving
        new_robot_pos_2d = np.array([self.robot_pos[0], self.robot_pos[1]])
        dist2obstacle = np.linalg.norm(new_robot_pos_2d - obs_pos_2d) - self.obstacle_radius
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        
        terminated = False
        reward = 0.0

        if dist2obstacle < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target 
            reward = progress * 50.0
            # Penalize the y slightly steering to encourage smooth nominal driving
            reward -= (abs(k_nom[1]) * 0.1) * self.dt

            # epsilon_penalty = 5.0 * (1.0/epsilon)
            # reward -= epsilon_penalty
            self.prev_dist2target = dist2target
        
        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True
        
        if self.robot_pos[0] < -2.0 or self.robot_pos[0] > 11.0 or abs(self.robot_pos[1]) > 5.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {"safe_u": safe_u, "h_x": float(dist2obstacle), "alpha": float(alpha)}
    
    def _get_obs(self):
        return np.array([
            float(self.robot_pos[0]), 
            float(self.robot_pos[1]),
            float(self.obstacle_pos[0]), 
            float(self.obstacle_pos[1]),
            float(self.obstacle_radius),
            float(self.target_pos[0]), 
            float(self.target_pos[1]),
            float(self.target_radius)
        ], dtype=np.float32)
    


        