"""
Exp 58 Phase 1: Easy navigation environment — NO OBSTACLES.
Agent controls [kx, ky]. Learns to go straight to target.

No obstacles, no CBF. The nav policy will learn to beeline
toward the target. When frozen and placed on a hard map (Phase 2),
it will drive straight at obstacles, forcing the CBF to intervene.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NavEasyEnv(gym.Env):
    def __init__(self, bias_magnitude_range=(0.3, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.bias_magnitude_range = bias_magnitude_range
        self.bias = np.zeros(2)

        # ACTION: [kx, ky] only
        self.action_space = spaces.Box(
            low=np.array([-6.0, -6.0], dtype=np.float32),
            high=np.array([6.0, 6.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Same observation space as Phase 2 — 3 obstacles + target + velocity
        # During Phase 1, obstacles are placed far away (effectively invisible)
        self.observation_space = spaces.Box(
            low=np.array([-180, -35, 0, -180, -35, 0, -180, -35, 0,
                          -180, -35, 0.1, -7, -7], dtype=np.float32),
            high=np.array([180, 35, 10, 180, 35, 10, 180, 35, 10,
                           180, 35, 5.0, 7, 7], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obs_pos = [np.zeros(2) for _ in range(3)]
        self.estimated_radius = [5.0, 5.0, 5.0]
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0
        self.max_steps = 800
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0

        target_y = self.np_random.uniform(-5.0, 5.0)
        self.target_pos = np.array([150.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        # Place obstacles FAR away — effectively no obstacles
        for i in range(3):
            self.obs_pos[i] = np.array([500.0 + i * 100, 500.0])
            self.estimated_radius[i] = 5.0

        bias_mag = self.np_random.uniform(
            self.bias_magnitude_range[0], self.bias_magnitude_range[1])
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_mag * np.array([np.cos(angle), np.sin(angle)])

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx, ky = float(action[0]), float(action[1])
        u = np.clip(np.array([kx, ky]), -6.0, 6.0)
        self.current_step += 1

        self.robot_pos += u * self.dt + self.bias * self.dt
        self.velocity = u.copy()

        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        truncated = False
        progress = self.prev_dist2target - dist2target
        reward = progress * 50.0 - 1.0
        self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True
        if self.robot_pos[0] < -5 or self.robot_pos[0] > 160 or abs(self.robot_pos[1]) > 20:
            reward -= 50.0
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
