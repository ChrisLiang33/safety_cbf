"""
Exp 56 Phase 1: Navigation-only environment.
Agent controls [kx, ky]. No CBF, no QP, no alpha, no phi.
Just learn to navigate to the target while avoiding obstacles.

Same environment geometry as exp 45/53 (6 m/s, target x=150).
Includes dynamics bias and sensor error so the nav policy is robust.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NavOnlyEnv(gym.Env):
    def __init__(self, systematic_bias_range=(-0.6, 0.4), jitter_range=(-0.2, 0.2),
                 bias_magnitude_range=(0.3, 1.0), obs_radius_range=(3.0, 7.0)):
        super().__init__()
        self.dt = 0.1
        self.systematic_bias_range = systematic_bias_range
        self.jitter_range = jitter_range
        self.bias_magnitude_range = bias_magnitude_range
        self.obs_radius_range = obs_radius_range
        self.true_radius = [5.0, 5.0, 5.0]
        self.estimated_radius = [5.0, 5.0, 5.0]
        self.bias = np.zeros(2)
        self.sensor_bias = 0.0

        # ACTION: [kx, ky] only — no CBF parameters
        self.action_space = spaces.Box(
            low=np.array([-6.0, -6.0], dtype=np.float32),
            high=np.array([6.0, 6.0], dtype=np.float32),
            dtype=np.float32,
        )
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
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0
        self.max_steps = 800
        self.current_step = 0

    def _place_obstacles(self):
        placed = []
        x_bands = [(20.0, 50.0), (60.0, 100.0), (110.0, 140.0)]
        for x_low, x_high in x_bands:
            x = self.np_random.uniform(x_low, x_high)
            y = self.np_random.uniform(-10.0, 10.0)
            placed.append(np.array([x, y]))
        return placed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0

        target_y = self.np_random.uniform(-5.0, 5.0)
        self.target_pos = np.array([150.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        placed = self._place_obstacles()
        for i in range(3):
            self.obs_pos[i] = placed[i]

        for i in range(3):
            self.true_radius[i] = self.np_random.uniform(
                self.obs_radius_range[0], self.obs_radius_range[1])

        self.sensor_bias = self.np_random.uniform(
            self.systematic_bias_range[0], self.systematic_bias_range[1])
        for i in range(3):
            jitter = self.np_random.uniform(self.jitter_range[0], self.jitter_range[1])
            error = self.sensor_bias + jitter
            self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

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

        # No CBF — direct control + bias
        self.robot_pos += u * self.dt + self.bias * self.dt
        self.velocity = u.copy()

        dist2obs_true = [np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.true_radius[i] for i in range(3)]
        min_obs_dist_true = min(dist2obs_true)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        truncated = False
        reward = 0.0
        if min_obs_dist_true < 0:
            reward = -100.0
            terminated = True
        else:
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
