"""
3-obstacle navigation baseline — NO CBF, NO alpha, NO phi.
Agent controls [kx, ky] only. Pure navigation learning.

The robot moves directly with the policy output: robot_pos += action * dt.
No QP, no safety filter. Collisions are punished by reward only.

Goal: verify that the policy can learn basic obstacle avoidance and target-reaching
with randomized obstacle placement before adding CBF/alpha/phi on top.

Obstacles placed with gentler randomization:
  - Spaced along x-axis in thirds (not fully random)
  - y position randomized within [-8, 8]
  - More structured than fully random, more diverse than forced slalom
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NavigationBaselineEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.obs_radius = 5.0

        # ACTION: [kx, ky] only — direct velocity control
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0], dtype=np.float32),
            high=np.array([3.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: 11 dims — 3 obstacles (rel_x, rel_y) + target (rel_x, rel_y, radius) + velocity
        # No obstacle radius in obs since there's no CBF
        self.observation_space = spaces.Box(
            low=np.array([-120, -25, -120, -25, -120, -25,
                          -120, -25, 0.1, -4, -4], dtype=np.float32),
            high=np.array([120, 25, 120, 25, 120, 25,
                           120, 25, 5.0, 4, 4], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obs_pos = [np.zeros(2) for _ in range(3)]
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0

    def _place_obstacles(self):
        """Gentler randomization: obstacles in 3 x-bands, y randomized."""
        # Divide the corridor into 3 bands along x
        # Band 1: x in [15, 35], Band 2: x in [40, 60], Band 3: x in [65, 85]
        placed = []
        x_bands = [(15.0, 35.0), (40.0, 60.0), (65.0, 85.0)]
        for x_low, x_high in x_bands:
            x = self.np_random.uniform(x_low, x_high)
            y = self.np_random.uniform(-8.0, 8.0)
            placed.append(np.array([x, y]))
        return placed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        placed = self._place_obstacles()
        for i in range(3):
            self.obs_pos[i] = placed[i]

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx = float(action[0])
        ky = float(action[1])
        u = np.array([kx, ky])

        # Direct movement — no CBF, no safety filter
        self.robot_pos += u * self.dt
        self.velocity = u.copy()

        # Distances
        dist2obs = []
        for i in range(3):
            dist2obs.append(np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.obs_radius)
        min_obs_dist = min(dist2obs)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        reward = 0.0

        if min_obs_dist < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0
            reward -= 1.0
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        if self.robot_pos[0] < -5.0 or self.robot_pos[0] > 105.0 or abs(self.robot_pos[1]) > 15.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {
            "u": u, "dist2obs": dist2obs, "min_obs_dist": float(min_obs_dist),
        }

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
