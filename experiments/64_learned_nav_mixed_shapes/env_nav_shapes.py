"""
Exp 64 Phase 1: Navigation with mixed obstacle shapes.
Agent controls [kx, ky]. No CBF.
Obstacles can be circles, rectangles, or line segments.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

CIRCLE = 0
RECTANGLE = 1
LINE = 2


def sdf_circle(p, center, radius):
    return np.linalg.norm(p - center) - radius

def sdf_rectangle(p, center, half_w, half_h):
    d = np.abs(p - center) - np.array([half_w, half_h])
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = min(max(d[0], d[1]), 0.0)
    return outside + inside

def sdf_line(p, a, b, thickness):
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8), 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest) - thickness


class NavMixedShapesEnv(gym.Env):
    def __init__(self, bias_magnitude_range=(0.3, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.bias_magnitude_range = bias_magnitude_range
        self.bias = np.zeros(2)
        self.n_obs = 4
        self.obstacles = []
        self.effective_radius = []

        self.action_space = spaces.Box(
            low=np.array([-6.0, -6.0], dtype=np.float32),
            high=np.array([6.0, 6.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-180, -35, 0] * self.n_obs + [-180, -35, 0.1, -7, -7], dtype=np.float32),
            high=np.array([180, 35, 15] * self.n_obs + [180, 35, 5.0, 7, 7], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0
        self.max_steps = 800
        self.current_step = 0

    def _generate_obstacles(self):
        obstacles = []
        x_bands = [(20.0, 50.0), (55.0, 85.0), (90.0, 120.0), (125.0, 145.0)]
        for x_low, x_high in x_bands:
            cx = self.np_random.uniform(x_low, x_high)
            cy = self.np_random.uniform(-10.0, 10.0)
            shape = self.np_random.integers(0, 3)
            if shape == CIRCLE:
                r = self.np_random.uniform(3.0, 7.0)
                obstacles.append({"type": CIRCLE, "center": np.array([cx, cy]),
                                  "radius": r, "effective_radius": r})
            elif shape == RECTANGLE:
                hw = self.np_random.uniform(2.0, 5.0)
                hh = self.np_random.uniform(2.0, 5.0)
                obstacles.append({"type": RECTANGLE, "center": np.array([cx, cy]),
                                  "half_w": hw, "half_h": hh, "effective_radius": max(hw, hh)})
            else:
                length = self.np_random.uniform(6.0, 12.0)
                angle = self.np_random.uniform(0, np.pi)
                dx = length/2 * np.cos(angle)
                dy = length/2 * np.sin(angle)
                a = np.array([cx - dx, cy - dy])
                b = np.array([cx + dx, cy + dy])
                thickness = self.np_random.uniform(0.5, 1.5)
                obstacles.append({"type": LINE, "center": np.array([cx, cy]),
                                  "a": a, "b": b, "thickness": thickness,
                                  "effective_radius": length/2 + thickness})
        return obstacles

    def _obs_sdf_true(self, p, obs):
        if obs["type"] == CIRCLE:
            return sdf_circle(p, obs["center"], obs["radius"])
        elif obs["type"] == RECTANGLE:
            return sdf_rectangle(p, obs["center"], obs["half_w"], obs["half_h"])
        else:
            return sdf_line(p, obs["a"], obs["b"], obs["thickness"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0

        self.target_pos = np.array([150.0, self.np_random.uniform(-5.0, 5.0)])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        self.obstacles = self._generate_obstacles()
        self.effective_radius = [obs["effective_radius"] for obs in self.obstacles]

        bias_mag = self.np_random.uniform(*self.bias_magnitude_range)
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

        true_sdfs = [self._obs_sdf_true(self.robot_pos, obs) for obs in self.obstacles]
        min_sdf = min(true_sdfs)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        truncated = False
        reward = 0.0
        if min_sdf < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0 - 1.0
            self.prev_dist2target = dist2target
        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True
        if self.robot_pos[0] < -5 or self.robot_pos[0] > 165 or abs(self.robot_pos[1]) > 22:
            reward -= 50.0
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    def _get_obs(self):
        parts = []
        for i, obs in enumerate(self.obstacles):
            rel = obs["center"] - self.robot_pos
            parts.extend([rel[0], rel[1], self.effective_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
