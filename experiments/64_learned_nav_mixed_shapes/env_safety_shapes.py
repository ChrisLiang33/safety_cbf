"""
Exp 64 Phase 2: Safety filter with mixed obstacle shapes.
Frozen nav policy provides kx, ky.
Agent controls [alpha, phi] ONLY.
SHIELD SDF pipeline with general shape SDFs.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO

LAMBDA_H = 1.0
GAMMA_H = 0.5

CIRCLE = 0
RECTANGLE = 1
LINE = 2


def sdf_circle(p, center, radius):
    return np.linalg.norm(p - center) - radius

def sdf_rectangle(p, center, half_w, half_h):
    d = np.abs(p - center) - np.array([half_w, half_h])
    return np.linalg.norm(np.maximum(d, 0.0)) + min(max(d[0], d[1]), 0.0)

def sdf_line(p, a, b, thickness):
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab)) - thickness

def sdf_gradient_circle(p, center):
    diff = p - center
    return diff / max(np.linalg.norm(diff), 1e-6)

def sdf_gradient_rectangle(p, center, half_w, half_h):
    d = np.abs(p - center) - np.array([half_w, half_h])
    if d[0] > 0 and d[1] > 0:
        corner = np.sign(p - center) * np.maximum(d, 0.0)
        return corner / max(np.linalg.norm(corner), 1e-6)
    elif d[0] > d[1]:
        return np.array([np.sign(p[0] - center[0]), 0.0])
    else:
        return np.array([0.0, np.sign(p[1] - center[1])])

def sdf_gradient_line(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-8), 0.0, 1.0)
    diff = p - (a + t * ab)
    return diff / max(np.linalg.norm(diff), 1e-6)


class SafetyMixedShapesEnv(gym.Env):
    def __init__(self, nav_model_path="./models_dynamic/nav_shapes_model",
                 systematic_bias_range=(-0.6, 0.4), jitter_range=(-0.2, 0.2),
                 bias_magnitude_range=(0.3, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.systematic_bias_range = systematic_bias_range
        self.jitter_range = jitter_range
        self.bias_magnitude_range = bias_magnitude_range
        self.bias = np.zeros(2)
        self.sensor_bias = 0.0
        self.prev_e_i = np.array([1.0, 0.0])
        self.n_obs = 4
        self.obstacles = []
        self.sensor_errors = []
        self.est_effective_radius = []

        self.nav_model = PPO.load(nav_model_path)

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.01], dtype=np.float32),
            high=np.array([5.0, 10.0], dtype=np.float32),
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

        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = cp.Parameter(2)
        self._rhs = cp.Parameter()
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h @ self._u >= self._rhs]
        self._prob = cp.Problem(cost, constraints)

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

    def _obs_sdf_est(self, p, obs, error):
        if obs["type"] == CIRCLE:
            return sdf_circle(p, obs["center"], obs["radius"] + error)
        elif obs["type"] == RECTANGLE:
            return sdf_rectangle(p, obs["center"], obs["half_w"] + error, obs["half_h"] + error)
        else:
            return sdf_line(p, obs["a"], obs["b"], obs["thickness"] + error)

    def _obs_gradient(self, p, obs):
        if obs["type"] == CIRCLE:
            return sdf_gradient_circle(p, obs["center"])
        elif obs["type"] == RECTANGLE:
            return sdf_gradient_rectangle(p, obs["center"], obs["half_w"], obs["half_h"])
        else:
            return sdf_gradient_line(p, obs["a"], obs["b"])

    def _compute_shield_h(self, robot_pos):
        sdf_vals = [self._obs_sdf_est(robot_pos, obs, self.sensor_errors[i])
                    for i, obs in enumerate(self.obstacles)]
        closest_idx = int(np.argmin(sdf_vals))
        obs = self.obstacles[closest_idx]

        e_i_current = self._obs_gradient(robot_pos, obs)
        e_i = self.prev_e_i

        R_eff = obs["effective_radius"] + self.sensor_errors[closest_idx]
        projected_dist = np.dot(robot_pos - obs["center"], e_i) - R_eff

        if np.dot(robot_pos - obs["center"], e_i) >= 0:
            exp_term = np.exp(-GAMMA_H * projected_dist)
            h_val = LAMBDA_H * (1.0 - exp_term)
            Lgh = LAMBDA_H * GAMMA_H * exp_term * e_i
        else:
            exp_gR = np.exp(GAMMA_H * R_eff)
            grad_h = LAMBDA_H * GAMMA_H * exp_gR * e_i
            h_at_boundary = LAMBDA_H * (1.0 - exp_gR)
            h_val = np.dot(grad_h, robot_pos - obs["center"]) + h_at_boundary
            Lgh = grad_h

        return h_val, Lgh, closest_idx, e_i_current

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0
        self.prev_e_i = np.array([1.0, 0.0])

        self.target_pos = np.array([150.0, self.np_random.uniform(-5.0, 5.0)])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        self.obstacles = self._generate_obstacles()

        self.sensor_bias = self.np_random.uniform(*self.systematic_bias_range)
        self.sensor_errors = []
        self.est_effective_radius = []
        for obs in self.obstacles:
            jitter = self.np_random.uniform(*self.jitter_range)
            error = self.sensor_bias + jitter
            self.sensor_errors.append(error)
            self.est_effective_radius.append(max(obs["effective_radius"] + error, 1.0))

        bias_mag = self.np_random.uniform(*self.bias_magnitude_range)
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_mag * np.array([np.cos(angle), np.sin(angle)])

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        sdf_vals = [self._obs_sdf_est(self.robot_pos, obs, self.sensor_errors[i])
                    for i, obs in enumerate(self.obstacles)]
        ci = int(np.argmin(sdf_vals))
        self.prev_e_i = self._obs_gradient(self.robot_pos, self.obstacles[ci])

        return self._get_obs(), {}

    def step(self, action):
        alpha, phi = float(action[0]), float(action[1])
        self.current_step += 1

        obs_vec = self._get_obs()
        nav_action, _ = self.nav_model.predict(obs_vec, deterministic=True)
        k_nom = np.clip(np.array([float(nav_action[0]), float(nav_action[1])]), -6.0, 6.0)

        h_val, Lgh, closest_idx, e_i_current = self._compute_shield_h(self.robot_pos)

        self._k_nom_param.value = k_nom
        self._L_g_h.value = Lgh
        Lgh_norm_sq = np.sum(Lgh**2)
        h_safe = max(h_val, 1e-4)
        self._rhs.value = -alpha * h_val + (Lgh_norm_sq * phi) / h_safe

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value if self._u.value is not None else np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])
        safe_u = np.clip(safe_u, -6.0, 6.0)

        self.prev_e_i = e_i_current
        self.robot_pos += safe_u * self.dt + self.bias * self.dt
        self.velocity = safe_u.copy()

        true_sdfs = [self._obs_sdf_true(self.robot_pos, obs) for obs in self.obstacles]
        min_true_sdf = min(true_sdfs)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        truncated = False
        reward = 0.0
        if min_true_sdf < 0:
            reward = -250.0
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

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {
            "safe_u": safe_u, "k_nom": k_nom,
            "min_true_sdf": float(min_true_sdf),
            "alpha": float(alpha), "phi": float(phi),
            "h_val": float(h_val),
            "sensor_bias": float(self.sensor_bias),
            "bias": self.bias.copy(),
        }

    def _get_obs(self):
        parts = []
        for i, obs in enumerate(self.obstacles):
            rel = obs["center"] - self.robot_pos
            parts.extend([rel[0], rel[1], self.est_effective_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
