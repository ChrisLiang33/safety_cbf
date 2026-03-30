"""
Exp 51: ISSf-CBF using signed distance function (SDF) instead of squared distance.
Agent controls [kx, ky, alpha, phi].

Key change from exp 45: CBF uses SDF formulation.
  Previous: h(x) = ||x - obs||^2 - r^2,  Lgh = 2(x - obs)
  Now:      h(x) = ||x - obs|| - r,       Lgh = (x - obs) / ||x - obs||

With SDF, Lgh is a UNIT VECTOR — it doesn't scale with distance.
This makes phi's effect more uniform and interpretable:
  - In the squared version, ||Lgh||^2 grows with distance, making the
    phi term's effect inconsistent at different distances
  - In the SDF version, ||Lgh||^2 = 1 always, so the phi term
    simplifies to phi / max(h, eps) — purely a function of proximity

Based on exp 45 setup:
  Sensor error = systematic_bias + per-measurement jitter.
  Max speed 6 m/s, target at x=150, 800 max steps.
  Dynamics bias U(0.3, 1.0). Random obstacle radius U(3.0, 7.0).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class SDFCBFEnv(gym.Env):
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

        self.action_space = spaces.Box(
            low=np.array([-6.0, -6.0, 0.1, 0.01], dtype=np.float32),
            high=np.array([6.0, 6.0, 5.0, 10.0], dtype=np.float32),
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

        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = [cp.Parameter(2) for _ in range(3)]
        self._rhs = [cp.Parameter() for _ in range(3)]
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h[i] @ self._u >= self._rhs[i] for i in range(3)]
        self._prob = cp.Problem(cost, constraints)

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
            jitter = self.np_random.uniform(
                self.jitter_range[0], self.jitter_range[1])
            error = self.sensor_bias + jitter
            self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

        bias_mag = self.np_random.uniform(
            self.bias_magnitude_range[0], self.bias_magnitude_range[1])
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_mag * np.array([np.cos(angle), np.sin(angle)])

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx, ky, alpha, phi = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        k_nom = np.array([kx, ky])
        self.current_step += 1

        h_vals, L_g_h_vals = [], []
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            dist = np.linalg.norm(x_diff)
            dist_safe = max(dist, 1e-6)  # avoid division by zero

            # SDF: h = ||x - obs|| - r_est
            h_vals.append(dist - self.estimated_radius[i])
            # Lgh = (x - obs) / ||x - obs||  (unit vector)
            L_g_h_vals.append(x_diff / dist_safe)

        self._k_nom_param.value = k_nom
        for i in range(3):
            self._L_g_h[i].value = L_g_h_vals[i]
            Lgh_norm_sq = np.sum(L_g_h_vals[i]**2)  # ~1.0 (unit vector)
            h_safe = max(h_vals[i], 1e-4)
            self._rhs[i].value = -alpha * h_vals[i] + (Lgh_norm_sq * phi) / h_safe

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value if self._u.value is not None else np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])
        safe_u = np.clip(safe_u, -6.0, 6.0)

        self.robot_pos += safe_u * self.dt + self.bias * self.dt
        self.velocity = safe_u.copy()

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

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {
            "safe_u": safe_u, "k_nom": k_nom, "dist2obs_true": dist2obs_true,
            "min_obs_dist_true": float(min_obs_dist_true),
            "alpha": float(alpha), "phi": float(phi),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
            "sensor_bias": float(self.sensor_bias),
            "bias": self.bias.copy(),
        }

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
