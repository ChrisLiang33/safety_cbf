"""
DENSE obstacle navigation (6 obstacles) with learnable phi (ISSf-CBF).
Agent controls [kx, ky, alpha, phi].

ISSf-CBF constraint: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)

Key idea: With 6 obstacles instead of 3, the robot must navigate through tighter
gaps where the margin from phi becomes critical. More obstacles = more CBF constraints
= more situations where phi's robustness margin matters.

Includes radius estimation noise + constant bias + randomized obstacles.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp

N_OBSTACLES = 6


class PhiCBFDenseObstaclesEnv(gym.Env):
    def __init__(self, radius_error_range=(-1.0, 1.0), bias_magnitude_range=(0.0, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 3.0
        self.n_obs = N_OBSTACLES
        self.radius_error_range = radius_error_range
        self.bias_magnitude_range = bias_magnitude_range
        self.true_radius = [5.0] * N_OBSTACLES
        self.estimated_radius = [5.0] * N_OBSTACLES
        self.bias = np.zeros(2)

        # ACTION: [kx, ky, alpha, phi]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.1, 0.01], dtype=np.float32),
            high=np.array([3.0, 3.0, 5.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: 3 per obstacle (rel_x, rel_y, est_radius) + 3 target + 2 velocity
        # = 6*3 + 3 + 2 = 23 dims
        obs_low = []
        obs_high = []
        for _ in range(N_OBSTACLES):
            obs_low.extend([-120, -25, 0])
            obs_high.extend([120, 25, 10])
        obs_low.extend([-120, -25, 0.1, -4, -4])
        obs_high.extend([120, 25, 5.0, 4, 4])

        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obs_pos = [np.zeros(2) for _ in range(N_OBSTACLES)]
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0

        # Parametric QP with N_OBSTACLES CBF constraints
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = [cp.Parameter(2) for _ in range(N_OBSTACLES)]
        self._rhs = [cp.Parameter() for _ in range(N_OBSTACLES)]
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h[i] @ self._u >= self._rhs[i] for i in range(N_OBSTACLES)]
        self._prob = cp.Problem(cost, constraints)

    def _place_obstacles(self):
        min_spacing = 11.0  # slightly tighter than 12 to allow 6 to fit
        placed = []
        for _ in range(self.n_obs):
            for _attempt in range(200):
                x = self.np_random.uniform(12.0, 88.0)
                y = self.np_random.uniform(-10.0, 10.0)
                candidate = np.array([x, y])
                too_close = any(np.linalg.norm(candidate - p) < min_spacing for p in placed)
                if not too_close:
                    placed.append(candidate)
                    break
            else:
                placed.append(np.array([self.np_random.uniform(12.0, 88.0),
                                        self.np_random.uniform(-10.0, 10.0)]))
        return placed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        placed = self._place_obstacles()
        for i in range(self.n_obs):
            self.obs_pos[i] = placed[i]

        self.true_radius = [5.0] * self.n_obs
        for i in range(self.n_obs):
            error = self.np_random.uniform(
                self.radius_error_range[0], self.radius_error_range[1]
            )
            self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

        bias_magnitude = self.np_random.uniform(
            self.bias_magnitude_range[0], self.bias_magnitude_range[1]
        )
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_magnitude * np.array([np.cos(angle), np.sin(angle)])

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx = float(action[0])
        ky = float(action[1])
        alpha = float(action[2])
        phi = float(action[3])
        k_nom = np.array([kx, ky])

        h_vals = []
        L_g_h_vals = []
        for i in range(self.n_obs):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.estimated_radius[i]**2
            L_g_h = 2 * x_diff
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

        self._k_nom_param.value = k_nom
        for i in range(self.n_obs):
            self._L_g_h[i].value = L_g_h_vals[i]
            Lgh_norm_sq = np.sum(L_g_h_vals[i]**2)
            h_safe = max(h_vals[i], 1e-4)
            self._rhs[i].value = -alpha * h_vals[i] + (Lgh_norm_sq * phi) / h_safe

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])

        safe_u = np.clip(safe_u, -3.0, 3.0)

        self.robot_pos += safe_u * self.dt + self.bias * self.dt
        self.velocity = safe_u.copy()

        dist2obs_true = []
        for i in range(self.n_obs):
            dist2obs_true.append(
                np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.true_radius[i]
            )
        min_obs_dist_true = min(dist2obs_true)

        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        reward = 0.0

        if min_obs_dist_true < 0:
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
            "safe_u": safe_u, "k_nom": k_nom,
            "dist2obs_true": dist2obs_true,
            "min_obs_dist_true": float(min_obs_dist_true),
            "alpha": float(alpha), "phi": float(phi),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
            "bias": self.bias.copy(),
        }

    def _get_obs(self):
        parts = []
        for i in range(self.n_obs):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
