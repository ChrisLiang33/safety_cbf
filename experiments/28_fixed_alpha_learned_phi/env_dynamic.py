"""
3-obstacle navigation with FIXED alpha + learnable phi (ISSf-CBF).
Agent controls [kx, ky, phi] only — alpha is fixed at 1.0.

ISSf-CBF constraint: Lgh @ u >= -FIXED_ALPHA * h(x) + (||Lgh||^2 * phi) / h(x)

Key idea: In previous experiments, the policy could adjust alpha to handle obstacles
instead of using phi. By fixing alpha, phi is the ONLY safety tuning knob the policy
has. If it can't learn adaptive phi here, the learning signal itself may be the problem.

Includes radius estimation noise + constant bias + randomized obstacles.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp

FIXED_ALPHA = 1.0


class FixedAlphaLearnedPhiEnv(gym.Env):
    def __init__(self, radius_error_range=(-1.0, 1.0), bias_magnitude_range=(0.0, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 3.0
        self.radius_error_range = radius_error_range
        self.bias_magnitude_range = bias_magnitude_range
        self.true_radius = [5.0, 5.0, 5.0]
        self.estimated_radius = [5.0, 5.0, 5.0]
        self.bias = np.zeros(2)

        # ACTION: [kx, ky, phi] — alpha is FIXED, not learned
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.01], dtype=np.float32),
            high=np.array([3.0, 3.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: 14 dims
        self.observation_space = spaces.Box(
            low=np.array([-120, -25, 0, -120, -25, 0, -120, -25, 0,
                          -120, -25, 0.1, -4, -4], dtype=np.float32),
            high=np.array([120, 25, 10, 120, 25, 10, 120, 25, 10,
                           120, 25, 5.0, 4, 4], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obs_pos = [np.zeros(2) for _ in range(3)]
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0

        # Parametric QP with 3 CBF constraints
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = [cp.Parameter(2) for _ in range(3)]
        self._rhs = [cp.Parameter() for _ in range(3)]
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h[i] @ self._u >= self._rhs[i] for i in range(3)]
        self._prob = cp.Problem(cost, constraints)

    def _place_obstacles(self):
        min_spacing = 12.0
        placed = []
        for _ in range(3):
            for _attempt in range(100):
                x = self.np_random.uniform(15.0, 85.0)
                y = self.np_random.uniform(-10.0, 10.0)
                candidate = np.array([x, y])
                too_close = any(np.linalg.norm(candidate - p) < min_spacing for p in placed)
                if not too_close:
                    placed.append(candidate)
                    break
            else:
                placed.append(np.array([self.np_random.uniform(15.0, 85.0),
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
        for i in range(3):
            self.obs_pos[i] = placed[i]

        self.true_radius = [5.0, 5.0, 5.0]
        for i in range(3):
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
        phi = float(action[2])
        alpha = FIXED_ALPHA  # NOT learned
        k_nom = np.array([kx, ky])

        h_vals = []
        L_g_h_vals = []
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.estimated_radius[i]**2
            L_g_h = 2 * x_diff
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

        # ISSf-CBF with FIXED alpha
        self._k_nom_param.value = k_nom
        for i in range(3):
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
        for i in range(3):
            dist2obs_true.append(
                np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.true_radius[i]
            )
        min_obs_dist_true = min(dist2obs_true)

        dist2obs_est = []
        for i in range(3):
            dist2obs_est.append(
                np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.estimated_radius[i]
            )

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
            "dist2obs_true": dist2obs_true, "dist2obs_est": dist2obs_est,
            "min_obs_dist_true": float(min_obs_dist_true),
            "alpha": float(alpha), "phi": float(phi),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
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
