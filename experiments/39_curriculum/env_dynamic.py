"""
Exp 39: ISSf-CBF with CURRICULUM LEARNING.
Agent controls [kx, ky, alpha, phi].

Key idea: gradually increase difficulty over training.
  Phase 1 (0-1M steps): clean — no bias, no radius noise
  Phase 2 (1M-3M steps): moderate — bias U(0, 0.5), radius error U(-0.5, 0.5)
  Phase 3 (3M-5M steps): hard — bias U(0, 1.0), radius error U(-1.0, 1.0)

The policy first learns basic navigation + alpha/phi usage, then faces
increasing uncertainty. Random obstacle radius throughout.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class CurriculumEnv(gym.Env):
    def __init__(self, obs_radius_range=(3.0, 7.0)):
        super().__init__()
        self.dt = 0.1
        self.obs_radius_range = obs_radius_range
        self.true_radius = [5.0, 5.0, 5.0]
        self.estimated_radius = [5.0, 5.0, 5.0]
        self.bias = np.zeros(2)

        # Curriculum phase — controlled externally via set_phase()
        self.phase = 1  # 1=clean, 2=moderate, 3=hard

        # ACTION: [kx, ky, alpha, phi]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.1, 0.01], dtype=np.float32),
            high=np.array([3.0, 3.0, 5.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

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

        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = [cp.Parameter(2) for _ in range(3)]
        self._rhs = [cp.Parameter() for _ in range(3)]
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h[i] @ self._u >= self._rhs[i] for i in range(3)]
        self._prob = cp.Problem(cost, constraints)

    def set_phase(self, phase):
        """Set curriculum phase: 1=clean, 2=moderate, 3=hard."""
        self.phase = phase

    def _get_noise_params(self):
        """Return (bias_range, radius_error_range) based on phase."""
        if self.phase == 1:
            return (0.0, 0.0), (0.0, 0.0)  # clean
        elif self.phase == 2:
            return (0.0, 0.5), (-0.5, 0.5)  # moderate
        else:
            return (0.0, 1.0), (-1.0, 1.0)  # hard

    def _place_obstacles(self):
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

        # RANDOM obstacle radius
        for i in range(3):
            self.true_radius[i] = self.np_random.uniform(
                self.obs_radius_range[0], self.obs_radius_range[1]
            )

        bias_range, error_range = self._get_noise_params()

        # ESTIMATED radius
        for i in range(3):
            if error_range[0] == 0.0 and error_range[1] == 0.0:
                self.estimated_radius[i] = self.true_radius[i]
            else:
                error = self.np_random.uniform(error_range[0], error_range[1])
                self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

        # CONSTANT BIAS
        if bias_range[1] > 0:
            bias_magnitude = self.np_random.uniform(bias_range[0], bias_range[1])
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.bias = bias_magnitude * np.array([np.cos(angle), np.sin(angle)])
        else:
            self.bias = np.zeros(2)

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
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.estimated_radius[i]**2
            L_g_h = 2 * x_diff
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

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
            "phase": self.phase,
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
