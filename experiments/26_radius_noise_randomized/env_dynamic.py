"""
3-obstacle navigation with learnable phi (ISSf-CBF) + OBSTACLE RADIUS ESTIMATION NOISE.
Agent controls [kx, ky, alpha, phi].

ISSf-CBF constraint: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)

Key differences from exp 25:
  - Fully randomized obstacle placement (not forced slalom pattern)
  - Obstacles can appear anywhere in the corridor with minimum spacing
  - No alpha-only baseline — we only train the ISSf-CBF agent
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class PhiCBFRadiusNoiseRandomizedEnv(gym.Env):
    def __init__(self, radius_error_range=(-1.0, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 3.0
        self.radius_error_range = radius_error_range
        self.true_radius = [5.0, 5.0, 5.0]
        self.estimated_radius = [5.0, 5.0, 5.0]

        # ACTION: [kx, ky, alpha, phi]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.1, 0.01], dtype=np.float32),
            high=np.array([3.0, 3.0, 5.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: 14 dims (uses estimated radius in observation)
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
        """Place 3 obstacles randomly in the corridor with minimum spacing."""
        min_spacing = 12.0  # minimum distance between obstacle centers
        placed = []
        for _ in range(3):
            for _attempt in range(100):
                x = self.np_random.uniform(15.0, 85.0)
                y = self.np_random.uniform(-10.0, 10.0)
                candidate = np.array([x, y])
                # Check spacing from other obstacles
                too_close = False
                for p in placed:
                    if np.linalg.norm(candidate - p) < min_spacing:
                        too_close = True
                        break
                # Check not blocking start or target
                if x < 10.0 or x > 90.0:
                    too_close = True
                if not too_close:
                    placed.append(candidate)
                    break
            else:
                # Fallback: place with less constraint
                x = self.np_random.uniform(15.0, 85.0)
                y = self.np_random.uniform(-10.0, 10.0)
                placed.append(np.array([x, y]))
        return placed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        # Fully randomized obstacle placement
        placed = self._place_obstacles()
        for i in range(3):
            self.obs_pos[i] = placed[i]

        # TRUE radius is always 5.0 (for collision detection)
        self.true_radius = [5.0, 5.0, 5.0]

        # ESTIMATED radius = true + error (sampled ONCE per episode per obstacle)
        for i in range(3):
            error = self.np_random.uniform(
                self.radius_error_range[0], self.radius_error_range[1]
            )
            self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx = float(action[0])
        ky = float(action[1])
        alpha = float(action[2])
        phi = float(action[3])
        k_nom = np.array([kx, ky])

        # CBF uses ESTIMATED radius
        h_vals = []
        L_g_h_vals = []
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.estimated_radius[i]**2
            L_g_h = 2 * x_diff
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

        # ISSf-CBF: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)
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

        # No dynamics disturbance
        self.robot_pos += safe_u * self.dt
        self.velocity = safe_u.copy()

        # Collision detection uses TRUE radius
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
            "min_obs_dist_est": float(min(dist2obs_est)),
            "alpha": float(alpha), "phi": float(phi),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
            "radius_errors": [self.estimated_radius[i] - self.true_radius[i] for i in range(3)],
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
