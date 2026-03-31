"""
Exp 53: ISSf-CBF using SHIELD SDF pipeline (Yang et al., IROS 2025).
Agent controls [kx, ky, alpha, phi].

Implements equations 19-22 from the SHIELD paper:
  1. Multi-obstacle SDF (Eq 19): sdf(x) = min_i ||p - rho_i|| - R_i
  2. Smooth SDF (Eq 20): h_smooth = lambda * (1 - exp(-gamma * sdf))
  3. Concave approximation (Eq 21): projected distance along e_i from PREVIOUS timestep
  4. Piecewise extension (Eq 22): linear approximation when behind obstacle

Connection to Theorem 2 (Eq 23) and full constraint (Eq 24):
  Paper's constraint: h_tilde(F + Gu + E[d]) - (lambda_max/2) e^T cov(d) e >= alpha * h_k
  Our ISSf-CBF:       Lgh @ u >= -alpha * h + (||Lgh||^2 * phi) / h

  The paper's uncertainty penalty (lambda_max/2) e^T cov(d) e uses the PROJECTED
  covariance along e_i only (Theorem 2), not total variance (Prop 1). This is
  less conservative — only penalizes uncertainty in the obstacle direction.

  Our phi term serves the same role: since Lgh = lambda*gamma*exp(-gamma*proj_dist)*e_i,
  ||Lgh||^2 already captures only the e_i direction. So (||Lgh||^2 * phi) / h
  is conceptually the learned version of the projected covariance penalty.
  phi learns to approximate the uncertainty magnitude that the paper computes
  from the CVAE's covariance.

Key differences from exp 51/52:
  - Single constraint (closest obstacle) instead of 3 separate constraints
  - Direction vector e_i from previous timestep makes h concave in x
  - Piecewise extension handles edge cases
  - Smoothing handles chattering when closest obstacle switches

Based on exp 45 setup:
  Sensor error = systematic_bias + per-measurement jitter.
  Max speed 6 m/s, target at x=150, 800 max steps.
  Dynamics bias U(0.3, 1.0). Random obstacle radius U(3.0, 7.0).

Parameters: lambda=1.0, gamma=0.5
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp

# SHIELD smoothing parameters
LAMBDA_H = 1.0   # max magnitude of safety value
GAMMA_H = 0.5    # smoothness / sensitivity


class ShieldSDFEnv(gym.Env):
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

        # Previous timestep direction vector (for concave approximation)
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0

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

        # QP with single constraint (closest obstacle)
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h = cp.Parameter(2)
        self._rhs = cp.Parameter()
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h @ self._u >= self._rhs]
        self._prob = cp.Problem(cost, constraints)

    def _place_obstacles(self):
        placed = []
        x_bands = [(20.0, 50.0), (60.0, 100.0), (110.0, 140.0)]
        for x_low, x_high in x_bands:
            x = self.np_random.uniform(x_low, x_high)
            y = self.np_random.uniform(-10.0, 10.0)
            placed.append(np.array([x, y]))
        return placed

    def _compute_shield_h(self, robot_pos):
        """
        Compute h and Lgh using the SHIELD SDF pipeline (Eq 19-22).
        Returns h_val, Lgh, closest_idx, e_i (for storing as prev)
        """
        # Step 1: Find closest obstacle (Eq 19)
        sdf_vals = []
        for i in range(3):
            dist = np.linalg.norm(robot_pos - self.obs_pos[i])
            sdf_vals.append(dist - self.estimated_radius[i])
        closest_idx = int(np.argmin(sdf_vals))
        rho_i = self.obs_pos[closest_idx]
        R_i = self.estimated_radius[closest_idx]

        # Compute current direction vector (for storing as prev_e_i)
        diff = robot_pos - rho_i
        dist = np.linalg.norm(diff)
        dist_safe = max(dist, 1e-6)
        e_i_current = diff / dist_safe

        # Use direction from PREVIOUS timestep for concave approximation
        e_i = self.prev_e_i

        # Step 3: Concave approximation (Eq 21)
        # Projected distance along previous direction
        projected_dist = np.dot(diff, e_i) - R_i

        # Step 4: Piecewise extension (Eq 22)
        if np.dot(diff, e_i) >= 0:
            # Robot is "in front of" obstacle — use concave approx
            # h_hat = lambda * (1 - exp(-gamma * projected_dist))
            exp_term = np.exp(-GAMMA_H * projected_dist)
            h_val = LAMBDA_H * (1.0 - exp_term)

            # Gradient: dh/dx = lambda * gamma * exp(-gamma * proj_dist) * e_i
            Lgh = LAMBDA_H * GAMMA_H * exp_term * e_i
        else:
            # Robot is "behind" obstacle — linear approximation (Eq 22)
            # At the boundary where (p-rho)^T e_i = 0, projected_dist = -R_i
            # h_hat at boundary: lambda * (1 - exp(-gamma * (-R_i))) = lambda * (1 - exp(gamma*R_i))
            # gradient at boundary: lambda * gamma * exp(gamma*R_i) * e_i
            # Linear extension: h = gradient^T * (p - rho) + h_at_boundary
            #                     = gradient^T * diff + lambda*(1 - exp(gamma*R_i))
            exp_gR = np.exp(GAMMA_H * R_i)
            grad_h = LAMBDA_H * GAMMA_H * exp_gR * e_i
            h_at_boundary = LAMBDA_H * (1.0 - exp_gR)
            h_val = np.dot(grad_h, diff) + h_at_boundary
            Lgh = grad_h

        return h_val, Lgh, closest_idx, e_i_current

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0

        # Reset previous direction
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0

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

        # Initialize prev_e_i from first closest obstacle
        sdf_vals = [np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.estimated_radius[i]
                    for i in range(3)]
        ci = int(np.argmin(sdf_vals))
        diff = self.robot_pos - self.obs_pos[ci]
        d = max(np.linalg.norm(diff), 1e-6)
        self.prev_e_i = diff / d
        self.prev_closest_idx = ci

        return self._get_obs(), {}

    def step(self, action):
        kx, ky, alpha, phi = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        k_nom = np.array([kx, ky])
        self.current_step += 1

        # SHIELD SDF pipeline
        h_val, Lgh, closest_idx, e_i_current = self._compute_shield_h(self.robot_pos)

        # ISSf-CBF constraint: Lgh @ u >= -alpha * h + (||Lgh||^2 * phi) / max(h, eps)
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

        # Update previous direction for next step's concave approximation
        self.prev_e_i = e_i_current
        self.prev_closest_idx = closest_idx

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
            "h_val": float(h_val), "closest_obs": closest_idx,
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
