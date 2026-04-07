"""
Exp 72 Phase 2: Discrete safety filter with moving obstacles (step=0.5).
Frozen navigation policy provides kx, ky.
Agent controls discrete alpha/phi adjustments: Discrete(9).
Persistent alpha/phi state updated by step increments.

SHIELD SDF pipeline (Eq 19-22) with single constraint.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from stable_baselines3 import PPO

LAMBDA_H = 1.0
GAMMA_H = 0.5
OBS_SPEED_RANGE = (0.3, 1.0)

ALPHA_STEP = 0.5
PHI_STEP = 0.5
ALPHA_MIN, ALPHA_MAX = 0.1, 5.0
PHI_MIN, PHI_MAX = 0.01, 10.0

# Discrete(9) action mapping:
# 0: alpha down, phi down    1: alpha down, phi stay    2: alpha down, phi up
# 3: alpha stay, phi down    4: alpha stay, phi stay    5: alpha stay, phi up
# 6: alpha up,   phi down    7: alpha up,   phi stay    8: alpha up,   phi up
ALPHA_DELTAS = [-ALPHA_STEP, -ALPHA_STEP, -ALPHA_STEP,
                 0.0,          0.0,          0.0,
                 ALPHA_STEP,   ALPHA_STEP,   ALPHA_STEP]
PHI_DELTAS   = [-PHI_STEP,    0.0,          PHI_STEP,
                -PHI_STEP,    0.0,          PHI_STEP,
                -PHI_STEP,    0.0,          PHI_STEP]


class SafetyDiscreteMovingObsEnv05(gym.Env):
    def __init__(self, nav_model_path="./models_dynamic/nav_moving_discrete05_model_v2",
                 systematic_bias_range=(-0.6, 0.4), jitter_range=(-0.2, 0.2),
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
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0
        self.obs_vel = [np.zeros(2) for _ in range(3)]

        # Persistent alpha/phi state
        self.alpha = 2.5
        self.phi = 1.0

        self.nav_model = PPO.load(nav_model_path)

        # Discrete(9): 3 alpha choices x 3 phi choices
        self.action_space = spaces.Discrete(9)

        # Observation: same 14-dim nav obs + alpha + phi = 16
        self.observation_space = spaces.Box(
            low=np.array([-180, -35, 0, -180, -35, 0, -180, -35, 0,
                          -180, -35, 0.1, -7, -7, ALPHA_MIN, PHI_MIN],
                         dtype=np.float32),
            high=np.array([180, 35, 10, 180, 35, 10, 180, 35, 10,
                           180, 35, 5.0, 7, 7, ALPHA_MAX, PHI_MAX],
                          dtype=np.float32),
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

    def _move_obstacles(self):
        for i in range(3):
            self.obs_pos[i] += self.obs_vel[i] * self.dt
            if self.obs_pos[i][1] > 18.0:
                self.obs_pos[i][1] = 18.0
                self.obs_vel[i][1] *= -1
            elif self.obs_pos[i][1] < -18.0:
                self.obs_pos[i][1] = -18.0
                self.obs_vel[i][1] *= -1
            self.obs_pos[i][0] = np.clip(self.obs_pos[i][0], 5.0, 155.0)

    def _compute_shield_h(self, robot_pos):
        sdf_vals = []
        for i in range(3):
            dist = np.linalg.norm(robot_pos - self.obs_pos[i])
            sdf_vals.append(dist - self.estimated_radius[i])
        closest_idx = int(np.argmin(sdf_vals))
        rho_i = self.obs_pos[closest_idx]
        R_i = self.estimated_radius[closest_idx]

        diff = robot_pos - rho_i
        dist = np.linalg.norm(diff)
        dist_safe = max(dist, 1e-6)
        e_i_current = diff / dist_safe

        e_i = self.prev_e_i
        projected_dist = np.dot(diff, e_i) - R_i

        if np.dot(diff, e_i) >= 0:
            exp_term = np.exp(-GAMMA_H * projected_dist)
            h_val = LAMBDA_H * (1.0 - exp_term)
            Lgh = LAMBDA_H * GAMMA_H * exp_term * e_i
        else:
            exp_gR = np.exp(GAMMA_H * R_i)
            grad_h = LAMBDA_H * GAMMA_H * exp_gR * e_i
            h_at_boundary = LAMBDA_H * (1.0 - exp_gR)
            h_val = np.dot(grad_h, diff) + h_at_boundary
            Lgh = grad_h

        return h_val, Lgh, closest_idx, e_i_current

    def _get_nav_obs(self):
        """14-dim observation for the frozen nav policy."""
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0

        # Reset persistent alpha/phi to midpoints
        self.alpha = 2.5
        self.phi = 1.0

        target_y = self.np_random.uniform(-5.0, 5.0)
        self.target_pos = np.array([150.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        placed = self._place_obstacles()
        for i in range(3):
            self.obs_pos[i] = placed[i]
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

        self.obs_vel = []
        for _ in range(3):
            speed = self.np_random.uniform(*OBS_SPEED_RANGE)
            ang = self.np_random.uniform(0, 2 * np.pi)
            self.obs_vel.append(speed * np.array([np.cos(ang), np.sin(ang)]))

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        sdf_vals = [np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.estimated_radius[i]
                    for i in range(3)]
        ci = int(np.argmin(sdf_vals))
        diff = self.robot_pos - self.obs_pos[ci]
        d = max(np.linalg.norm(diff), 1e-6)
        self.prev_e_i = diff / d
        self.prev_closest_idx = ci

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        # Update persistent alpha/phi
        self.alpha = np.clip(self.alpha + ALPHA_DELTAS[action], ALPHA_MIN, ALPHA_MAX)
        self.phi = np.clip(self.phi + PHI_DELTAS[action], PHI_MIN, PHI_MAX)
        alpha = self.alpha
        phi = self.phi

        self.current_step += 1
        self._move_obstacles()

        nav_obs = self._get_nav_obs()
        nav_action, _ = self.nav_model.predict(nav_obs, deterministic=True)
        kx, ky = float(nav_action[0]), float(nav_action[1])
        k_nom = np.clip(np.array([kx, ky]), -6.0, 6.0)

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
            "safe_u": safe_u, "k_nom": k_nom, "dist2obs_true": dist2obs_true,
            "min_obs_dist_true": float(min_obs_dist_true),
            "alpha": float(alpha), "phi": float(phi),
            "h_val": float(h_val),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
            "sensor_bias": float(self.sensor_bias),
            "bias": self.bias.copy(),
            "discrete_action": action,
        }

    def _get_obs(self):
        """16-dim: 14-dim nav obs + alpha + phi."""
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        parts.extend([self.alpha, self.phi])
        return np.array(parts, dtype=np.float32)
