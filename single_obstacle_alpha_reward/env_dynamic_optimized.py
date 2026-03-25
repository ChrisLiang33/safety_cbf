"""
Same as fixedVSdynamic_ablation/env_dynamic_optimized.py but with alpha reward bonus.
Only change: reward += 0.3 * alpha
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp

class AdaptiveCBFEnvOptimized(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1

        # ACTION SPACE: [alpha, k_x, k_y]
        self.action_space = spaces.Box(
            low=np.array([0.1, -2.0, -2.0], dtype=np.float32),
            high=np.array([5.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        # OBSERVATION: [rel_obs_x, rel_obs_y, obs_radius, rel_target_x, rel_target_y, target_radius, vel_x, vel_y]
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -20.0, 0.0, -20.0, -20.0, 0.1, -2.0, -2.0], dtype=np.float32),
            high=np.array([20.0, 20.0, 5.0, 20.0, 20.0, 3.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        self.target_pos = np.zeros(2)
        self.target_radius = 1.0
        self.prev_dist2target = 0.0
        self.robot_pos = np.zeros(2)
        self.obstacle_pos = np.zeros(2)
        self.obstacle_radius = 1.0
        self.velocity = np.zeros(2)

        # Pre-build parametric QP
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h_param = cp.Parameter(2)
        self._rhs_param = cp.Parameter()
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h_param @ self._u >= self._rhs_param]
        self._prob = cp.Problem(cost, constraints)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        self.obstacle_pos = np.array([
            self.np_random.uniform(2.0, 8.0),
            self.np_random.uniform(-2.0, 2.0)
        ])

        target_y = self.np_random.uniform(-4.0, 4.0)
        self.target_pos = np.array([9.0, target_y])
        self.target_radius = self.np_random.uniform(0.5, 2.0)

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        alpha, k_x, k_y = action
        k_nom = np.array([k_x, k_y])

        x_diff = self.robot_pos - self.obstacle_pos
        h_x = np.sum(x_diff**2) - (self.obstacle_radius**2)
        L_g_h = 2 * x_diff

        self._k_nom_param.value = k_nom
        self._L_g_h_param.value = L_g_h
        self._rhs_param.value = -alpha * h_x

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])

        safe_u = np.clip(safe_u, -2.0, 2.0)
        self.robot_pos += safe_u * self.dt
        self.velocity = safe_u.copy()

        dist2obstacle = np.linalg.norm(self.robot_pos - self.obstacle_pos) - self.obstacle_radius
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        reward = 0.0

        if dist2obstacle < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0
            reward -= (abs(k_nom[1]) * 0.1) * self.dt
            reward -= 0.5
            reward += 0.3 * alpha  # incentivize keeping alpha high when safe
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        if self.robot_pos[0] < -2.0 or self.robot_pos[0] > 11.0 or abs(self.robot_pos[1]) > 5.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {"safe_u": safe_u, "h_x": float(dist2obstacle), "alpha": float(alpha)}

    def _get_obs(self):
        rel_obs_x = self.obstacle_pos[0] - self.robot_pos[0]
        rel_obs_y = self.obstacle_pos[1] - self.robot_pos[1]
        rel_target_x = self.target_pos[0] - self.robot_pos[0]
        rel_target_y = self.target_pos[1] - self.robot_pos[1]

        return np.array([
            float(rel_obs_x), float(rel_obs_y), float(self.obstacle_radius),
            float(rel_target_x), float(rel_target_y), float(self.target_radius),
            float(self.velocity[0]), float(self.velocity[1]),
        ], dtype=np.float32)
