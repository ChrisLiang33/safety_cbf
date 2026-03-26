"""
Single-obstacle alpha-only CBF environment.
Agent outputs [alpha] only. k_nom is a fixed proportional controller toward the target.
This isolates alpha as the ONLY lever — the agent cannot optimize the path.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class AlphaOnlyDynamicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 2.0  # fixed nominal speed toward target

        # ACTION: [alpha] only
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: [rel_obs_x, rel_obs_y, obs_r, rel_tgt_x, rel_tgt_y, tgt_r, vel_x, vel_y]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0.0, -20, -20, 0.1, -2, -2], dtype=np.float32),
            high=np.array([20, 20, 5.0, 20, 20, 3.0, 2, 2], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obstacle_pos = np.zeros(2)
        self.obstacle_radius = 1.0
        self.target_pos = np.zeros(2)
        self.target_radius = 1.0
        self.prev_dist2target = 0.0

        # Parametric QP with 1 CBF constraint
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h_param = cp.Parameter(2)
        self._rhs_param = cp.Parameter()
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [self._L_g_h_param @ self._u >= self._rhs_param]
        self._prob = cp.Problem(cost, constraints)

    def _compute_k_nom(self):
        """Fixed proportional controller: always head toward target at fixed speed."""
        diff = self.target_pos - self.robot_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2)
        return self.k_nom_speed * diff / dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        self.obstacle_pos = np.array([
            self.np_random.uniform(3.0, 7.0),
            self.np_random.uniform(-2.0, 2.0),
        ])
        self.obstacle_radius = 1.0

        target_y = self.np_random.uniform(-4.0, 4.0)
        self.target_pos = np.array([10.0, target_y])
        self.target_radius = self.np_random.uniform(0.5, 2.0)

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        alpha = float(action[0])
        k_nom = self._compute_k_nom()

        # CBF
        x_diff = self.robot_pos - self.obstacle_pos
        h = np.sum(x_diff**2) - self.obstacle_radius**2
        L_g_h = 2 * x_diff

        self._k_nom_param.value = k_nom
        self._L_g_h_param.value = L_g_h
        self._rhs_param.value = -alpha * h

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

        dist2obs = np.linalg.norm(self.robot_pos - self.obstacle_pos) - self.obstacle_radius
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        reward = 0.0

        if dist2obs < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0
            reward -= 0.5  # time penalty
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        if self.robot_pos[0] < -2.0 or self.robot_pos[0] > 12.0 or abs(self.robot_pos[1]) > 6.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {
            "safe_u": safe_u, "k_nom": k_nom, "h": float(dist2obs),
            "min_obs_dist": float(dist2obs), "alpha": float(alpha),
        }

    def _get_obs(self):
        rel_obs = self.obstacle_pos - self.robot_pos
        rel_tgt = self.target_pos - self.robot_pos
        return np.array([
            rel_obs[0], rel_obs[1], self.obstacle_radius,
            rel_tgt[0], rel_tgt[1], self.target_radius,
            self.velocity[0], self.velocity[1],
        ], dtype=np.float32)
