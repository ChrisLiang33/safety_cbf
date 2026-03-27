"""
2-obstacle alpha-only CBF environment — squeeze gate design.
Agent outputs [alpha] only. k_nom is a fixed proportional controller.

Obstacles straddle the direct path to target, creating a gate.
Low alpha → both CBFs squeeze the feasible velocity set → robot crawls.
High alpha → constraints relaxed → robot passes at full speed.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class AlphaOnly2ObsDynamicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 2.0

        # ACTION: [alpha] only
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: [rel_obs1_x, rel_obs1_y, obs1_r, rel_obs2_x, rel_obs2_y, obs2_r,
        #        rel_tgt_x, rel_tgt_y, tgt_r, vel_x, vel_y]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0.0, -20, -20, 0.0, -20, -20, 0.1, -2, -2], dtype=np.float32),
            high=np.array([20, 20, 5.0, 20, 20, 5.0, 20, 20, 3.0, 2, 2], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obstacle1_pos = np.zeros(2)
        self.obstacle1_radius = 1.0
        self.obstacle2_pos = np.zeros(2)
        self.obstacle2_radius = 1.0
        self.target_pos = np.zeros(2)
        self.target_radius = 1.0
        self.prev_dist2target = 0.0

        # Parametric QP with 2 CBF constraints
        self._u = cp.Variable(2)
        self._k_nom_param = cp.Parameter(2)
        self._L_g_h1_param = cp.Parameter(2)
        self._rhs1_param = cp.Parameter()
        self._L_g_h2_param = cp.Parameter(2)
        self._rhs2_param = cp.Parameter()
        cost = cp.Minimize(0.5 * cp.sum_squares(self._u - self._k_nom_param))
        constraints = [
            self._L_g_h1_param @ self._u >= self._rhs1_param,
            self._L_g_h2_param @ self._u >= self._rhs2_param,
        ]
        self._prob = cp.Problem(cost, constraints)

    def _compute_k_nom(self):
        """Fixed proportional controller: head toward target at fixed speed."""
        diff = self.target_pos - self.robot_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(2)
        return self.k_nom_speed * diff / dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        # Target — small y range to keep path roughly horizontal
        target_y = self.np_random.uniform(-1.5, 1.5)
        self.target_pos = np.array([10.0, target_y])
        self.target_radius = self.np_random.uniform(0.5, 1.5)

        # Obstacles straddle the DIRECT path from start to target
        obs_x = self.np_random.uniform(4.0, 6.0)
        path_y_at_gate = target_y * obs_x / 10.0  # y-intercept of direct line at obs_x
        # offset = distance from path center to obstacle center
        # surface gap = 2*(offset - radius). offset in [1.3, 2.0] → gap in [0.6, 2.0]m
        offset = self.np_random.uniform(1.3, 2.0)

        self.obstacle1_pos = np.array([obs_x, path_y_at_gate + offset])
        self.obstacle1_radius = 1.0
        self.obstacle2_pos = np.array([obs_x, path_y_at_gate - offset])
        self.obstacle2_radius = 1.0

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        alpha = float(action[0])
        k_nom = self._compute_k_nom()

        # CBF for obstacle 1
        x_diff1 = self.robot_pos - self.obstacle1_pos
        h1 = np.sum(x_diff1**2) - self.obstacle1_radius**2
        L_g_h1 = 2 * x_diff1

        # CBF for obstacle 2
        x_diff2 = self.robot_pos - self.obstacle2_pos
        h2 = np.sum(x_diff2**2) - self.obstacle2_radius**2
        L_g_h2 = 2 * x_diff2

        # Solve QP
        self._k_nom_param.value = k_nom
        self._L_g_h1_param.value = L_g_h1
        self._rhs1_param.value = -alpha * h1
        self._L_g_h2_param.value = L_g_h2
        self._rhs2_param.value = -alpha * h2

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

        dist2obs1 = np.linalg.norm(self.robot_pos - self.obstacle1_pos) - self.obstacle1_radius
        dist2obs2 = np.linalg.norm(self.robot_pos - self.obstacle2_pos) - self.obstacle2_radius
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        min_obs_dist = min(dist2obs1, dist2obs2)

        terminated = False
        reward = 0.0

        if min_obs_dist < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0
            reward -= 1.0  # time penalty — makes crawling very costly
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        if self.robot_pos[0] < -2.0 or self.robot_pos[0] > 12.0 or abs(self.robot_pos[1]) > 8.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {
            "safe_u": safe_u, "k_nom": k_nom,
            "h1": float(dist2obs1), "h2": float(dist2obs2),
            "min_obs_dist": float(min_obs_dist), "alpha": float(alpha),
        }

    def _get_obs(self):
        rel1 = self.obstacle1_pos - self.robot_pos
        rel2 = self.obstacle2_pos - self.robot_pos
        rel_t = self.target_pos - self.robot_pos
        return np.array([
            rel1[0], rel1[1], self.obstacle1_radius,
            rel2[0], rel2[1], self.obstacle2_radius,
            rel_t[0], rel_t[1], self.target_radius,
            self.velocity[0], self.velocity[1],
        ], dtype=np.float32)
