"""
3-obstacle slalom with CBF intervention penalty.
Agent controls [kx, ky, alpha]. Same layout as exp 9 but adds a penalty
proportional to ||u_safe - k_nom|| to incentivize the agent to preemptively
lower alpha so the QP doesn't need to correct its steering.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class CBFPenaltySlalomEnv(gym.Env):
    def __init__(self, cbf_penalty_weight=5.0):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 3.0
        self.cbf_penalty_weight = cbf_penalty_weight

        # ACTION: [kx, ky, alpha]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.1], dtype=np.float32),
            high=np.array([3.0, 3.0, 5.0], dtype=np.float32),
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
        self.obs_radius = [5.0, 5.0, 5.0]
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        # Same slalom layout as exp 9
        x1 = self.np_random.uniform(20.0, 35.0)
        y1 = self.np_random.uniform(5.0, 8.0)
        self.obs_pos[0] = np.array([x1, y1])
        self.obs_radius[0] = 5.0

        x2 = self.np_random.uniform(45.0, 60.0)
        y2 = self.np_random.uniform(-8.0, -5.0)
        self.obs_pos[1] = np.array([x2, y2])
        self.obs_radius[1] = 5.0

        x3 = self.np_random.uniform(65.0, 80.0)
        y3 = self.np_random.uniform(5.0, 8.0)
        self.obs_pos[2] = np.array([x3, y3])
        self.obs_radius[2] = 5.0

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def step(self, action):
        kx = float(action[0])
        ky = float(action[1])
        alpha = float(action[2])
        k_nom = np.array([kx, ky])

        # CBF for each obstacle
        h_vals = []
        L_g_h_vals = []
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.obs_radius[i]**2
            L_g_h = 2 * x_diff
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

        # Solve QP
        self._k_nom_param.value = k_nom
        for i in range(3):
            self._L_g_h[i].value = L_g_h_vals[i]
            self._rhs[i].value = -alpha * h_vals[i]

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])

        safe_u = np.clip(safe_u, -3.0, 3.0)

        # CBF intervention: how much did the QP modify the action?
        cbf_intervention = np.linalg.norm(safe_u - k_nom)

        self.robot_pos += safe_u * self.dt
        self.velocity = safe_u.copy()

        dist2obs = []
        for i in range(3):
            dist2obs.append(np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.obs_radius[i])
        min_obs_dist = min(dist2obs)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        reward = 0.0

        if min_obs_dist < 0:
            reward = -100.0
            terminated = True
        else:
            progress = self.prev_dist2target - dist2target
            reward = progress * 50.0
            reward -= 1.0  # time penalty
            # Penalize CBF intervention — incentivize preemptive alpha lowering
            reward -= self.cbf_penalty_weight * cbf_intervention
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        if self.robot_pos[0] < -5.0 or self.robot_pos[0] > 105.0 or abs(self.robot_pos[1]) > 15.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {
            "safe_u": safe_u, "k_nom": k_nom,
            "dist2obs": dist2obs, "min_obs_dist": float(min_obs_dist),
            "alpha": float(alpha), "cbf_intervention": float(cbf_intervention),
        }

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.obs_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
