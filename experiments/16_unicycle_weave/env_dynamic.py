"""
3-obstacle weave with UNICYCLE dynamics.
Agent controls [v_des, omega_des, alpha].

Dynamics:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega

CBF constraint:  Lgh @ [v, omega] >= -alpha * h(x)
where Lgh = [2*dx*cos(theta) + 2*dy*sin(theta), 0]

Key difference from exp 9: the robot has a heading angle and can only
move forward. It must learn to coordinate steering (omega) with speed (v)
and alpha. The CBF can only slow down v — it cannot force the robot to turn.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class UnicycleWeaveDynamicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_v = 3.0
        self.max_omega = 1.5  # rad/s

        # ACTION: [v_des, omega_des, alpha]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.5, 0.1], dtype=np.float32),
            high=np.array([3.0, 1.5, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: [rel_obs1_xy, r1, rel_obs2_xy, r2, rel_obs3_xy, r3,
        #       rel_tgt_xy, tgt_r, v, omega, cos(theta), sin(theta)] = 16 dims
        self.observation_space = spaces.Box(
            low=np.array([-120, -25, 0, -120, -25, 0, -120, -25, 0,
                          -120, -25, 0.1, -4, -2, -1, -1], dtype=np.float32),
            high=np.array([120, 25, 10, 120, 25, 10, 120, 25, 10,
                           120, 25, 5.0, 4, 2, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.theta = 0.0  # heading angle
        self.v = 0.0      # current forward speed
        self.omega = 0.0   # current turn rate
        self.obs_pos = [np.zeros(2) for _ in range(3)]
        self.obs_radius = [5.0, 5.0, 5.0]
        self.target_pos = np.zeros(2)
        self.target_radius = 2.0
        self.prev_dist2target = 0.0

        # Parametric QP: 2 control vars [v, omega], 3 CBF constraints
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
        self.theta = 0.0  # facing right (toward target)
        self.v = 0.0
        self.omega = 0.0

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        # 3 obstacles in slalom pattern (same as exp 9)
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
        v_des = float(action[0])
        omega_des = float(action[1])
        alpha = float(action[2])
        k_nom = np.array([v_des, omega_des])

        cos_th = np.cos(self.theta)
        sin_th = np.sin(self.theta)

        # CBF for each obstacle (unicycle Lie derivative)
        h_vals = []
        L_g_h_vals = []
        for i in range(3):
            dx = self.robot_pos[0] - self.obs_pos[i][0]
            dy = self.robot_pos[1] - self.obs_pos[i][1]
            h = dx**2 + dy**2 - self.obs_radius[i]**2
            # Lgh = [2*dx*cos(theta) + 2*dy*sin(theta), 0]
            # omega has no direct effect on h_dot
            Lgh_v = 2 * dx * cos_th + 2 * dy * sin_th
            L_g_h = np.array([Lgh_v, 0.0])
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)

        # Solve QP: min ||[v, omega] - [v_des, omega_des]||^2  s.t. CBF
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

        safe_v = np.clip(safe_u[0], 0.0, self.max_v)
        safe_omega = np.clip(safe_u[1], -self.max_omega, self.max_omega)

        # Unicycle dynamics update
        self.robot_pos[0] += safe_v * cos_th * self.dt
        self.robot_pos[1] += safe_v * sin_th * self.dt
        self.theta += safe_omega * self.dt
        # Wrap theta to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.v = safe_v
        self.omega = safe_omega

        # Distances
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
            self.prev_dist2target = dist2target

        if dist2target < self.target_radius:
            reward += 100.0
            terminated = True

        # OOB
        if self.robot_pos[0] < -5.0 or self.robot_pos[0] > 105.0 or abs(self.robot_pos[1]) > 15.0:
            reward -= 50.0
            terminated = True

        return self._get_obs(), float(reward), bool(terminated), False, {
            "safe_v": float(safe_v), "safe_omega": float(safe_omega),
            "v_des": float(v_des), "omega_des": float(omega_des),
            "dist2obs": dist2obs, "min_obs_dist": float(min_obs_dist),
            "alpha": float(alpha), "theta": float(self.theta),
        }

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.obs_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.v, self.omega,
                       np.cos(self.theta), np.sin(self.theta)])
        return np.array(parts, dtype=np.float32)
