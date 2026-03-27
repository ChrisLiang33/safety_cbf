"""
3-obstacle weave with MOVING obstacles.
Agent controls [kx, ky, alpha].

Obstacles drift with bounded random velocity, reversing when they stray
too far from their initial position (oscillating drift).

CBF with moving obstacles:
    h = ||p_r - p_o||^2 - r^2
    h_dot = Lg_h @ u + Lf_h
    Lg_h  = 2(p_r - p_o)                     (same as static)
    Lf_h  = -2(p_r - p_o)^T * v_obs          (drift from obstacle motion)

    Constraint: Lg_h @ u >= -alpha * h - Lf_h
              = Lg_h @ u >= -alpha * h + 2(p_r - p_o)^T * v_obs

Observation includes obstacle velocities so the agent can anticipate motion.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp


class MovingObsWeaveDynamicEnv(gym.Env):
    def __init__(self, obs_speed_range=(0.1, 0.5), obs_drift_limit=3.0):
        super().__init__()
        self.dt = 0.1
        self.k_nom_speed = 3.0
        self.obs_speed_range = obs_speed_range
        self.obs_drift_limit = obs_drift_limit  # max drift from initial position

        # ACTION: [kx, ky, alpha]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, 0.1], dtype=np.float32),
            high=np.array([3.0, 3.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # OBS: [rel_obs1_xy, r1, obs1_vel_xy,
        #       rel_obs2_xy, r2, obs2_vel_xy,
        #       rel_obs3_xy, r3, obs3_vel_xy,
        #       rel_tgt_xy, tgt_r, vel_xy] = 20 dims
        self.observation_space = spaces.Box(
            low=np.array([-120, -25, 0, -1, -1,
                          -120, -25, 0, -1, -1,
                          -120, -25, 0, -1, -1,
                          -120, -25, 0.1, -4, -4], dtype=np.float32),
            high=np.array([120, 25, 10, 1, 1,
                           120, 25, 10, 1, 1,
                           120, 25, 10, 1, 1,
                           120, 25, 5.0, 4, 4], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
        self.obs_pos = [np.zeros(2) for _ in range(3)]
        self.obs_vel = [np.zeros(2) for _ in range(3)]
        self.obs_init_pos = [np.zeros(2) for _ in range(3)]  # anchor for drift limit
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

    def _sample_obs_velocity(self):
        """Sample a random velocity for an obstacle."""
        speed = self.np_random.uniform(*self.obs_speed_range)
        angle = self.np_random.uniform(0, 2 * np.pi)
        return np.array([speed * np.cos(angle), speed * np.sin(angle)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)

        target_y = self.np_random.uniform(-3.0, 3.0)
        self.target_pos = np.array([100.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        # 3 obstacles in slalom pattern (same layout as exp 9)
        x1 = self.np_random.uniform(20.0, 35.0)
        y1 = self.np_random.uniform(5.0, 8.0)
        self.obs_pos[0] = np.array([x1, y1])
        self.obs_init_pos[0] = self.obs_pos[0].copy()
        self.obs_vel[0] = self._sample_obs_velocity()
        self.obs_radius[0] = 5.0

        x2 = self.np_random.uniform(45.0, 60.0)
        y2 = self.np_random.uniform(-8.0, -5.0)
        self.obs_pos[1] = np.array([x2, y2])
        self.obs_init_pos[1] = self.obs_pos[1].copy()
        self.obs_vel[1] = self._sample_obs_velocity()
        self.obs_radius[1] = 5.0

        x3 = self.np_random.uniform(65.0, 80.0)
        y3 = self.np_random.uniform(5.0, 8.0)
        self.obs_pos[2] = np.array([x3, y3])
        self.obs_init_pos[2] = self.obs_pos[2].copy()
        self.obs_vel[2] = self._sample_obs_velocity()
        self.obs_radius[2] = 5.0

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)
        return self._get_obs(), {}

    def _update_obstacles(self):
        """Move obstacles and bounce off drift limits."""
        for i in range(3):
            self.obs_pos[i] += self.obs_vel[i] * self.dt
            # Bounce: if drifted too far from initial position, reverse velocity
            drift = np.linalg.norm(self.obs_pos[i] - self.obs_init_pos[i])
            if drift > self.obs_drift_limit:
                # Reflect velocity away from drift direction
                drift_dir = (self.obs_pos[i] - self.obs_init_pos[i]) / (drift + 1e-8)
                # Reverse the component along drift direction
                self.obs_vel[i] -= 2 * np.dot(self.obs_vel[i], drift_dir) * drift_dir

    def step(self, action):
        kx = float(action[0])
        ky = float(action[1])
        alpha = float(action[2])
        k_nom = np.array([kx, ky])

        # CBF for each MOVING obstacle
        h_vals = []
        L_g_h_vals = []
        Lf_h_vals = []
        for i in range(3):
            x_diff = self.robot_pos - self.obs_pos[i]
            h = np.sum(x_diff**2) - self.obs_radius[i]**2
            L_g_h = 2 * x_diff
            # Drift term from obstacle motion: Lf_h = -2 * x_diff^T * v_obs
            Lf_h = -2 * np.dot(x_diff, self.obs_vel[i])
            h_vals.append(h)
            L_g_h_vals.append(L_g_h)
            Lf_h_vals.append(Lf_h)

        # QP: Lg_h @ u >= -alpha * h - Lf_h
        self._k_nom_param.value = k_nom
        for i in range(3):
            self._L_g_h[i].value = L_g_h_vals[i]
            self._rhs[i].value = -alpha * h_vals[i] - Lf_h_vals[i]

        try:
            self._prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            safe_u = self._u.value
            if safe_u is None:
                safe_u = np.array([0.0, 0.0])
        except Exception:
            safe_u = np.array([0.0, 0.0])

        safe_u = np.clip(safe_u, -3.0, 3.0)

        # Update robot
        self.robot_pos += safe_u * self.dt
        self.velocity = safe_u.copy()

        # Update obstacles (they move too)
        self._update_obstacles()

        # Distances (after both robot and obstacles moved)
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
            "safe_u": safe_u, "k_nom": k_nom,
            "dist2obs": dist2obs, "min_obs_dist": float(min_obs_dist),
            "alpha": float(alpha),
            "obs_vel": [v.copy() for v in self.obs_vel],
        }

    def _get_obs(self):
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.obs_radius[i],
                          self.obs_vel[i][0], self.obs_vel[i][1]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
