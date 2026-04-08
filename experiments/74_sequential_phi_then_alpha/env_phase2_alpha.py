"""
Exp 74 Phase 2: Learn ALPHA with frozen phi from Phase 1.

Loads the Phase 1 phi model and queries it each step for the phi action.
The alpha policy learns on top of the frozen phi behavior.

Discrete actions: 3 choices (alpha_down / alpha_stay / alpha_up), step=0.5.
Based on exp 73's degraded A* environment.

SHIELD SDF pipeline (Eq 19-22) with single constraint.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
import heapq
from stable_baselines3 import PPO

LAMBDA_H = 1.0
GAMMA_H = 0.5

# A* planner config -- deliberately degraded
GRID_RESOLUTION = 1.0
GRID_X_RANGE = (-5, 165)
GRID_Y_RANGE = (-22, 22)
INFLATION_MARGIN = 0.0   # NO inflation -- A* plans right along edges
K_PROPORTIONAL = 4.0
WAYPOINT_THRESHOLD = 3.0
REPLAN_INTERVAL = 30     # replan every 3 seconds -- path gets stale

# Moving obstacle config
OBS_SPEED_RANGE = (0.3, 1.0)  # m/s per obstacle

# Discrete alpha config
ALPHA_STEP = 0.5
ALPHA_INIT = 2.0
ALPHA_MIN, ALPHA_MAX = 0.1, 5.0

# Phi config (driven by frozen model)
PHI_STEP = 0.5
PHI_INIT = 1.0
PHI_MIN, PHI_MAX = 0.01, 10.0


class AlphaWithFrozenPhiEnv(gym.Env):
    def __init__(self, phi_model_path="./models_dynamic/phi_only_model",
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

        # Persistent alpha and phi state
        self.alpha = ALPHA_INIT
        self.phi = PHI_INIT

        # Load frozen phi model
        self.phi_model = PPO.load(phi_model_path)

        # Moving obstacle velocities
        self.obs_vel = [np.zeros(2) for _ in range(3)]

        # A* state
        self.path = []
        self.path_idx = 0
        self.steps_since_replan = 0

        # ACTION: Discrete(3) = alpha_down / alpha_stay / alpha_up
        self.action_space = spaces.Discrete(3)

        # OBSERVATION: 3 obs x 3 + target 3 + velocity 2 + alpha 1 + phi 1 = 16
        # Alpha policy sees both alpha and phi
        self.observation_space = spaces.Box(
            low=np.array([-180, -35, 0, -180, -35, 0, -180, -35, 0,
                          -180, -35, 0.1, -7, -7,
                          ALPHA_MIN, PHI_MIN], dtype=np.float32),
            high=np.array([180, 35, 10, 180, 35, 10, 180, 35, 10,
                           180, 35, 5.0, 7, 7,
                           ALPHA_MAX, PHI_MAX], dtype=np.float32),
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

    def _build_phi_obs(self):
        """Build 15-dim observation for the frozen phi model (same format as Phase 1)."""
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        parts.extend([self.phi])
        return np.array(parts, dtype=np.float32)

    def _place_obstacles(self):
        placed = []
        x_bands = [(20.0, 50.0), (60.0, 100.0), (110.0, 140.0)]
        for x_low, x_high in x_bands:
            x = self.np_random.uniform(x_low, x_high)
            y = self.np_random.uniform(-10.0, 10.0)
            placed.append(np.array([x, y]))
        return placed

    def _sample_obs_velocities(self):
        vels = []
        for _ in range(3):
            speed = self.np_random.uniform(*OBS_SPEED_RANGE)
            angle = self.np_random.uniform(0, 2 * np.pi)
            vels.append(speed * np.array([np.cos(angle), np.sin(angle)]))
        return vels

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

    def _world_to_grid(self, pos):
        gx = int((pos[0] - GRID_X_RANGE[0]) / GRID_RESOLUTION)
        gy = int((pos[1] - GRID_Y_RANGE[0]) / GRID_RESOLUTION)
        return (gx, gy)

    def _grid_to_world(self, gx, gy):
        wx = gx * GRID_RESOLUTION + GRID_X_RANGE[0] + GRID_RESOLUTION / 2
        wy = gy * GRID_RESOLUTION + GRID_Y_RANGE[0] + GRID_RESOLUTION / 2
        return np.array([wx, wy])

    def _build_occupancy_grid(self):
        nx = int((GRID_X_RANGE[1] - GRID_X_RANGE[0]) / GRID_RESOLUTION)
        ny = int((GRID_Y_RANGE[1] - GRID_Y_RANGE[0]) / GRID_RESOLUTION)
        grid = np.zeros((nx, ny), dtype=bool)
        for i in range(3):
            r = self.estimated_radius[i] + INFLATION_MARGIN
            cx, cy = self._world_to_grid(self.obs_pos[i])
            r_cells = int(r / GRID_RESOLUTION) + 1
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < nx and 0 <= gy < ny:
                        world_pt = self._grid_to_world(gx, gy)
                        if np.linalg.norm(world_pt - self.obs_pos[i]) < r:
                            grid[gx, gy] = True
        return grid

    def _astar(self, start_pos, goal_pos):
        grid = self._build_occupancy_grid()
        nx, ny = grid.shape
        start = self._world_to_grid(start_pos)
        goal = self._world_to_grid(goal_pos)
        start = (max(0, min(start[0], nx-1)), max(0, min(start[1], ny-1)))
        goal = (max(0, min(goal[0], nx-1)), max(0, min(goal[1], ny-1)))

        if grid[goal[0], goal[1]]:
            best_d = float('inf')
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    gx, gy = goal[0]+dx, goal[1]+dy
                    if 0 <= gx < nx and 0 <= gy < ny and not grid[gx, gy]:
                        d = abs(dx) + abs(dy)
                        if d < best_d:
                            best_d = d
                            goal = (gx, gy)

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(self._grid_to_world(current[0], current[1]))
                    current = came_from[current]
                path.reverse()
                return path
            if current in closed:
                continue
            closed.add(current)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx_, ny_ = current[0]+dx, current[1]+dy
                if 0 <= nx_ < nx and 0 <= ny_ < ny and not grid[nx_, ny_]:
                    cost = 1.0 if dx == 0 or dy == 0 else 1.414
                    tent_g = g_score[current] + cost
                    if tent_g < g_score.get((nx_, ny_), float('inf')):
                        g_score[(nx_, ny_)] = tent_g
                        h = abs(nx_ - goal[0]) + abs(ny_ - goal[1])
                        heapq.heappush(open_set, (tent_g + h, (nx_, ny_)))
                        came_from[(nx_, ny_)] = current

        return [goal_pos.copy()]

    def _get_nav_command(self):
        if self.path_idx >= len(self.path):
            direction = self.target_pos - self.robot_pos
        else:
            waypoint = self.path[self.path_idx]
            direction = waypoint - self.robot_pos
            if np.linalg.norm(direction) < WAYPOINT_THRESHOLD:
                self.path_idx += 1
                if self.path_idx < len(self.path):
                    direction = self.path[self.path_idx] - self.robot_pos
                else:
                    direction = self.target_pos - self.robot_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.array([0.0, 0.0])
        unit_dir = direction / dist
        speed = min(K_PROPORTIONAL * dist, 6.0)
        return np.clip(unit_dir * speed, -6.0, 6.0)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0

        self.alpha = ALPHA_INIT
        self.phi = PHI_INIT

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
            jitter = self.np_random.uniform(self.jitter_range[0], self.jitter_range[1])
            error = self.sensor_bias + jitter
            self.estimated_radius[i] = max(self.true_radius[i] + error, 1.0)

        bias_mag = self.np_random.uniform(
            self.bias_magnitude_range[0], self.bias_magnitude_range[1])
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_mag * np.array([np.cos(angle), np.sin(angle)])

        self.obs_vel = self._sample_obs_velocities()

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        sdf_vals = [np.linalg.norm(self.robot_pos - self.obs_pos[i]) - self.estimated_radius[i]
                    for i in range(3)]
        ci = int(np.argmin(sdf_vals))
        diff = self.robot_pos - self.obs_pos[ci]
        d = max(np.linalg.norm(diff), 1e-6)
        self.prev_e_i = diff / d
        self.prev_closest_idx = ci

        self.path = self._astar(self.robot_pos, self.target_pos)
        self.path_idx = 0
        self.steps_since_replan = 0

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        self.current_step += 1
        self.steps_since_replan += 1

        # Step 1: Query frozen phi model for phi action
        phi_obs = self._build_phi_obs()
        phi_action_raw, _ = self.phi_model.predict(phi_obs, deterministic=True)
        phi_action = int(phi_action_raw)
        phi_delta = (phi_action - 1) * PHI_STEP
        self.phi = np.clip(self.phi + phi_delta, PHI_MIN, PHI_MAX)

        # Step 2: Decode alpha action from current policy
        alpha_action = action  # 0=down, 1=stay, 2=up
        alpha_delta = (alpha_action - 1) * ALPHA_STEP
        self.alpha = np.clip(self.alpha + alpha_delta, ALPHA_MIN, ALPHA_MAX)

        alpha = self.alpha
        phi = self.phi

        # Move obstacles
        self._move_obstacles()

        # Replan infrequently -- path gets stale
        if self.steps_since_replan >= REPLAN_INTERVAL:
            self.path = self._astar(self.robot_pos, self.target_pos)
            self.path_idx = 0
            self.steps_since_replan = 0

        k_nom = self._get_nav_command()

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
            "alpha_action": alpha_action, "phi_action": phi_action,
            "h_val": float(h_val),
            "true_radius": list(self.true_radius),
            "estimated_radius": list(self.estimated_radius),
            "sensor_bias": float(self.sensor_bias),
            "bias": self.bias.copy(),
        }

    def _get_obs(self):
        """16-dim observation: alpha policy sees everything including phi."""
        parts = []
        for i in range(3):
            rel = self.obs_pos[i] - self.robot_pos
            parts.extend([rel[0], rel[1], self.estimated_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        parts.extend([self.alpha, self.phi])
        return np.array(parts, dtype=np.float32)
