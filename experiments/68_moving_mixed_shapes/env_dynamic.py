"""
Exp 68: A* + moving mixed-shape obstacles
Agent controls [alpha, phi] ONLY.

Key change: obstacles are a mix of circles, rectangles, and line segments
AND they move with constant velocity during the episode.
  - Obstacle velocity magnitude: U(0.3, 1.0) m/s per obstacle
  - Obstacle velocity direction: U(0, 2*pi) per obstacle
  - Sampled once per episode, constant throughout
  - Obstacles bounce off y-boundaries to stay in play
  - For LINE obstacles, both endpoints (a, b) are also moved

A* replans every 10 steps (more frequent since obstacles move).

This tests whether alpha/phi can learn reactive behavior when obstacles
have mixed shapes AND move — combining exp 61 (moving circular) and
exp 63 (static mixed shapes).

SHIELD SDF pipeline (Eq 19-22) with single constraint.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
import heapq

LAMBDA_H = 1.0
GAMMA_H = 0.5

GRID_RESOLUTION = 1.0
GRID_X_RANGE = (-5, 165)
GRID_Y_RANGE = (-22, 22)
INFLATION_MARGIN = 1.0
K_PROPORTIONAL = 4.0
WAYPOINT_THRESHOLD = 3.0
REPLAN_INTERVAL = 10  # more frequent — obstacles move
WAYPOINT_NOISE = 2.0  # meters — random perturbation to each A* waypoint

# Shape types
CIRCLE = 0
RECTANGLE = 1
LINE = 2

# Moving obstacle config
OBS_SPEED_RANGE = (0.3, 1.0)  # m/s per obstacle


def sdf_circle(p, center, radius):
    """Signed distance from point p to circle surface."""
    return np.linalg.norm(p - center) - radius


def sdf_rectangle(p, center, half_w, half_h):
    """Signed distance from point p to axis-aligned rectangle."""
    d = np.abs(p - center) - np.array([half_w, half_h])
    outside = np.linalg.norm(np.maximum(d, 0.0))
    inside = min(max(d[0], d[1]), 0.0)
    return outside + inside


def sdf_line(p, a, b, thickness):
    """Signed distance from point p to a line segment (wall) with thickness."""
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8), 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest) - thickness


def sdf_gradient_circle(p, center):
    """Gradient of circle SDF w.r.t. p."""
    diff = p - center
    dist = max(np.linalg.norm(diff), 1e-6)
    return diff / dist


def sdf_gradient_rectangle(p, center, half_w, half_h):
    """Gradient of rectangle SDF w.r.t. p (approximate)."""
    d = np.abs(p - center) - np.array([half_w, half_h])
    if d[0] > 0 and d[1] > 0:
        # corner case
        corner = np.sign(p - center) * np.maximum(d, 0.0)
        norm = max(np.linalg.norm(corner), 1e-6)
        return corner / norm
    elif d[0] > d[1]:
        return np.array([np.sign(p[0] - center[0]), 0.0])
    else:
        return np.array([0.0, np.sign(p[1] - center[1])])


def sdf_gradient_line(p, a, b):
    """Gradient of line SDF w.r.t. p."""
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-8), 0.0, 1.0)
    closest = a + t * ab
    diff = p - closest
    dist = max(np.linalg.norm(diff), 1e-6)
    return diff / dist


class AStarMovingMixedShapesEnv(gym.Env):
    def __init__(self, systematic_bias_range=(-0.6, 0.4), jitter_range=(-0.2, 0.2),
                 bias_magnitude_range=(0.3, 1.0)):
        super().__init__()
        self.dt = 0.1
        self.systematic_bias_range = systematic_bias_range
        self.jitter_range = jitter_range
        self.bias_magnitude_range = bias_magnitude_range
        self.bias = np.zeros(2)
        self.sensor_bias = 0.0
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0
        self.n_obs = 4  # mix of shapes

        # Obstacle data: list of dicts with shape info
        self.obstacles = []
        # For collision detection (true) and CBF (estimated)
        self.true_sdfs = []     # functions
        self.est_sdfs = []      # functions with sensor error
        self.est_effective_radius = []  # for observation space

        # Moving obstacle velocities
        self.obs_vel = [np.zeros(2) for _ in range(self.n_obs)]

        # A* state
        self.path = []
        self.path_idx = 0
        self.steps_since_replan = 0

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.01], dtype=np.float32),
            high=np.array([5.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        # 4 obstacles x 3 (rel_x, rel_y, effective_radius) + target x 3 + velocity x 2 = 17
        self.observation_space = spaces.Box(
            low=np.array([-180, -35, 0] * self.n_obs + [-180, -35, 0.1, -7, -7], dtype=np.float32),
            high=np.array([180, 35, 15] * self.n_obs + [180, 35, 5.0, 7, 7], dtype=np.float32),
            dtype=np.float32,
        )

        self.robot_pos = np.zeros(2)
        self.velocity = np.zeros(2)
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

    def _generate_obstacles(self):
        """Generate a mix of circles, rectangles, and line segments."""
        obstacles = []
        x_bands = [(20.0, 50.0), (55.0, 85.0), (90.0, 120.0), (125.0, 145.0)]

        for x_low, x_high in x_bands:
            cx = self.np_random.uniform(x_low, x_high)
            cy = self.np_random.uniform(-10.0, 10.0)
            shape = self.np_random.integers(0, 3)  # 0=circle, 1=rect, 2=line

            if shape == CIRCLE:
                r = self.np_random.uniform(3.0, 7.0)
                obstacles.append({
                    "type": CIRCLE, "center": np.array([cx, cy]),
                    "radius": r, "effective_radius": r,
                })
            elif shape == RECTANGLE:
                half_w = self.np_random.uniform(2.0, 5.0)
                half_h = self.np_random.uniform(2.0, 5.0)
                obstacles.append({
                    "type": RECTANGLE, "center": np.array([cx, cy]),
                    "half_w": half_w, "half_h": half_h,
                    "effective_radius": max(half_w, half_h),
                })
            else:  # LINE
                length = self.np_random.uniform(6.0, 12.0)
                angle = self.np_random.uniform(0, np.pi)
                dx = length / 2 * np.cos(angle)
                dy = length / 2 * np.sin(angle)
                a = np.array([cx - dx, cy - dy])
                b = np.array([cx + dx, cy + dy])
                thickness = self.np_random.uniform(0.5, 1.5)
                obstacles.append({
                    "type": LINE, "center": np.array([cx, cy]),
                    "a": a, "b": b, "thickness": thickness,
                    "effective_radius": length / 2 + thickness,
                })

        return obstacles

    def _sample_obs_velocities(self):
        """Sample constant velocity for each obstacle."""
        vels = []
        for _ in range(self.n_obs):
            speed = self.np_random.uniform(*OBS_SPEED_RANGE)
            angle = self.np_random.uniform(0, 2 * np.pi)
            vels.append(speed * np.array([np.cos(angle), np.sin(angle)]))
        return vels

    def _move_obstacles(self):
        """Move obstacles and bounce off y-boundaries."""
        for i in range(self.n_obs):
            vel_step = self.obs_vel[i] * self.dt
            self.obstacles[i]["center"] += vel_step
            # For LINE obstacles, also move the endpoints
            if self.obstacles[i]["type"] == LINE:
                self.obstacles[i]["a"] += vel_step
                self.obstacles[i]["b"] += vel_step
            # Bounce off y-boundaries
            if self.obstacles[i]["center"][1] > 18.0:
                self.obstacles[i]["center"][1] = 18.0
                self.obs_vel[i][1] *= -1
            elif self.obstacles[i]["center"][1] < -18.0:
                self.obstacles[i]["center"][1] = -18.0
                self.obs_vel[i][1] *= -1
            # Clamp x to stay in environment
            self.obstacles[i]["center"][0] = np.clip(
                self.obstacles[i]["center"][0], 5.0, 155.0)

    def _obs_sdf_true(self, p, obs):
        """Compute true SDF for an obstacle."""
        if obs["type"] == CIRCLE:
            return sdf_circle(p, obs["center"], obs["radius"])
        elif obs["type"] == RECTANGLE:
            return sdf_rectangle(p, obs["center"], obs["half_w"], obs["half_h"])
        else:  # LINE
            return sdf_line(p, obs["a"], obs["b"], obs["thickness"])

    def _obs_sdf_est(self, p, obs, error):
        """Compute estimated SDF (with sensor error applied to size)."""
        if obs["type"] == CIRCLE:
            return sdf_circle(p, obs["center"], obs["radius"] + error)
        elif obs["type"] == RECTANGLE:
            return sdf_rectangle(p, obs["center"], obs["half_w"] + error, obs["half_h"] + error)
        else:  # LINE
            return sdf_line(p, obs["a"], obs["b"], obs["thickness"] + error)

    def _obs_gradient(self, p, obs):
        """Compute SDF gradient for an obstacle."""
        if obs["type"] == CIRCLE:
            return sdf_gradient_circle(p, obs["center"])
        elif obs["type"] == RECTANGLE:
            return sdf_gradient_rectangle(p, obs["center"], obs["half_w"], obs["half_h"])
        else:  # LINE
            return sdf_gradient_line(p, obs["a"], obs["b"])

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
        for obs_idx, obs in enumerate(self.obstacles):
            er = self.est_effective_radius[obs_idx] + INFLATION_MARGIN
            cx, cy = self._world_to_grid(obs["center"])
            r_cells = int(er / GRID_RESOLUTION) + 2
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if 0 <= gx < nx and 0 <= gy < ny:
                        world_pt = self._grid_to_world(gx, gy)
                        if self._obs_sdf_est(world_pt, obs, self.sensor_errors[obs_idx]) < INFLATION_MARGIN:
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
            for ddx in range(-5, 6):
                for ddy in range(-5, 6):
                    gx, gy = goal[0]+ddx, goal[1]+ddy
                    if 0 <= gx < nx and 0 <= gy < ny and not grid[gx, gy]:
                        d = abs(ddx) + abs(ddy)
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
            for ddx, ddy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx_, ny_ = current[0]+ddx, current[1]+ddy
                if 0 <= nx_ < nx and 0 <= ny_ < ny and not grid[nx_, ny_]:
                    cost = 1.0 if ddx == 0 or ddy == 0 else 1.414
                    tent_g = g_score[current] + cost
                    if tent_g < g_score.get((nx_, ny_), float('inf')):
                        g_score[(nx_, ny_)] = tent_g
                        h = abs(nx_ - goal[0]) + abs(ny_ - goal[1])
                        heapq.heappush(open_set, (tent_g + h, (nx_, ny_)))
                        came_from[(nx_, ny_)] = current
        return [goal_pos.copy()]

    def _perturb_path(self, path):
        """Add random noise to waypoints — makes A* stochastic."""
        perturbed = []
        for wp in path:
            noise = self.np_random.uniform(-WAYPOINT_NOISE, WAYPOINT_NOISE, size=2)
            perturbed.append(wp + noise)
        return perturbed

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
        # Find closest obstacle using estimated SDF
        sdf_vals = []
        for i, obs in enumerate(self.obstacles):
            sdf_vals.append(self._obs_sdf_est(robot_pos, obs, self.sensor_errors[i]))
        closest_idx = int(np.argmin(sdf_vals))
        obs = self.obstacles[closest_idx]
        sdf_est = sdf_vals[closest_idx]

        e_i_current = self._obs_gradient(robot_pos, obs)
        e_i = self.prev_e_i

        projected_dist = np.dot(robot_pos - obs["center"], e_i) - (obs["effective_radius"] + self.sensor_errors[closest_idx])

        if np.dot(robot_pos - obs["center"], e_i) >= 0:
            exp_term = np.exp(-GAMMA_H * projected_dist)
            h_val = LAMBDA_H * (1.0 - exp_term)
            Lgh = LAMBDA_H * GAMMA_H * exp_term * e_i
        else:
            R_eff = obs["effective_radius"] + self.sensor_errors[closest_idx]
            exp_gR = np.exp(GAMMA_H * R_eff)
            grad_h = LAMBDA_H * GAMMA_H * exp_gR * e_i
            h_at_boundary = LAMBDA_H * (1.0 - exp_gR)
            h_val = np.dot(grad_h, robot_pos - obs["center"]) + h_at_boundary
            Lgh = grad_h

        return h_val, Lgh, closest_idx, e_i_current

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0.0, 0.0])
        self.velocity = np.zeros(2)
        self.current_step = 0
        self.prev_e_i = np.array([1.0, 0.0])
        self.prev_closest_idx = 0

        target_y = self.np_random.uniform(-5.0, 5.0)
        self.target_pos = np.array([150.0, target_y])
        self.target_radius = self.np_random.uniform(1.5, 3.0)

        self.obstacles = self._generate_obstacles()

        self.sensor_bias = self.np_random.uniform(
            self.systematic_bias_range[0], self.systematic_bias_range[1])
        self.sensor_errors = []
        self.est_effective_radius = []
        for obs in self.obstacles:
            jitter = self.np_random.uniform(self.jitter_range[0], self.jitter_range[1])
            error = self.sensor_bias + jitter
            self.sensor_errors.append(error)
            self.est_effective_radius.append(max(obs["effective_radius"] + error, 1.0))

        bias_mag = self.np_random.uniform(
            self.bias_magnitude_range[0], self.bias_magnitude_range[1])
        angle = self.np_random.uniform(0, 2 * np.pi)
        self.bias = bias_mag * np.array([np.cos(angle), np.sin(angle)])

        # Sample obstacle velocities
        self.obs_vel = self._sample_obs_velocities()

        self.prev_dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        sdf_vals = [self._obs_sdf_est(self.robot_pos, obs, self.sensor_errors[i])
                    for i, obs in enumerate(self.obstacles)]
        ci = int(np.argmin(sdf_vals))
        self.prev_e_i = self._obs_gradient(self.robot_pos, self.obstacles[ci])
        self.prev_closest_idx = ci

        self.path = self._perturb_path(self._astar(self.robot_pos, self.target_pos))
        self.path_idx = 0
        self.steps_since_replan = 0

        return self._get_obs(), {}

    def step(self, action):
        alpha, phi = float(action[0]), float(action[1])
        self.current_step += 1
        self.steps_since_replan += 1

        # Move obstacles
        self._move_obstacles()

        # Replan more frequently since obstacles move
        if self.steps_since_replan >= REPLAN_INTERVAL:
            self.path = self._perturb_path(self._astar(self.robot_pos, self.target_pos))
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

        # Collision detection uses TRUE SDF (no sensor error)
        true_sdfs = [self._obs_sdf_true(self.robot_pos, obs) for obs in self.obstacles]
        min_true_sdf = min(true_sdfs)
        dist2target = np.linalg.norm(self.robot_pos - self.target_pos)

        terminated = False
        truncated = False
        reward = 0.0
        if min_true_sdf < 0:
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
            "safe_u": safe_u, "k_nom": k_nom,
            "min_true_sdf": float(min_true_sdf),
            "alpha": float(alpha), "phi": float(phi),
            "h_val": float(h_val),
            "sensor_bias": float(self.sensor_bias),
            "bias": self.bias.copy(),
        }

    def _get_obs(self):
        parts = []
        for i, obs in enumerate(self.obstacles):
            rel = obs["center"] - self.robot_pos
            parts.extend([rel[0], rel[1], self.est_effective_radius[i]])
        rel_t = self.target_pos - self.robot_pos
        parts.extend([rel_t[0], rel_t[1], self.target_radius])
        parts.extend([self.velocity[0], self.velocity[1]])
        return np.array(parts, dtype=np.float32)
