"""
Exp 67: A* + noisy planner + mixed obstacle shapes -- fully randomized eval.
Action is [alpha, phi]. 4 obstacles: circles, rectangles, line segments.

Outputs:
  plots_randomized/combined_traj_alpha_phi.png
  plots_randomized/scenarios/scenario_*.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, FancyArrowPatch
from stable_baselines3 import PPO
import os

from env_dynamic import AStarNoisyPlannerEnv, CIRCLE, RECTANGLE, LINE

MODEL_PATH = "./models_dynamic/dynamic_10000k_model"
MAX_STEPS = 800
N_OBS = 4

BIAS_MAG_RANGE = (0.3, 1.0)
SYSTEMATIC_BIAS_RANGE = (-0.6, 0.4)
JITTER_RANGE = (-0.2, 0.2)

OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71"]
TIME_MARKER_INTERVAL = 50


def generate_scenarios(n=10, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    x_bands = [(20.0, 50.0), (55.0, 85.0), (90.0, 120.0), (125.0, 145.0)]

    for i in range(n):
        obstacles = []
        sensor_bias = rng.uniform(*SYSTEMATIC_BIAS_RANGE)
        sensor_errors = []

        for x_low, x_high in x_bands:
            cx = rng.uniform(x_low, x_high)
            cy = rng.uniform(-10.0, 10.0)
            shape = rng.randint(0, 3)

            if shape == CIRCLE:
                r = rng.uniform(3.0, 7.0)
                obstacles.append({
                    "type": CIRCLE, "center": np.array([cx, cy]),
                    "radius": r, "effective_radius": r,
                })
            elif shape == RECTANGLE:
                half_w = rng.uniform(2.0, 5.0)
                half_h = rng.uniform(2.0, 5.0)
                obstacles.append({
                    "type": RECTANGLE, "center": np.array([cx, cy]),
                    "half_w": half_w, "half_h": half_h,
                    "effective_radius": max(half_w, half_h),
                })
            else:  # LINE
                length = rng.uniform(6.0, 12.0)
                angle = rng.uniform(0, np.pi)
                dx = length / 2 * np.cos(angle)
                dy = length / 2 * np.sin(angle)
                a = np.array([cx - dx, cy - dy])
                b = np.array([cx + dx, cy + dy])
                thickness = rng.uniform(0.5, 1.5)
                obstacles.append({
                    "type": LINE, "center": np.array([cx, cy]),
                    "a": a, "b": b, "thickness": thickness,
                    "effective_radius": length / 2 + thickness,
                })

            jitter = rng.uniform(*JITTER_RANGE)
            sensor_errors.append(sensor_bias + jitter)

        scenarios.append({
            "name": f"Random_{i+1}",
            "obstacles": obstacles,
            "sensor_errors": sensor_errors,
            "sensor_bias": sensor_bias,
            "target_pos": np.array([150.0, rng.uniform(-5.0, 5.0)]),
            "target_radius": rng.uniform(1.5, 3.0),
            "bias_angle": rng.uniform(0, 2 * np.pi),
            "bias_mag": rng.uniform(*BIAS_MAG_RANGE),
        })
    return scenarios


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.current_step = 0
    env.obstacles = [obs.copy() for obs in scen["obstacles"]]
    # Deep copy arrays inside obstacle dicts
    for i, obs in enumerate(env.obstacles):
        for key in obs:
            if isinstance(obs[key], np.ndarray):
                env.obstacles[i][key] = obs[key].copy()

    env.sensor_bias = scen["sensor_bias"]
    env.sensor_errors = list(scen["sensor_errors"])
    env.est_effective_radius = [
        max(obs["effective_radius"] + scen["sensor_errors"][i], 1.0)
        for i, obs in enumerate(scen["obstacles"])
    ]
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    env.prev_e_i = np.array([1.0, 0.0])
    env.bias = scen["bias_mag"] * np.array([np.cos(scen["bias_angle"]),
                                             np.sin(scen["bias_angle"])])
    env.path = env._perturb_path(env._astar(env.robot_pos, env.target_pos))
    env.path_idx = 0
    env.steps_since_replan = 0
    return env._get_obs()


def path_length(traj_x, traj_y):
    return np.sum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2))


def run_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, speeds = [], [], []
    alphas, phis = [], []
    dist_list = []
    per_obs_dists = [[] for _ in range(N_OBS)]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])

        # True SDF distance per obstacle
        dists = [env._obs_sdf_true(env.robot_pos, obs)
                 for obs in scen["obstacles"]]
        dist_list.append(min(dists))
        for oi in range(N_OBS):
            per_obs_dists[oi].append(dists[oi])

        action, _ = model.predict(obs, deterministic=True)
        alphas.append(float(action[0]))
        phis.append(float(action[1]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speeds.append(np.linalg.norm(info.get("safe_u", np.zeros(2))))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break
    else:
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(dist_list) < 0
    plen = path_length(traj_x, traj_y)
    straight = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))

    return {
        "traj_x": traj_x, "traj_y": traj_y, "speeds": speeds,
        "alphas": alphas, "phis": phis,
        "dist": dist_list, "per_obs_dists": per_obs_dists,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": plen / straight if straight > 0 else 1.0,
    }


def _shape_label(obs):
    if obs["type"] == CIRCLE:
        return f"circle r={obs['radius']:.1f}"
    elif obs["type"] == RECTANGLE:
        return f"rect {obs['half_w']*2:.1f}x{obs['half_h']*2:.1f}"
    else:
        length = np.linalg.norm(obs["b"] - obs["a"])
        return f"wall L={length:.1f} t={obs['thickness']:.1f}"


def draw_obstacle(ax, obs, sensor_error, color="red"):
    """Draw an obstacle shape on the trajectory plot."""
    if obs["type"] == CIRCLE:
        # True shape
        ax.add_patch(plt.Circle(obs["center"], obs["radius"],
                                color=color, alpha=0.2))
        # Estimated shape
        est_r = max(obs["radius"] + sensor_error, 1.0)
        ax.add_patch(plt.Circle(obs["center"], est_r, color="blue", alpha=0.1,
                                linestyle="--", linewidth=1.5, fill=False))
    elif obs["type"] == RECTANGLE:
        hw, hh = obs["half_w"], obs["half_h"]
        c = obs["center"]
        # True shape
        ax.add_patch(Rectangle((c[0]-hw, c[1]-hh), 2*hw, 2*hh,
                                color=color, alpha=0.2))
        # Estimated shape
        ehw = max(hw + sensor_error, 0.5)
        ehh = max(hh + sensor_error, 0.5)
        ax.add_patch(Rectangle((c[0]-ehw, c[1]-ehh), 2*ehw, 2*ehh,
                                color="blue", alpha=0.1, linestyle="--",
                                linewidth=1.5, fill=False))
    else:  # LINE
        a, b = obs["a"], obs["b"]
        thick = obs["thickness"]
        # Draw line segment with thickness
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        perp = np.array([-direction[1], direction[0]]) / length
        # True wall as polygon
        corners = np.array([
            a + perp * thick, b + perp * thick,
            b - perp * thick, a - perp * thick,
        ])
        ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.2)
        # Estimated
        ethick = max(thick + sensor_error, 0.1)
        ecorners = np.array([
            a + perp * ethick, b + perp * ethick,
            b - perp * ethick, a - perp * ethick,
        ])
        ax.plot(np.append(ecorners[:, 0], ecorners[0, 0]),
                np.append(ecorners[:, 1], ecorners[0, 1]),
                color="blue", alpha=0.3, linestyle="--", linewidth=1.5)


def plot_trajectory(ax, scen, r, fontsize_title=14, fontsize_label=12,
                    fontsize_metrics=9, fontsize_cbar=10, fontsize_bias=10,
                    fontsize_radius=9, fontsize_time=9, markersize=5):
    for j, obs in enumerate(scen["obstacles"]):
        err = scen["sensor_errors"][j]
        draw_obstacle(ax, obs, err, color="red")
        label = _shape_label(obs)
        ax.text(obs["center"][0], obs["center"][1] - obs["effective_radius"] - 1.5,
                f"{label}\nerr={err:.2f}",
                fontsize=fontsize_radius - 1, color="red",
                ha="center", fontweight="bold")

    ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"], color="green", alpha=0.3))

    if len(r["traj_x"]) > 1 and len(r["alphas"]) > 0:
        points = np.array([r["traj_x"][:-1], r["traj_y"][:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alpha_vals = np.array(r["alphas"][:len(segments)])
        lc = LineCollection(segments, cmap="coolwarm", linewidth=2.5, zorder=5)
        lc.set_array(alpha_vals)
        lc.set_clim(0.1, 5.0)
        ax.add_collection(lc)
        cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("alpha", fontsize=fontsize_cbar)

    bias_mag, bias_angle = scen["bias_mag"], scen["bias_angle"]
    ax.annotate("", xy=(5 + 4*np.cos(bias_angle), 12 + 4*np.sin(bias_angle)),
                xytext=(5, 12), arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
    ax.text(5, 17, f"bias={bias_mag:.2f}", fontsize=fontsize_bias,
            color="purple", ha="center", fontweight="bold")
    ax.text(145, -18, f"sys_bias={scen['sensor_bias']:.2f}m",
            fontsize=fontsize_radius, color="blue", ha="center",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    for t in range(0, len(r["traj_x"]) - 1, TIME_MARKER_INTERVAL):
        ax.plot(r["traj_x"][t], r["traj_y"][t], 's', color='black', ms=markersize, zorder=6)
        ax.text(r["traj_x"][t], r["traj_y"][t]+1, f"t={t}", fontsize=fontsize_time,
                color='black', ha='center', zorder=7)

    status = ("REACHED" if r["reached_target"] else "FAIL") + \
             (" COLLISION" if r["collided"] else "")
    ax.set_title(f"{scen['name']} -- {status}", fontsize=fontsize_title)
    ax.set_xlim(-5, 165); ax.set_ylim(-22, 22)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.10, f"Steps:{r['steps']} Reward:{r['total_reward']:.0f} "
            f"MinCl:{r['min_clearance']:.2f}m Eff:{r['path_efficiency']:.2f}",
            transform=ax.transAxes, fontsize=fontsize_metrics, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))


def plot_alpha_phi_time(ax, scen, r, fontsize_title=14, fontsize_label=12, fontsize_legend=10):
    ax.plot(r["alphas"], color="blue", linewidth=1.5, label="alpha")
    ax2 = ax.twinx()
    ax2.plot(r["phis"], color="red", linewidth=1.5, label="phi", alpha=0.7)
    ax2.set_ylabel("Phi", color="red", fontsize=fontsize_label)
    ax2.tick_params(axis='y', labelcolor='red')
    for oi in range(N_OBS):
        close_mask = np.array(r["per_obs_dists"][oi]) < 5.0
        for si in range(len(close_mask)):
            if close_mask[si]:
                ax.axvspan(si, si+1, color=OBS_COLORS[oi % len(OBS_COLORS)], alpha=0.1)
    ax.set_title(f"{scen['name']} -- Alpha+Phi + Obs Zone", fontsize=fontsize_title)
    ax.set_xlabel("Time Step", fontsize=fontsize_label)
    ax.set_ylabel("Alpha", color="blue", fontsize=fontsize_label)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(0, 5.5); ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=fontsize_legend)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots_randomized/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = AStarNoisyPlannerEnv()
    model = PPO.load(MODEL_PATH)

    scenarios = generate_scenarios(n=10, seed=42)

    print(f"\nRunning {len(scenarios)} scenarios...")
    all_scenarios = []
    for scen in scenarios:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # --- COMBINED: trajectory + alpha/phi ---
    n_scen = len(scenarios)
    fig, axs = plt.subplots(n_scen, 2, figsize=(28, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Exp 67: A* + Noisy Planner + Mixed Shapes + Alpha/Phi (fully randomized eval)",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        plot_trajectory(axs[i, 0], scen, r)
        plot_alpha_phi_time(axs[i, 1], scen, r)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_traj_alpha_phi.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots_randomized/combined_traj_alpha_phi.png")

    # --- PER-SCENARIO ---
    scenario_dir = os.path.join(save_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        fig_s, axs_s = plt.subplots(1, 2, figsize=(28, 7),
                                     gridspec_kw={"width_ratios": [1.4, 1]})
        plot_trajectory(axs_s[0], scen, r)
        plot_alpha_phi_time(axs_s[1], scen, r)
        fig_s.tight_layout()
        fname = f"scenario_{i+1}_{scen['name'].replace(' ','_')}.png"
        fig_s.savefig(os.path.join(scenario_dir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig_s)
        print(f"Saved: plots_randomized/scenarios/{fname}")

    # --- Console summary ---
    print(f"\n{'='*90}")
    print(f"{'Scenario':<12} {'Reached':>8} {'Collided':>9} {'MinDist':>9} "
          f"{'Steps':>7} {'AvgSpd':>8} {'PathEff':>9} {'BiasMag':>9} {'SysBias':>8}")
    print(f"{'-'*90}")
    for data in all_scenarios:
        scen, r = data["scen"], data["result"]
        print(f"{scen['name']:<12} "
              f"{'Yes' if r['reached_target'] else 'No':>8} "
              f"{'YES' if r['collided'] else 'No':>9} "
              f"{r['min_clearance']:>9.3f} {r['steps']:>7} "
              f"{np.mean(r['speeds']):>8.2f} {r['path_efficiency']:>9.2f} "
              f"{scen['bias_mag']:>9.2f} {scen['sensor_bias']:>8.2f}")
    print(f"\nDone! Plots saved to {save_dir}")
