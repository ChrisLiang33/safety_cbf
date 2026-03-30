"""
Exp 51: SDF-based CBF -- fully randomized eval.
h = ||x-obs|| - r (signed distance), Lgh = unit vector.
Action is [kx, ky, alpha, phi].

Outputs:
  plots_randomized/combined_scenarios.png
  plots_randomized/scenarios/scenario_*.png
  plots_randomized/aggregate_policy_map.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
import os

from env_dynamic import SDFCBFEnv

# --- CONFIG ---
MODEL_PATH = "./models_dynamic/dynamic_10000k_model"
MAX_STEPS = 800

BIAS_MAG_RANGE = (0.3, 1.0)
SYSTEMATIC_BIAS_RANGE = (-0.6, 0.4)
JITTER_RANGE = (-0.2, 0.2)
OBS_RADIUS_RANGE = (3.0, 7.0)

OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
TIME_MARKER_INTERVAL = 50


def generate_scenarios(n=10, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    x_bands = [(20.0, 50.0), (60.0, 100.0), (110.0, 140.0)]
    for i in range(n):
        obs_list = []
        obs_radii = []
        obs_errors = []
        sensor_bias = rng.uniform(*SYSTEMATIC_BIAS_RANGE)
        for x_low, x_high in x_bands:
            x = rng.uniform(x_low, x_high)
            y = rng.uniform(-10.0, 10.0)
            obs_list.append(np.array([x, y]))
            obs_radii.append(rng.uniform(*OBS_RADIUS_RANGE))
            jitter = rng.uniform(*JITTER_RANGE)
            obs_errors.append(sensor_bias + jitter)
        target_y = rng.uniform(-5.0, 5.0)
        target_radius = rng.uniform(1.5, 3.0)
        bias_angle = rng.uniform(0, 2 * np.pi)
        bias_mag = rng.uniform(*BIAS_MAG_RANGE)
        scenarios.append({
            "name": f"Random_{i+1}",
            "obs": obs_list,
            "obs_radii": obs_radii,
            "obs_errors": obs_errors,
            "sensor_bias": sensor_bias,
            "target_pos": np.array([150.0, target_y]),
            "target_radius": target_radius,
            "bias_angle": bias_angle,
            "bias_mag": bias_mag,
        })
    return scenarios


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.current_step = 0
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
        env.true_radius[i] = scen["obs_radii"][i]
        env.estimated_radius[i] = max(scen["obs_radii"][i] + scen["obs_errors"][i], 1.0)
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    env.sensor_bias = scen["sensor_bias"]
    env.bias = scen["bias_mag"] * np.array([np.cos(scen["bias_angle"]),
                                             np.sin(scen["bias_angle"])])
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, speeds = [], [], []
    alphas, phis = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.true_radius[i]
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])

        action, _ = model.predict(obs, deterministic=True)
        alphas.append(float(action[2]))
        phis.append(float(action[3]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speeds.append(np.linalg.norm(np.array([float(action[0]), float(action[1])])))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break
    else:
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "speeds": speeds,
        "alphas": alphas, "phis": phis,
        "dist": dist_list, "per_obs_dists": per_obs_dists,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def plot_trajectory(ax, scen, r, fontsize_title=11, fontsize_label=None, fontsize_legend=None,
                    fontsize_metrics=7, fontsize_cbar=8, fontsize_bias=8, fontsize_radius=7,
                    fontsize_time=7, markersize=4):
    fontsize_label = fontsize_label or fontsize_title

    for j in range(3):
        tr = scen["obs_radii"][j]
        er = scen["obs_errors"][j]
        est_r = max(tr + er, 1.0)
        ax.add_patch(plt.Circle(scen["obs"][j], tr,
                                color="red", alpha=0.2, linestyle="-"))
        ax.add_patch(plt.Circle(scen["obs"][j], est_r,
                                color="blue", alpha=0.1, linestyle="--", linewidth=1.5,
                                fill=False))
        ax.text(scen["obs"][j][0], scen["obs"][j][1] - tr - 1.5,
                f"r={tr:.1f}, err={er:.2f}", fontsize=fontsize_radius, color="red",
                ha="center", fontweight="bold")

    ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                            color="green", alpha=0.3))

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

    bias_mag = scen["bias_mag"]
    bias_angle = scen["bias_angle"]
    arrow_dx = 4.0 * np.cos(bias_angle)
    arrow_dy = 4.0 * np.sin(bias_angle)
    ax.annotate("", xy=(5 + arrow_dx, 12 + arrow_dy), xytext=(5, 12),
                arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
    ax.text(5, 17, f"bias={bias_mag:.2f}", fontsize=fontsize_bias,
            color="purple", ha="center", fontweight="bold")

    ax.text(145, -18, f"sys_bias={scen['sensor_bias']:.2f}m",
            fontsize=fontsize_radius, color="blue", ha="center",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    for t in range(0, len(r["traj_x"]) - 1, TIME_MARKER_INTERVAL):
        ax.plot(r["traj_x"][t], r["traj_y"][t], marker='s', color='black',
                markersize=markersize, zorder=6)
        ax.text(r["traj_x"][t], r["traj_y"][t] + 1.0, f"t={t}", fontsize=fontsize_time,
                color='black', ha='center', zorder=7)

    status = ("REACHED" if r["reached_target"] else "FAIL") + \
             (" COLLISION" if r["collided"] else "")
    ax.set_title(f"{scen['name']} -- {status}", fontsize=fontsize_title)
    ax.set_xlabel("x (m)", fontsize=fontsize_label)
    ax.set_ylabel("y (m)", fontsize=fontsize_label)
    ax.set_xlim(-5, 165)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    metrics_text = (f"Steps: {r['steps']}  Reward: {r['total_reward']:.0f}  "
                    f"MinClearance: {r['min_clearance']:.2f}m  Efficiency: {r['path_efficiency']:.2f}")
    ax.text(0.02, -0.10, metrics_text, transform=ax.transAxes, fontsize=fontsize_metrics,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))


def plot_alpha_phi_time(ax, scen, r, fontsize_title=11, fontsize_label=9, fontsize_legend=7):
    ax.plot(r["alphas"], color="blue", linewidth=1.5, label="alpha")
    ax2 = ax.twinx()
    ax2.plot(r["phis"], color="red", linewidth=1.5, label="phi", alpha=0.7)
    ax2.set_ylabel("Phi", color="red", fontsize=fontsize_label)
    ax2.tick_params(axis='y', labelcolor='red')
    for oi in range(3):
        close_mask = np.array(r["per_obs_dists"][oi]) < 5.0
        for start_idx in range(len(close_mask)):
            if close_mask[start_idx]:
                ax.axvspan(start_idx, start_idx + 1, color=OBS_COLORS[oi], alpha=0.1)
    ax.set_title(f"{scen['name']} -- Alpha+Phi + Obs Zone", fontsize=fontsize_title)
    ax.set_xlabel("Time Step", fontsize=fontsize_label)
    ax.set_ylabel("Alpha", color="blue", fontsize=fontsize_label)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=fontsize_legend)


def plot_speed(ax, scen, r, fontsize_title=11, fontsize_label=None):
    fontsize_label = fontsize_label or fontsize_title
    ax.plot(r["speeds"], color="black", linewidth=2)
    ax.axhline(6.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, label="Max")
    ax.set_title(f"{scen['name']} -- Speed", fontsize=fontsize_title)
    ax.set_xlabel("Time Step", fontsize=fontsize_label)
    ax.set_ylabel("Speed (m/s)", fontsize=fontsize_label)
    ax.set_ylim(-0.1, 8.0)
    ax.grid(True, alpha=0.3)


def get_min_dists_at_step(r):
    return [min(r["per_obs_dists"][oi][t] for oi in range(3))
            for t in range(len(r["alphas"]))]


def plot_alpha_vs_dist(ax, scen, r, min_dists, fontsize_title=11, fontsize_label=None,
                       fontsize_legend=7, show_legend=True):
    fontsize_label = fontsize_label or fontsize_title
    ax.scatter(min_dists, r["alphas"], c=range(len(r["alphas"])),
               cmap="viridis", s=8, alpha=0.6)
    ax.axvline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7, label="Collision")
    ax.set_title(f"{scen['name']} -- Alpha vs Obs Dist", fontsize=fontsize_title)
    ax.set_xlabel("Min Distance to Obstacle Surface (m)", fontsize=fontsize_label)
    ax.set_ylabel("Alpha", fontsize=fontsize_label)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right", fontsize=fontsize_legend)


def plot_phi_vs_dist(ax, scen, r, min_dists, fontsize_title=11, fontsize_label=None,
                     fontsize_legend=7, show_legend=True):
    fontsize_label = fontsize_label or fontsize_title
    ax.scatter(min_dists, r["phis"], c=range(len(r["phis"])),
               cmap="magma", s=8, alpha=0.6)
    ax.axvline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7, label="Collision")
    ax.set_title(f"{scen['name']} -- Phi vs Obs Dist", fontsize=fontsize_title)
    ax.set_xlabel("Min Distance to Obstacle Surface (m)", fontsize=fontsize_label)
    ax.set_ylabel("Phi", fontsize=fontsize_label)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="upper right", fontsize=fontsize_legend)


def plot_policy_map(ax, r, scen, min_dists, fontsize_title=11, fontsize_label=None,
                    fontsize_cbar=8):
    fontsize_label = fontsize_label or fontsize_title
    sc = ax.scatter(r["alphas"], r["phis"], c=min_dists,
                    cmap="RdYlGn", s=10, alpha=0.6)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Min Obs Dist (m)", fontsize=fontsize_cbar)
    ax.set_title(f"{scen['name']} -- Policy Map", fontsize=fontsize_title)
    ax.set_xlabel("Alpha", fontsize=fontsize_label)
    ax.set_ylabel("Phi", fontsize=fontsize_label)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots_randomized/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = SDFCBFEnv()
    model = PPO.load(MODEL_PATH)

    scenarios = generate_scenarios(n=10, seed=42)

    print(f"\nRunning {len(scenarios)} fully randomized scenarios...")
    all_scenarios = []
    for scen in scenarios:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # --- COMBINED PLOT ---
    n_scen = len(scenarios)
    fig, axs = plt.subplots(n_scen, 6, figsize=(58, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1, 1]})
    fig.suptitle("Exp 51: SDF-BASED CBF (fully randomized eval)", fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        min_dists = get_min_dists_at_step(r)
        plot_trajectory(axs[i, 0], scen, r)
        plot_alpha_phi_time(axs[i, 1], scen, r)
        plot_speed(axs[i, 2], scen, r)
        plot_alpha_vs_dist(axs[i, 3], scen, r, min_dists, show_legend=(i == 0))
        plot_phi_vs_dist(axs[i, 4], scen, r, min_dists, show_legend=(i == 0))
        plot_policy_map(axs[i, 5], r, scen, min_dists)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_scenarios.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots_randomized/combined_scenarios.png")

    # --- PER-SCENARIO PLOTS ---
    scenario_dir = os.path.join(save_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        name = scen["name"].replace(" ", "_")
        min_dists = get_min_dists_at_step(r)

        fig_s, axs_s = plt.subplots(1, 6, figsize=(48, 7),
                                     gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1, 1]})
        plot_trajectory(axs_s[0], scen, r, fontsize_title=14, fontsize_label=12,
                        fontsize_metrics=9, fontsize_cbar=10, fontsize_bias=10,
                        fontsize_radius=9, fontsize_time=9, markersize=5)
        plot_alpha_phi_time(axs_s[1], scen, r, fontsize_title=14, fontsize_label=12,
                            fontsize_legend=10)
        plot_speed(axs_s[2], scen, r, fontsize_title=14, fontsize_label=12)
        plot_alpha_vs_dist(axs_s[3], scen, r, min_dists, fontsize_title=14,
                           fontsize_label=12, fontsize_legend=10)
        plot_phi_vs_dist(axs_s[4], scen, r, min_dists, fontsize_title=14,
                         fontsize_label=12, fontsize_legend=10)
        plot_policy_map(axs_s[5], r, scen, min_dists, fontsize_title=14,
                        fontsize_label=12, fontsize_cbar=10)

        fig_s.tight_layout()
        fname = f"scenario_{i+1}_{name}.png"
        fig_s.savefig(os.path.join(scenario_dir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig_s)
        print(f"Saved: plots_randomized/scenarios/{fname}")

    # --- AGGREGATE POLICY MAP ---
    fig_agg, (ax_alpha, ax_phi) = plt.subplots(1, 2, figsize=(18, 8))
    fig_agg.suptitle("Exp 51 (SDF-Based CBF): Alpha & Phi vs Distance (fully randomized eval)",
                     fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_scenarios)))
    for idx, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        label = scen["name"]
        c = colors[idx]
        dists = r["dist"]
        n = min(len(dists), len(r["alphas"]))
        ax_alpha.scatter(dists[:n], r["alphas"][:n], color=c, s=8, alpha=0.5, label=label)
        ax_phi.scatter(dists[:n], r["phis"][:n], color=c, s=8, alpha=0.5, label=label)

    for ax in (ax_alpha, ax_phi):
        ax.axvline(0, color="red", linewidth=1.5, label="True surface (collision)")
        ax.set_xlabel("Min Distance to Obstacle Surface (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax_alpha.set_ylabel("Alpha")
    ax_alpha.set_title("Alpha vs Distance")
    ax_phi.set_ylabel("Phi")
    ax_phi.set_title("Phi vs Distance")

    fig_agg.tight_layout()
    fig_agg.savefig(os.path.join(save_dir, "aggregate_policy_map.png"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig_agg)
    print("Saved: plots_randomized/aggregate_policy_map.png")

    # --- Console summary ---
    print(f"\n{'='*100}")
    print(f"FULLY RANDOMIZED SCENARIOS (max speed 6 m/s)")
    print(f"{'='*100}")
    print(f"{'Scenario':<12} {'Reached':>8} {'Collided':>9} {'MinDist':>9} "
          f"{'Steps':>7} {'AvgSpd':>8} {'MaxSpd':>8} {'PathEff':>9} {'BiasMag':>9} {'SysBias':>8} {'Errors':>20}")
    print(f"{'-'*120}")

    for data in all_scenarios:
        scen, r = data["scen"], data["result"]
        errs = ", ".join(f"{e:.2f}" for e in scen["obs_errors"])
        print(f"{scen['name']:<12} "
              f"{'Yes' if r['reached_target'] else 'No':>8} "
              f"{'YES' if r['collided'] else 'No':>9} "
              f"{r['min_clearance']:>9.3f} {r['steps']:>7} "
              f"{np.mean(r['speeds']):>8.2f} {max(r['speeds']):>8.2f} "
              f"{r['path_efficiency']:>9.2f} "
              f"{scen['bias_mag']:>9.2f} {scen['sensor_bias']:>8.2f} [{errs}]")

    print(f"\nDone! Plots saved to {save_dir}")
