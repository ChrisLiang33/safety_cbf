"""
Exp 71: Discrete Two-Policy Moving Obs -- fully randomized eval.
Action is Discrete(9): 3 alpha x 3 phi (down/stay/up). Step=0.2.
Frozen nav + discrete safety filter. 3 moving circular obstacles.

3-column layout:
  Col 1 (wide): trajectory (moving obs, initial solid / final dashed / arrows)
  Col 2: alpha + phi time series (twin y-axis)
  Col 3: discrete actions (two stacked: alpha actions, phi actions)

Outputs:
  plots_randomized/combined_traj_alpha_phi.png
  plots_randomized/scenarios/scenario_*.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
import os

from env_safety_discrete import SafetyDiscreteMovingObsEnv

MODEL_PATH = "./models_dynamic/safety_discrete_model"
MAX_STEPS = 800
N_OBS = 3

BIAS_MAG_RANGE = (0.3, 1.0)
SYSTEMATIC_BIAS_RANGE = (-0.6, 0.4)
JITTER_RANGE = (-0.2, 0.2)
OBS_RADIUS_RANGE = (3.0, 7.0)
OBS_SPEED_RANGE = (0.3, 1.0)

OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
TIME_MARKER_INTERVAL = 50


def generate_scenarios(n=10, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    x_bands = [(20.0, 50.0), (60.0, 100.0), (110.0, 140.0)]
    for i in range(n):
        obs_list, obs_radii, obs_errors = [], [], []
        obs_vels = []
        sensor_bias = rng.uniform(*SYSTEMATIC_BIAS_RANGE)
        for x_low, x_high in x_bands:
            x = rng.uniform(x_low, x_high)
            y = rng.uniform(-10.0, 10.0)
            obs_list.append(np.array([x, y]))
            obs_radii.append(rng.uniform(*OBS_RADIUS_RANGE))
            obs_errors.append(sensor_bias + rng.uniform(*JITTER_RANGE))
            speed = rng.uniform(*OBS_SPEED_RANGE)
            angle = rng.uniform(0, 2 * np.pi)
            obs_vels.append(speed * np.array([np.cos(angle), np.sin(angle)]))
        scenarios.append({
            "name": f"Random_{i+1}",
            "obs": obs_list, "obs_radii": obs_radii, "obs_errors": obs_errors,
            "obs_vels": obs_vels,
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
    env.alpha = 2.5
    env.phi = 1.0
    for i in range(N_OBS):
        env.obs_pos[i] = scen["obs"][i].copy()
        env.true_radius[i] = scen["obs_radii"][i]
        env.estimated_radius[i] = max(scen["obs_radii"][i] + scen["obs_errors"][i], 1.0)
        env.obs_vel[i] = scen["obs_vels"][i].copy()
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    env.sensor_bias = scen["sensor_bias"]
    env.prev_e_i = np.array([1.0, 0.0])
    env.bias = scen["bias_mag"] * np.array([np.cos(scen["bias_angle"]),
                                             np.sin(scen["bias_angle"])])
    return env._get_obs()


def path_length(traj_x, traj_y):
    return np.sum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2))


def run_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, speeds = [], [], []
    alphas, phis = [], []
    alpha_actions, phi_actions = [], []
    dist_list = []
    per_obs_dists = [[] for _ in range(N_OBS)]
    obs_trails = [[scen["obs"][i].copy()] for i in range(N_OBS)]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.true_radius[i]
                 for i in range(N_OBS)]
        dist_list.append(min(dists))
        for oi in range(N_OBS):
            per_obs_dists[oi].append(dists[oi])

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speeds.append(np.linalg.norm(info.get("safe_u", np.zeros(2))))
        alphas.append(info["alpha"])
        phis.append(info["phi"])
        alpha_actions.append(info["alpha_action"])
        phi_actions.append(info["phi_action"])

        for i in range(N_OBS):
            obs_trails[i].append(env.obs_pos[i].copy())

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
        "alpha_actions": alpha_actions, "phi_actions": phi_actions,
        "dist": dist_list, "per_obs_dists": per_obs_dists,
        "obs_trails": obs_trails,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": plen / straight if straight > 0 else 1.0,
    }


def plot_trajectory(ax, scen, r, fontsize_title=14, fontsize_label=12,
                    fontsize_metrics=9, fontsize_cbar=10, fontsize_bias=10,
                    fontsize_radius=9, fontsize_time=9, markersize=5):
    # Draw initial obstacle positions (solid) and final positions (dashed)
    for j in range(N_OBS):
        tr = scen["obs_radii"][j]
        er = scen["obs_errors"][j]
        est_r = max(tr + er, 1.0)
        init_pos = scen["obs"][j]
        final_pos = r["obs_trails"][j][-1]

        # Initial position -- solid
        ax.add_patch(plt.Circle(init_pos, tr, color="red", alpha=0.15))
        ax.add_patch(plt.Circle(init_pos, est_r, color="blue", alpha=0.08,
                                linestyle="--", linewidth=1.0, fill=False))

        # Final position -- dashed outline
        ax.add_patch(plt.Circle(final_pos, tr, color="red", alpha=0.3,
                                linestyle="--", linewidth=2.0, fill=False))

        # Arrow from initial to final
        dx = final_pos[0] - init_pos[0]
        dy = final_pos[1] - init_pos[1]
        if np.sqrt(dx**2 + dy**2) > 0.5:
            ax.annotate("", xy=final_pos, xytext=init_pos,
                        arrowprops=dict(arrowstyle="->", color=OBS_COLORS[j],
                                        lw=1.5, alpha=0.6))

        vel = scen["obs_vels"][j]
        speed = np.linalg.norm(vel)
        ax.text(init_pos[0], init_pos[1] - tr - 1.5,
                f"r={tr:.1f}, err={er:.2f}\nv={speed:.1f}m/s",
                fontsize=fontsize_radius, color="red",
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
    ax.set_title(f"{scen['name']} -- Alpha+Phi", fontsize=fontsize_title)
    ax.set_xlabel("Time Step", fontsize=fontsize_label)
    ax.set_ylabel("Alpha", color="blue", fontsize=fontsize_label)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(0, 5.5); ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="upper right", fontsize=fontsize_legend)


def plot_discrete_actions(ax_alpha, ax_phi, scen, r, fontsize_title=12, fontsize_label=11):
    """Plot discrete action choices: alpha actions (top) and phi actions (bottom)."""
    steps = np.arange(len(r["alpha_actions"]))
    aa = np.array(r["alpha_actions"])
    pa = np.array(r["phi_actions"])

    # Alpha actions: 0=down (blue), 1=stay (gray), 2=up (red)
    action_colors_alpha = {0: "#3498db", 1: "#95a5a6", 2: "#e74c3c"}
    action_labels_alpha = {0: "down", 1: "stay", 2: "up"}
    for a_val in [0, 1, 2]:
        mask = aa == a_val
        if np.any(mask):
            ax_alpha.bar(steps[mask], np.ones(np.sum(mask)), bottom=0, width=1.0,
                         color=action_colors_alpha[a_val], alpha=0.7,
                         label=f"alpha {action_labels_alpha[a_val]}")
    ax_alpha.set_ylabel("Alpha Act", fontsize=fontsize_label)
    ax_alpha.set_yticks([0.5])
    ax_alpha.set_yticklabels([""])
    ax_alpha.set_title(f"{scen['name']} -- Discrete Actions", fontsize=fontsize_title)
    ax_alpha.legend(loc="upper right", fontsize=8, ncol=3)
    ax_alpha.set_xlim(0, len(aa))
    ax_alpha.grid(True, alpha=0.2, axis='x')

    # Phi actions: 0=down (blue), 1=stay (gray), 2=up (red)
    action_colors_phi = {0: "#2ecc71", 1: "#95a5a6", 2: "#f39c12"}
    action_labels_phi = {0: "down", 1: "stay", 2: "up"}
    for a_val in [0, 1, 2]:
        mask = pa == a_val
        if np.any(mask):
            ax_phi.bar(steps[mask], np.ones(np.sum(mask)), bottom=0, width=1.0,
                       color=action_colors_phi[a_val], alpha=0.7,
                       label=f"phi {action_labels_phi[a_val]}")
    ax_phi.set_ylabel("Phi Act", fontsize=fontsize_label)
    ax_phi.set_xlabel("Time Step", fontsize=fontsize_label)
    ax_phi.set_yticks([0.5])
    ax_phi.set_yticklabels([""])
    ax_phi.legend(loc="upper right", fontsize=8, ncol=3)
    ax_phi.set_xlim(0, len(pa))
    ax_phi.grid(True, alpha=0.2, axis='x')


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots_randomized/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = SafetyDiscreteMovingObsEnv(
        nav_model_path="./models_dynamic/nav_moving_discrete_model_v2"
    )
    model = PPO.load(MODEL_PATH)

    scenarios = generate_scenarios(n=10, seed=42)

    print(f"\nRunning {len(scenarios)} scenarios...")
    all_scenarios = []
    for scen in scenarios:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # --- COMBINED: 3-column layout ---
    n_scen = len(scenarios)
    fig = plt.figure(figsize=(38, 9 * n_scen))
    fig.suptitle("Exp 71: Discrete Two-Policy Moving Obs (step=0.2)",
                 fontsize=20, y=1.002)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]

        # Column 1: trajectory (wide)
        ax_traj = fig.add_subplot(n_scen, 4, i * 4 + 1)
        plot_trajectory(ax_traj, scen, r)

        # Column 2: alpha + phi time series
        ax_ap = fig.add_subplot(n_scen, 4, i * 4 + 2)
        plot_alpha_phi_time(ax_ap, scen, r)

        # Column 3: discrete actions (two stacked subplots using gridspec)
        ax_da_alpha = fig.add_subplot(n_scen, 4, i * 4 + 3)
        ax_da_phi = fig.add_subplot(n_scen, 4, i * 4 + 4)
        plot_discrete_actions(ax_da_alpha, ax_da_phi, scen, r)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_traj_alpha_phi.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots_randomized/combined_traj_alpha_phi.png")

    # --- PER-SCENARIO ---
    scenario_dir = os.path.join(save_dir, "scenarios")
    os.makedirs(scenario_dir, exist_ok=True)
    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        fig_s = plt.figure(figsize=(38, 9))

        ax_traj = fig_s.add_subplot(1, 4, 1)
        plot_trajectory(ax_traj, scen, r)

        ax_ap = fig_s.add_subplot(1, 4, 2)
        plot_alpha_phi_time(ax_ap, scen, r)

        ax_da_alpha = fig_s.add_subplot(1, 4, 3)
        ax_da_phi = fig_s.add_subplot(1, 4, 4)
        plot_discrete_actions(ax_da_alpha, ax_da_phi, scen, r)

        fig_s.tight_layout()
        fname = f"scenario_{i+1}_{scen['name'].replace(' ','_')}.png"
        fig_s.savefig(os.path.join(scenario_dir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig_s)
        print(f"Saved: plots_randomized/scenarios/{fname}")

    # --- Console summary ---
    print(f"\n{'='*100}")
    print(f"{'Scenario':<12} {'Reached':>8} {'Collided':>9} {'MinDist':>9} "
          f"{'Steps':>7} {'AvgSpd':>8} {'PathEff':>9} {'BiasMag':>9} {'SysBias':>8} "
          f"{'AvgAlpha':>9} {'AvgPhi':>8}")
    print(f"{'-'*100}")
    for data in all_scenarios:
        scen, r = data["scen"], data["result"]
        avg_alpha = np.mean(r["alphas"]) if r["alphas"] else 0
        avg_phi = np.mean(r["phis"]) if r["phis"] else 0
        print(f"{scen['name']:<12} "
              f"{'Yes' if r['reached_target'] else 'No':>8} "
              f"{'YES' if r['collided'] else 'No':>9} "
              f"{r['min_clearance']:>9.3f} {r['steps']:>7} "
              f"{np.mean(r['speeds']):>8.2f} {r['path_efficiency']:>9.2f} "
              f"{scen['bias_mag']:>9.2f} {scen['sensor_bias']:>8.2f} "
              f"{avg_alpha:>9.2f} {avg_phi:>8.2f}")
    print(f"\nDone! Plots saved to {save_dir}")
