"""
Unicycle weave ablation: dynamic [v,omega,alpha] vs fixed alpha + proportional go-to-goal.

Outputs:
  plots/combined_scenarios.png  (traj w/ heading | v+omega | alpha+dist | speed+alpha | policy map)
  plots/aggregate_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env_dynamic import UnicycleWeaveDynamicEnv
from env_fixed_alpha import FixedAlphaUnicycleEnv

# --- CONFIG ---
DYNAMIC_MODEL_PATH = "./models_dynamic/dynamic_2000k_model"
FIXED_ALPHAS = [0.1, 0.5, 1.0, 5.0]
_cmap = plt.cm.plasma
FIXED_ALPHA_COLORS = {a: _cmap((a - 0.1) / (5.0 - 0.1)) for a in FIXED_ALPHAS}
DYNAMIC_COLOR = "black"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
OBS_RADIUS = 5.0

SCENARIOS = [
    {"name": "Standard Slalom",
     "obs": [np.array([30.0, 6.0]), np.array([50.0, -6.0]), np.array([70.0, 6.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Tight Slalom",
     "obs": [np.array([30.0, 5.0]), np.array([50.0, -5.0]), np.array([70.0, 5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Wide Slalom",
     "obs": [np.array([30.0, 8.0]), np.array([50.0, -8.0]), np.array([70.0, 8.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Clustered Obstacles",
     "obs": [np.array([25.0, 6.0]), np.array([35.0, -6.0]), np.array([45.0, 6.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Spread Out",
     "obs": [np.array([20.0, 7.0]), np.array([50.0, -7.0]), np.array([80.0, 7.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Off-Center Target",
     "obs": [np.array([30.0, 6.0]), np.array([50.0, -6.0]), np.array([70.0, 6.0])],
     "target_pos": np.array([100.0, 3.0]), "target_radius": 2.0},
    {"name": "All Same Side",
     "obs": [np.array([25.0, 6.0]), np.array([50.0, 7.0]), np.array([75.0, 5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Diagonal Scatter",
     "obs": [np.array([25.0, 8.0]), np.array([55.0, -5.0]), np.array([75.0, 6.0])],
     "target_pos": np.array([100.0, -2.0]), "target_radius": 2.0},
]


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.theta = 0.0
    env.v = 0.0
    env.omega = 0.0
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
        env.obs_radius[i] = OBS_RADIUS
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_dynamic_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, thetas, alphas = [], [], [], []
    v_speeds, omega_vals = [], []
    v_des_list, omega_des_list = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    h_vals_list = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        thetas.append(env.theta)
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.obs_radius[i]
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])
            x_diff = env.robot_pos - env.obs_pos[oi]
            h = np.sum(x_diff**2) - env.obs_radius[oi]**2
            h_vals_list[oi].append(h)

        action, _ = model.predict(obs, deterministic=True)
        v_des_list.append(float(action[0]))
        omega_des_list.append(float(action[1]))
        alpha_val = float(action[2])
        alphas.append(alpha_val)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        v_speeds.append(info.get("safe_v", 0.0))
        omega_vals.append(info.get("safe_omega", 0.0))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            thetas.append(env.theta)
            break
    else:
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        thetas.append(env.theta)

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "thetas": thetas,
        "alphas": alphas, "v_speeds": v_speeds, "omega_vals": omega_vals,
        "v_des": v_des_list, "omega_des": omega_des_list,
        "dist": dist_list, "per_obs_dists": per_obs_dists,
        "h_vals": h_vals_list,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def run_fixed_episode(env, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, thetas = [], [], []
    v_speeds, omega_vals = [], []
    v_des_list, omega_des_list = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        thetas.append(env.theta)
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.obs_radius[i]
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])

        k_nom_before = env._compute_k_nom()
        v_des_list.append(float(k_nom_before[0]))
        omega_des_list.append(float(k_nom_before[1]))
        dummy_action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(dummy_action)
        total_reward += reward
        v_speeds.append(info.get("safe_v", 0.0))
        omega_vals.append(info.get("safe_omega", 0.0))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            thetas.append(env.theta)
            break
    else:
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        thetas.append(env.theta)

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "thetas": thetas,
        "v_speeds": v_speeds, "omega_vals": omega_vals,
        "v_des": v_des_list, "omega_des": omega_des_list,
        "dist": dist_list, "per_obs_dists": per_obs_dists,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def generate_random_scenarios(n, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    for i in range(n):
        target_y = rng.uniform(-3.0, 3.0)
        target_radius = rng.uniform(1.5, 3.0)
        x1 = rng.uniform(20.0, 35.0)
        y1 = rng.uniform(5.0, 8.0)
        x2 = rng.uniform(45.0, 60.0)
        y2 = rng.uniform(-8.0, -5.0)
        x3 = rng.uniform(65.0, 80.0)
        y3 = rng.uniform(5.0, 8.0)
        scenarios.append({
            "name": f"Random_{i}",
            "obs": [np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])],
            "target_pos": np.array([100.0, target_y]),
            "target_radius": target_radius,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading unicycle weave model...")
    dyn_env = UnicycleWeaveDynamicEnv()
    dyn_model = PPO.load(DYNAMIC_MODEL_PATH)

    fixed_envs = {}
    for alpha in FIXED_ALPHAS:
        fixed_envs[alpha] = FixedAlphaUnicycleEnv(alpha=alpha)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        dyn = run_dynamic_episode(dyn_env, dyn_model, scen)
        fixed_results = {}
        for alpha in FIXED_ALPHAS:
            fixed_results[alpha] = run_fixed_episode(fixed_envs[alpha], scen)
        all_scenarios.append({"scen": scen, "dyn": dyn, "fixed": fixed_results})

    # =====================================================================
    # COMBINED PLOT: 5 columns
    # Traj+heading | v+omega | alpha+dist | speed+alpha | policy map
    # =====================================================================
    TIME_MARKER_INTERVAL = 50
    ARROW_INTERVAL = 15  # draw heading arrow every N steps
    OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
    OBS_LABELS = ["Obs 1", "Obs 2", "Obs 3"]
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 5, figsize=(48, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1, 1, 1, 1]})
    fig.suptitle(r"Unicycle Weave: Dynamic [v,$\omega$,$\alpha$] vs Fixed $\alpha$ + Prop. Control",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]

        # --- Column 1: Trajectory with heading arrows ---
        ax = axs[i, 0]
        for j in range(3):
            ax.add_patch(plt.Circle(scen["obs"][j], OBS_RADIUS, color="red", alpha=0.2))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                                color="green", alpha=0.3))

        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["traj_x"], r["traj_y"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.5, linestyle="--", alpha=0.6, label=rf"$\alpha$={fa}")

        dyn_x, dyn_y = dyn["traj_x"], dyn["traj_y"]
        dyn_th = dyn["thetas"]
        step_skip = 2
        sc = ax.scatter(dyn_x[:-1:step_skip], dyn_y[:-1:step_skip],
                        c=dyn["alphas"][::step_skip], cmap="coolwarm",
                        vmin=0.1, vmax=5.0, s=14, zorder=5, label=r"Dynamic $\alpha$")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$")

        # Heading arrows
        arrow_len = 2.0
        for t in range(0, len(dyn_x) - 1, ARROW_INTERVAL):
            dx = arrow_len * np.cos(dyn_th[t])
            dy = arrow_len * np.sin(dyn_th[t])
            ax.annotate("", xy=(dyn_x[t] + dx, dyn_y[t] + dy),
                        xytext=(dyn_x[t], dyn_y[t]),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.0, alpha=0.6))

        # Timestep markers
        for t in range(0, len(dyn_x) - 1, TIME_MARKER_INTERVAL):
            ax.plot(dyn_x[t], dyn_y[t], marker='s', color='black', markersize=4, zorder=6)
            ax.text(dyn_x[t], dyn_y[t] + 0.8, f"t={t}", fontsize=7, color='black',
                    ha='center', zorder=7)

        ax.set_title(f"{scen['name']}", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-5, 110)
        ax.set_ylim(-16, 16)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

        metrics_lines = []
        metrics_lines.append(f"{'Method':<10} {'Steps':>5} {'Reward':>7}")
        metrics_lines.append("-" * 25)
        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            metrics_lines.append(f"a={fa:<5.1f}  {r['steps']:>5} {r['total_reward']:>7.0f}")
        metrics_lines.append(f"{'Dyn':<8}  {dyn['steps']:>5} {dyn['total_reward']:>7.0f}")
        metrics_text = "\n".join(metrics_lines)
        ax.text(0.02, -0.15, metrics_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

        # --- Column 2: Forward speed (v) + Turn rate (omega) ---
        ax = axs[i, 1]
        ax.set_ylabel("Forward Speed v (m/s)", color="teal")
        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["v_speeds"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.2, linestyle="--", alpha=0.6, label=rf"v $\alpha$={fa}")
        ax.plot(dyn["v_speeds"], color="teal", linewidth=2, label=r"v (dynamic)")
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)
        ax.tick_params(axis="y", labelcolor="teal")
        ax.set_ylim(-0.1, 4.0)

        ax_om = ax.twinx()
        ax_om.set_ylabel(r"$\omega$ (rad/s)", color="brown")
        ax_om.plot(dyn["omega_vals"], color="brown", linewidth=1.5, linestyle="-.",
                   alpha=0.7, label=r"$\omega$ (dynamic)")
        ax_om.tick_params(axis="y", labelcolor="brown")
        ax_om.set_ylim(-2.0, 2.0)

        ax.set_title(f"{scen['name']} — Speed + Turn Rate", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_om.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 3: Alpha + Per-Obstacle Distance ---
        ax = axs[i, 2]
        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.plot(dyn["alphas"], color="purple", linewidth=2.5, label=r"$\alpha$", zorder=5)
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_dist = ax.twinx()
        ax_dist.set_ylabel("Dist to Obs (m)", color="gray")
        ax_dist.tick_params(axis="y", labelcolor="gray")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for oi in range(3):
            d = dyn["per_obs_dists"][oi]
            ax_dist.plot(d, color=OBS_COLORS[oi], linewidth=1.2, linestyle="-.",
                         alpha=0.7, label=OBS_LABELS[oi])
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            ax.text(t_min, 5.3, f"{OBS_LABELS[oi]}\nt={t_min}", fontsize=6,
                    color=OBS_COLORS[oi], ha="center", va="top", fontweight="bold")
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.04)

        for fa in FIXED_ALPHAS:
            ax.axhline(fa, color=FIXED_ALPHA_COLORS[fa], linewidth=1, linestyle=":", alpha=0.4)

        ax.set_title(f"{scen['name']} — Alpha vs Obstacle Proximity", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 4: v_des vs safe_v + Alpha overlay ---
        ax = axs[i, 3]
        ax.set_ylabel("Speed (m/s)", color="teal")
        ax.plot(dyn["v_speeds"], color="teal", linewidth=2, label=r"$v_{safe}$ (actual)")
        ax.plot(dyn["v_des"], color="teal", linewidth=1, linestyle=":",
                alpha=0.5, label=r"$v_{des}$ (requested)")
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.tick_params(axis="y", labelcolor="teal")
        ax.set_ylim(-0.1, 4.0)

        ax_alpha = ax.twinx()
        ax_alpha.set_ylabel(r"$\alpha$ Value", color="purple")
        ax_alpha.plot(dyn["alphas"], color="purple", linewidth=1.5, linestyle="--",
                      alpha=0.7, label=r"$\alpha$")
        ax_alpha.tick_params(axis="y", labelcolor="purple")
        ax_alpha.set_ylim(0, 5.5)

        for t_idx in range(len(dyn["dist"])):
            if dyn["dist"][t_idx] < 8.0:
                ax.axvspan(t_idx, t_idx + 1, color="red", alpha=0.05)

        ax.set_title(f"{scen['name']} — CBF Intervention on v", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_alpha.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 5: Policy Map ---
        ax = axs[i, 4]
        min_dists_per_t = np.array(dyn["dist"])
        alphas_arr = np.array(dyn["alphas"])
        sc5 = ax.scatter(min_dists_per_t, alphas_arr,
                         c=np.arange(len(alphas_arr)), cmap="viridis",
                         s=10, alpha=0.7, zorder=3)
        if i == 0:
            cbar5 = plt.colorbar(sc5, ax=ax, fraction=0.046, pad=0.04)
            cbar5.set_label("Time Step")
        ax.set_xlabel("Min Dist to Obs Surface (m)")
        ax.set_ylabel(r"$\alpha$")
        ax.set_ylim(0, 5.5)
        ax.set_xlim(-1, max(min_dists_per_t) * 1.05)
        ax.axvline(0, color="red", linewidth=1, linestyle=":", alpha=0.5, label="Collision boundary")
        ax.set_title(f"{scen['name']} — Learned Policy Map", fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_scenarios.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/combined_scenarios.png")

    # =====================================================================
    # BATCH: 100 random scenarios
    # =====================================================================
    print(f"\nRunning {N_RANDOM_SCENARIOS} random scenarios...")
    random_scenarios = generate_random_scenarios(N_RANDOM_SCENARIOS)

    methods = [rf"$\alpha$={a}" for a in FIXED_ALPHAS] + [r"Dynamic $\alpha$"]
    method_colors = [FIXED_ALPHA_COLORS[a] for a in FIXED_ALPHAS] + [DYNAMIC_COLOR]
    agg = {m: {"success": 0, "collisions": 0, "min_clearances": [], "efficiencies": [],
               "steps": [], "avg_speeds": []}
           for m in methods}

    for idx, scen in enumerate(random_scenarios):
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{N_RANDOM_SCENARIOS}...")

        dyn = run_dynamic_episode(dyn_env, dyn_model, scen)
        m_dyn = methods[-1]
        agg[m_dyn]["success"] += int(dyn["reached_target"])
        agg[m_dyn]["collisions"] += int(dyn["collided"])
        agg[m_dyn]["min_clearances"].append(dyn["min_clearance"])
        agg[m_dyn]["efficiencies"].append(dyn["path_efficiency"])
        agg[m_dyn]["steps"].append(dyn["steps"])
        agg[m_dyn]["avg_speeds"].append(np.mean(dyn["v_speeds"]))

        for j, fa in enumerate(FIXED_ALPHAS):
            r = run_fixed_episode(fixed_envs[fa], scen)
            m = methods[j]
            agg[m]["success"] += int(r["reached_target"])
            agg[m]["collisions"] += int(r["collided"])
            agg[m]["min_clearances"].append(r["min_clearance"])
            agg[m]["efficiencies"].append(r["path_efficiency"])
            agg[m]["steps"].append(r["steps"])
            agg[m]["avg_speeds"].append(np.mean(r["v_speeds"]))

    # =====================================================================
    # AGGREGATE METRICS PLOT
    # =====================================================================
    n_methods = len(methods)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"Aggregate — Unicycle Weave ({N_RANDOM_SCENARIOS} Random Scenarios)",
                 fontsize=16)
    x = np.arange(n_methods)

    ax = axs[0, 0]
    vals = [agg[m]["success"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    ax = axs[0, 1]
    vals = [agg[m]["collisions"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, max(max(vals) * 1.3, 10))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    ax = axs[0, 2]
    vals = [np.mean(agg[m]["min_clearances"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    ax = axs[1, 0]
    vals = [np.mean(agg[m]["efficiencies"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)

    ax = axs[1, 1]
    vals = [np.mean(agg[m]["steps"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length (lower = faster)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=9)

    ax = axs[1, 2]
    vals = [np.mean(agg[m]["avg_speeds"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Forward Speed")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(3.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "aggregate_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/aggregate_metrics.png")

    # =====================================================================
    # Console summary
    # =====================================================================
    print(f"\n{'='*115}")
    print("HAND-PICKED SCENARIO RESULTS")
    print(f"{'='*115}")
    print(f"{'Scenario':<25} {'Method':<18} {'Reached':>8} {'Collided':>9} "
          f"{'MinDist':>9} {'Steps':>7} {'AvgSpd':>8} {'PathEff':>9}")
    print(f"{'-'*115}")

    for data in all_scenarios:
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        for alpha in FIXED_ALPHAS:
            r = fixed_results[alpha]
            print(f"{scen['name']:<25} {'Fixed a=' + str(alpha):<18} "
                  f"{'Yes' if r['reached_target'] else 'No':>8} "
                  f"{'YES' if r['collided'] else 'No':>9} "
                  f"{r['min_clearance']:>9.3f} {r['steps']:>7} "
                  f"{np.mean(r['v_speeds']):>8.2f} {r['path_efficiency']:>9.2f}")
        print(f"{scen['name']:<25} {'Dynamic':<18} "
              f"{'Yes' if dyn['reached_target'] else 'No':>8} "
              f"{'YES' if dyn['collided'] else 'No':>9} "
              f"{dyn['min_clearance']:>9.3f} {dyn['steps']:>7} "
              f"{np.mean(dyn['v_speeds']):>8.2f} {dyn['path_efficiency']:>9.2f}")
        print()

    print(f"{'='*115}")
    print(f"AGGREGATE ({N_RANDOM_SCENARIOS} RANDOM SCENARIOS)")
    print(f"{'='*115}")
    print(f"{'Method':<18} {'Success':>10} {'Collisions':>11} {'AvgMinDist':>11} "
          f"{'AvgSteps':>9} {'AvgSpeed':>9} {'AvgPathEff':>11}")
    print(f"{'-'*115}")
    for m in methods:
        a = agg[m]
        print(f"{m:<18} {a['success']:>8}/{N_RANDOM_SCENARIOS} "
              f"{a['collisions']:>9}/{N_RANDOM_SCENARIOS} "
              f"{np.mean(a['min_clearances']):>11.3f} {np.mean(a['steps']):>9.1f} "
              f"{np.mean(a['avg_speeds']):>9.2f} {np.mean(a['efficiencies']):>11.2f}")

    print(f"\nDone! Plots saved to {save_dir}")
