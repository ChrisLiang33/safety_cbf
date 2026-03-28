"""
ISSf-CBF evaluation with RADIUS ESTIMATION NOISE + RANDOMIZED OBSTACLES.
No alpha-only baseline — focus on alpha/phi behavior analysis.

Eval: worst-case radius error = -1.0 (thinks 4.0, true is 5.0).

Outputs:
  plots/combined_scenarios.png  (6-col: traj | alpha+phi+obs_zone | speed | alpha+dist | phi+dist | policy map)
  plots/aggregate_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env_dynamic import PhiCBFRadiusNoiseRandomizedEnv

# --- CONFIG ---
PHI_MODEL_PATH = "./models_dynamic/dynamic_2000k_model"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
TRUE_RADIUS = 5.0
EVAL_RADIUS_ERROR = -1.0
EVAL_ESTIMATED_RADIUS = TRUE_RADIUS + EVAL_RADIUS_ERROR  # = 4.0

PHI_COLOR = "black"

SCENARIOS = [
    {"name": "Standard Slalom",
     "obs": [np.array([30.0, 6.0]), np.array([50.0, -6.0]), np.array([70.0, 6.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Tight Slalom",
     "obs": [np.array([30.0, 5.0]), np.array([50.0, -5.0]), np.array([70.0, 5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Clustered Center",
     "obs": [np.array([40.0, 3.0]), np.array([50.0, -3.0]), np.array([60.0, 2.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Diagonal Line",
     "obs": [np.array([25.0, -5.0]), np.array([50.0, 0.0]), np.array([75.0, 5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "All Upper",
     "obs": [np.array([25.0, 6.0]), np.array([50.0, 7.0]), np.array([75.0, 5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "All Lower",
     "obs": [np.array([25.0, -6.0]), np.array([50.0, -7.0]), np.array([75.0, -5.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Narrow Gap",
     "obs": [np.array([50.0, 7.0]), np.array([50.0, -7.0]), np.array([75.0, 0.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Spread Random",
     "obs": [np.array([20.0, 8.0]), np.array([55.0, -5.0]), np.array([80.0, 3.0])],
     "target_pos": np.array([100.0, -2.0]), "target_radius": 2.0},
]


def setup_scenario(env, scen, radius_error):
    """Setup scenario with a specific radius estimation error."""
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
        env.true_radius[i] = TRUE_RADIUS
        env.estimated_radius[i] = max(TRUE_RADIUS + radius_error, 1.0)
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_episode(env, model, scen, radius_error):
    """Run episode with dynamic [kx, ky, alpha, phi] (ISSf-CBF)."""
    obs = setup_scenario(env, scen, radius_error)
    traj_x, traj_y, alphas, phis, speeds = [], [], [], [], []
    k_nom_speeds, safe_u_speeds = [], []
    dist_list_true = []
    per_obs_dists_true = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists_true = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.true_radius[i]
                      for i in range(3)]
        dist_list_true.append(min(dists_true))
        for oi in range(3):
            per_obs_dists_true[oi].append(dists_true[oi])

        action, _ = model.predict(obs, deterministic=True)
        alpha_val = float(action[2])
        phi_val = float(action[3])
        alphas.append(alpha_val)
        phis.append(phi_val)

        k_nom_before = np.array([float(action[0]), float(action[1])])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        safe_u = info.get("safe_u", k_nom_before)
        k_nom_speeds.append(np.linalg.norm(k_nom_before))
        safe_u_speeds.append(np.linalg.norm(safe_u))
        speeds.append(np.linalg.norm(safe_u))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break
    else:
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(dist_list_true) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "alphas": alphas, "phis": phis,
        "speeds": speeds, "k_nom_speeds": k_nom_speeds,
        "safe_u_speeds": safe_u_speeds,
        "dist": dist_list_true, "per_obs_dists": per_obs_dists_true,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list_true), "path_length": plen,
        "path_efficiency": efficiency,
    }


def generate_random_scenarios(n, seed=42):
    """Generate random scenarios with random obstacle placement."""
    rng = np.random.RandomState(seed)
    scenarios = []
    min_spacing = 12.0
    for i in range(n):
        target_y = rng.uniform(-3.0, 3.0)
        target_radius = rng.uniform(1.5, 3.0)
        # Random obstacle placement with spacing
        placed = []
        for _ in range(3):
            for _attempt in range(100):
                x = rng.uniform(15.0, 85.0)
                y = rng.uniform(-10.0, 10.0)
                candidate = np.array([x, y])
                too_close = any(np.linalg.norm(candidate - p) < min_spacing for p in placed)
                if not too_close:
                    placed.append(candidate)
                    break
            else:
                placed.append(np.array([rng.uniform(15.0, 85.0), rng.uniform(-10.0, 10.0)]))
        radius_error = rng.uniform(-1.0, 0.0)
        scenarios.append({
            "name": f"Random_{i}",
            "obs": placed,
            "target_pos": np.array([100.0, target_y]),
            "target_radius": target_radius,
            "radius_error": radius_error,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading model... (eval radius error={EVAL_RADIUS_ERROR}, est={EVAL_ESTIMATED_RADIUS})")
    env = PhiCBFRadiusNoiseRandomizedEnv()
    model = PPO.load(PHI_MODEL_PATH)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios (radius error={EVAL_RADIUS_ERROR})...")
    all_scenarios = []
    for scen in SCENARIOS:
        result = run_episode(env, model, scen, EVAL_RADIUS_ERROR)
        all_scenarios.append({"scen": scen, "result": result})

    # =====================================================================
    # COMBINED PLOT: 6 columns
    # Traj | Alpha+Phi+ObsZone | Speed | Alpha+Dist | Phi+Dist | Policy Map
    # =====================================================================
    TIME_MARKER_INTERVAL = 50
    OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
    OBS_LABELS = ["Obs 1", "Obs 2", "Obs 3"]
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 6, figsize=(54, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1.2, 1, 1, 1, 1]})
    fig.suptitle(rf"RADIUS EST. NOISE + RANDOM OBS (true={TRUE_RADIUS}, est={EVAL_ESTIMATED_RADIUS}): "
                 rf"ISSf-CBF $\alpha$+$\varphi$ Analysis",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]

        # --- Column 1: Trajectory ---
        ax = axs[i, 0]
        for j in range(3):
            ax.add_patch(plt.Circle(scen["obs"][j], TRUE_RADIUS, color="red", alpha=0.2))
            ax.add_patch(plt.Circle(scen["obs"][j], EVAL_ESTIMATED_RADIUS,
                                    color="gray", alpha=0.0, linestyle="--", linewidth=1.5))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                                color="green", alpha=0.3))

        ax.text(5, -12, f"TRUE r={TRUE_RADIUS}, EST r={EVAL_ESTIMATED_RADIUS}\n"
                        f"Gap={TRUE_RADIUS - EVAL_ESTIMATED_RADIUS}m",
                fontsize=7, color="red",
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

        phi_x, phi_y = r["traj_x"], r["traj_y"]
        step_skip = 2
        sc = ax.scatter(phi_x[:-1:step_skip], phi_y[:-1:step_skip],
                        c=r["alphas"][::step_skip], cmap="coolwarm",
                        vmin=0.1, vmax=5.0, s=14, zorder=5, label=r"$\alpha$+$\varphi$ (ISSf)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$")

        for t in range(0, len(phi_x) - 1, TIME_MARKER_INTERVAL):
            ax.plot(phi_x[t], phi_y[t], marker='s', color='black', markersize=4, zorder=6)
            ax.text(phi_x[t], phi_y[t] + 0.8, f"t={t}", fontsize=7, color='black',
                    ha='center', zorder=7)

        status = ("REACHED" if r["reached_target"] else "FAIL") + (" COLLISION" if r["collided"] else "")
        ax.set_title(f"{scen['name']} — {status}", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-5, 110)
        ax.set_ylim(-16, 16)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        metrics_text = (f"Steps: {r['steps']}  Reward: {r['total_reward']:.0f}  "
                        f"MinClearance: {r['min_clearance']:.2f}m  Efficiency: {r['path_efficiency']:.2f}")
        ax.text(0.02, -0.10, metrics_text, transform=ax.transAxes, fontsize=7,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

        # --- Column 2: Alpha & Phi with Obstacle Zones ---
        ax = axs[i, 1]
        ax_phi2 = ax.twinx()

        for oi in range(3):
            d = r["per_obs_dists"][oi]
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.08)
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            ax.text(t_min, 5.3, f"{OBS_LABELS[oi]}", fontsize=6,
                    color=OBS_COLORS[oi], ha="center", va="top", fontweight="bold")

        ax.plot(r["alphas"], color="purple", linewidth=2.5, label=r"$\alpha$", zorder=5)
        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_phi2.plot(r["phis"], color="darkorange", linewidth=2.5, linestyle="-.",
                     label=r"$\varphi$", zorder=5)
        ax_phi2.set_ylabel(r"$\varphi$ Value", color="darkorange")
        ax_phi2.tick_params(axis="y", labelcolor="darkorange")
        ax_phi2.set_ylim(-0.5, 10.5)

        ax.set_title(f"{scen['name']} -- Alpha & Phi (Obs Zones)", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_phi2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 3: Speed ---
        ax = axs[i, 2]
        ax.plot(r["speeds"], color=PHI_COLOR, linewidth=2, label=r"$\|u_{safe}\|$")
        ax.plot(r["k_nom_speeds"], color="gray", linewidth=1, linestyle=":",
                alpha=0.5, label=r"$\|k_{nom}\|$")
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, label="Max")
        ax.set_title(f"{scen['name']} -- Speed", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(-0.1, 4.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

        # --- Column 4: Alpha + Per-Obstacle Distance (TRUE) ---
        ax = axs[i, 3]
        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.plot(r["alphas"], color="purple", linewidth=2.5, label=r"$\alpha$", zorder=5)
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_dist = ax.twinx()
        ax_dist.set_ylabel("TRUE Dist to Obs (m)", color="gray")
        ax_dist.tick_params(axis="y", labelcolor="gray")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for oi in range(3):
            d = r["per_obs_dists"][oi]
            ax_dist.plot(d, color=OBS_COLORS[oi], linewidth=1.2, linestyle="-.",
                         alpha=0.7, label=OBS_LABELS[oi])
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.04)

        ax.set_title(f"{scen['name']} -- Alpha vs Obstacle", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 5: Phi + Per-Obstacle Distance (TRUE) ---
        ax = axs[i, 4]
        ax.set_ylabel(r"$\varphi$ Value", color="darkorange")
        ax.plot(r["phis"], color="darkorange", linewidth=2.5, label=r"$\varphi$", zorder=5)
        ax.tick_params(axis="y", labelcolor="darkorange")
        ax.set_ylim(-0.7, 10.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

        ax_dist2 = ax.twinx()
        ax_dist2.set_ylabel("TRUE Dist to Obs (m)", color="gray")
        ax_dist2.tick_params(axis="y", labelcolor="gray")
        ax_dist2.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for oi in range(3):
            d = r["per_obs_dists"][oi]
            ax_dist2.plot(d, color=OBS_COLORS[oi], linewidth=1.2, linestyle="-.",
                          alpha=0.7, label=OBS_LABELS[oi])
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.04)

        ax.set_title(f"{scen['name']} -- Phi vs Obstacle", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 6: Policy Map (alpha & phi vs TRUE distance) ---
        ax = axs[i, 5]
        min_dists_per_t = np.array(r["dist"])
        alphas_arr = np.array(r["alphas"])
        phis_arr = np.array(r["phis"])

        ax.scatter(min_dists_per_t, alphas_arr,
                   c="purple", s=10, alpha=0.5, zorder=3, label=r"$\alpha$")
        ax.scatter(min_dists_per_t, phis_arr,
                   c="darkorange", s=10, alpha=0.5, marker="^", zorder=3, label=r"$\varphi$")

        ax.set_xlabel("Min TRUE Dist to Obs Surface (m)")
        ax.set_ylabel(r"$\alpha$ / $\varphi$")
        ax.set_ylim(-1, 10.5)
        ax.set_xlim(-1, max(min_dists_per_t) * 1.05 if len(min_dists_per_t) > 0 else 20)
        ax.axvline(0, color="red", linewidth=1, linestyle=":", alpha=0.5, label="TRUE collision")
        ax.axvline(TRUE_RADIUS - EVAL_ESTIMATED_RADIUS, color="orange", linewidth=1,
                   linestyle="--", alpha=0.5, label=f"Danger zone (+{TRUE_RADIUS - EVAL_ESTIMATED_RADIUS}m)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.set_title(f"{scen['name']} -- Policy Map", fontsize=11)
        ax.grid(True, alpha=0.3)
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

    agg = {"success": 0, "collisions": 0, "min_clearances": [], "efficiencies": [],
           "steps": [], "avg_speeds": []}

    for idx, scen in enumerate(random_scenarios):
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{N_RANDOM_SCENARIOS}...")
        result = run_episode(env, model, scen, scen["radius_error"])
        agg["success"] += int(result["reached_target"])
        agg["collisions"] += int(result["collided"])
        agg["min_clearances"].append(result["min_clearance"])
        agg["efficiencies"].append(result["path_efficiency"])
        agg["steps"].append(result["steps"])
        agg["avg_speeds"].append(np.mean(result["speeds"]))

    # =====================================================================
    # AGGREGATE METRICS PLOT
    # =====================================================================
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(rf"Aggregate -- RADIUS EST. NOISE + RANDOM OBS (true={TRUE_RADIUS}, err$\sim$U(-1,0)): "
                 rf"ISSf-CBF ({N_RANDOM_SCENARIOS} Random Scenarios)",
                 fontsize=16)

    ax = axs[0, 0]
    val = agg["success"] / N_RANDOM_SCENARIOS * 100
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.set_ylim(0, 110)
    ax.text(0, val + 2, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 1]
    val = agg["collisions"] / N_RANDOM_SCENARIOS * 100
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.set_ylim(0, max(val * 1.3, 10))
    ax.text(0, val + 0.5, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 2]
    val = np.mean(agg["min_clearances"])
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin (TRUE distance)")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    ax.text(0, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 0]
    val = np.mean(agg["efficiencies"])
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 1]
    val = np.mean(agg["steps"])
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length (lower = faster)")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.text(0, val + 1, f"{val:.0f}", ha="center", fontsize=10)

    ax = axs[1, 2]
    val = np.mean(agg["avg_speeds"])
    bar = ax.bar([0], [val], color=PHI_COLOR, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Robot Speed")
    ax.set_xticks([0]); ax.set_xticklabels([r"$\alpha$+$\varphi$ (ISSf)"], fontsize=9)
    ax.axhline(3.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "aggregate_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/aggregate_metrics.png")

    # =====================================================================
    # Console summary
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"HAND-PICKED SCENARIOS (radius error={EVAL_RADIUS_ERROR}, est={EVAL_ESTIMATED_RADIUS})")
    print(f"{'='*80}")
    print(f"{'Scenario':<20} {'Reached':>8} {'Collided':>9} {'TrueMinD':>9} "
          f"{'Steps':>7} {'AvgSpd':>8} {'PathEff':>9}")
    print(f"{'-'*80}")

    for data in all_scenarios:
        scen, r = data["scen"], data["result"]
        print(f"{scen['name']:<20} "
              f"{'Yes' if r['reached_target'] else 'No':>8} "
              f"{'YES' if r['collided'] else 'No':>9} "
              f"{r['min_clearance']:>9.3f} {r['steps']:>7} "
              f"{np.mean(r['speeds']):>8.2f} {r['path_efficiency']:>9.2f}")

    print(f"\n{'='*80}")
    print(f"AGGREGATE ({N_RANDOM_SCENARIOS} RANDOM SCENARIOS)")
    print(f"{'='*80}")
    print(f"  Success:       {agg['success']}/{N_RANDOM_SCENARIOS} ({agg['success']/N_RANDOM_SCENARIOS*100:.0f}%)")
    print(f"  Collisions:    {agg['collisions']}/{N_RANDOM_SCENARIOS} ({agg['collisions']/N_RANDOM_SCENARIOS*100:.0f}%)")
    print(f"  Avg Min Dist:  {np.mean(agg['min_clearances']):.3f}m")
    print(f"  Avg Steps:     {np.mean(agg['steps']):.1f}")
    print(f"  Avg Speed:     {np.mean(agg['avg_speeds']):.2f} m/s")
    print(f"  Avg Path Eff:  {np.mean(agg['efficiencies']):.2f}")

    print(f"\nDone! Plots saved to {save_dir}")
