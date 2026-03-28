"""
Level 4: ISSf-CBF with learnable alpha + phi + bias + radius noise.
Action is [kx, ky, alpha, phi]. QP enforces safety with robustness margin.

ISSf-CBF constraint: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)

Combined uncertainty:
  1. DYNAMICS: constant bias (wind/external force)
  2. PERCEPTION: radius estimation error (obstacle appears smaller than it is)

Outputs:
  plots/combined_scenarios.png  (6-col: traj | alpha+phi+obszone | speed | alpha vs dist | phi vs dist | policy map)
  plots/aggregate_metrics.png
  plots/conclusion.txt
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
import os

from env_dynamic import Level4ISSfCBFEnv

# --- CONFIG ---
MODEL_PATH = "./models_dynamic/dynamic_5000k_model"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
TRUE_OBS_RADIUS = 5.0
EVAL_BIAS_MAGNITUDE = 0.7
EVAL_RADIUS_ERROR = -1.0  # obstacles are 1m larger than estimated

OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
OBS_LABELS = ["Obs 1", "Obs 2", "Obs 3"]
TIME_MARKER_INTERVAL = 50

SCENARIOS = [
    {"name": "Straight Path",
     "obs": [np.array([30.0, 6.0]), np.array([50.0, -6.0]), np.array([70.0, 6.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Obstacles on Path",
     "obs": [np.array([30.0, 0.0]), np.array([55.0, 2.0]), np.array([75.0, -1.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "All Upper",
     "obs": [np.array([25.0, 5.0]), np.array([50.0, 7.0]), np.array([75.0, 4.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "All Lower",
     "obs": [np.array([25.0, -5.0]), np.array([50.0, -7.0]), np.array([75.0, -4.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Narrow Gap",
     "obs": [np.array([50.0, 6.0]), np.array([50.0, -6.0]), np.array([75.0, 0.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Clustered",
     "obs": [np.array([40.0, 3.0]), np.array([50.0, -3.0]), np.array([45.0, -7.0])],
     "target_pos": np.array([100.0, 0.0]), "target_radius": 2.0},
    {"name": "Spread",
     "obs": [np.array([20.0, 7.0]), np.array([50.0, -5.0]), np.array([80.0, 3.0])],
     "target_pos": np.array([100.0, -2.0]), "target_radius": 2.0},
    {"name": "Off-Center Target",
     "obs": [np.array([30.0, 2.0]), np.array([55.0, -4.0]), np.array([70.0, 5.0])],
     "target_pos": np.array([100.0, 5.0]), "target_radius": 2.0},
]


def setup_scenario(env, scen, bias=None, radius_error=None):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    if bias is not None:
        env.bias = bias.copy()
    if radius_error is not None:
        env.true_radius = [TRUE_OBS_RADIUS] * 3
        env.estimated_radius = [max(TRUE_OBS_RADIUS + radius_error, 1.0)] * 3
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_episode(env, model, scen, bias=None, radius_error=None):
    obs = setup_scenario(env, scen, bias=bias, radius_error=radius_error)
    traj_x, traj_y, speeds = [], [], []
    alphas, phis = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    total_reward = 0.0

    true_radius = env.true_radius[0] if hasattr(env, 'true_radius') else TRUE_OBS_RADIUS

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - true_radius
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


def generate_random_scenarios(n, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    x_bands = [(15.0, 35.0), (40.0, 60.0), (65.0, 85.0)]
    for i in range(n):
        target_y = rng.uniform(-3.0, 3.0)
        target_radius = rng.uniform(1.5, 3.0)
        obs_list = []
        for x_low, x_high in x_bands:
            x = rng.uniform(x_low, x_high)
            y = rng.uniform(-8.0, 8.0)
            obs_list.append(np.array([x, y]))
        # Random bias angle
        bias_angle = rng.uniform(0, 2 * np.pi)
        bias = EVAL_BIAS_MAGNITUDE * np.array([np.cos(bias_angle), np.sin(bias_angle)])
        # Random radius error in U(-1, 0)
        radius_error = rng.uniform(-1.0, 0.0)
        scenarios.append({
            "name": f"Random_{i}",
            "obs": obs_list,
            "target_pos": np.array([100.0, target_y]),
            "target_radius": target_radius,
            "bias": bias,
            "radius_error": radius_error,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = Level4ISSfCBFEnv()
    model = PPO.load(MODEL_PATH)

    # Hand-picked: upward bias + fixed radius error
    hand_picked_bias = EVAL_BIAS_MAGNITUDE * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        result = run_episode(env, model, scen, bias=hand_picked_bias,
                             radius_error=EVAL_RADIUS_ERROR)
        all_scenarios.append({"scen": scen, "result": result})

    # =====================================================================
    # COMBINED PLOT: 6 columns
    # Trajectory | Alpha+Phi+ObsZone | Speed | Alpha vs Dist | Phi vs Dist | Policy Map
    # =====================================================================
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 6, figsize=(54, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1, 1, 1, 1, 1]})
    fig.suptitle(f"LEVEL 4: ISSf-CBF (kx, ky, alpha, phi) -- bias={EVAL_BIAS_MAGNITUDE}, radius_error={EVAL_RADIUS_ERROR}",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        est_radius = TRUE_OBS_RADIUS + EVAL_RADIUS_ERROR

        # --- Column 1: Trajectory with alpha colormap + bias arrow + radius gap ---
        ax = axs[i, 0]
        for j in range(3):
            # True radius (solid red)
            ax.add_patch(plt.Circle(scen["obs"][j], TRUE_OBS_RADIUS,
                                    color="red", alpha=0.2, linestyle="-"))
            # Estimated radius (dashed blue)
            ax.add_patch(plt.Circle(scen["obs"][j], est_radius,
                                    color="blue", alpha=0.1, linestyle="--", linewidth=1.5,
                                    fill=False))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                                color="green", alpha=0.3))

        # Alpha colormap on path
        if len(r["traj_x"]) > 1 and len(r["alphas"]) > 0:
            points = np.array([r["traj_x"][:-1], r["traj_y"][:-1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alpha_vals = np.array(r["alphas"][:len(segments)])
            lc = LineCollection(segments, cmap="coolwarm", linewidth=2.5, zorder=5)
            lc.set_array(alpha_vals)
            lc.set_clim(0.1, 5.0)
            ax.add_collection(lc)
            cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label("alpha", fontsize=8)

        # Bias arrow annotation
        ax.annotate("", xy=(5, 12), xytext=(5, 8),
                     arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
        ax.text(5, 13, f"bias={EVAL_BIAS_MAGNITUDE:.1f}", fontsize=8,
                color="purple", ha="center", fontweight="bold")

        # Radius gap annotation
        ax.text(95, -13, f"r_true={TRUE_OBS_RADIUS:.0f}, r_est={est_radius:.0f}\ngap={EVAL_RADIUS_ERROR:.1f}m",
                fontsize=7, color="blue", ha="center",
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

        for t in range(0, len(r["traj_x"]) - 1, TIME_MARKER_INTERVAL):
            ax.plot(r["traj_x"][t], r["traj_y"][t], marker='s', color='black',
                    markersize=4, zorder=6)
            ax.text(r["traj_x"][t], r["traj_y"][t] + 0.8, f"t={t}", fontsize=7,
                    color='black', ha='center', zorder=7)

        status = ("REACHED" if r["reached_target"] else "FAIL") + \
                 (" COLLISION" if r["collided"] else "")
        ax.set_title(f"{scen['name']} -- {status}", fontsize=11)
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

        # --- Column 2: Alpha + Phi over time + obstacle zone shading ---
        ax = axs[i, 1]
        ax.plot(r["alphas"], color="blue", linewidth=1.5, label="alpha")
        ax2 = ax.twinx()
        ax2.plot(r["phis"], color="red", linewidth=1.5, label="phi", alpha=0.7)
        ax2.set_ylabel("Phi", color="red", fontsize=9)
        ax2.tick_params(axis='y', labelcolor='red')
        # Shade regions where robot is close to each obstacle
        for oi in range(3):
            close_mask = np.array(r["per_obs_dists"][oi]) < 5.0
            for start_idx in range(len(close_mask)):
                if close_mask[start_idx]:
                    ax.axvspan(start_idx, start_idx + 1, color=OBS_COLORS[oi],
                               alpha=0.1)
        ax.set_title(f"{scen['name']} -- Alpha+Phi + Obs Zone", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Alpha", color="blue", fontsize=9)
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 5.5)
        ax.grid(True, alpha=0.3)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 3: Speed ---
        ax = axs[i, 2]
        ax.plot(r["speeds"], color="black", linewidth=2)
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, label="Max")
        ax.set_title(f"{scen['name']} -- Speed", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(-0.1, 4.5)
        ax.grid(True, alpha=0.3)

        # --- Column 4: Alpha vs Obstacle Distance ---
        ax = axs[i, 3]
        min_dists_at_step = [min(r["per_obs_dists"][oi][t] for oi in range(3))
                             for t in range(len(r["alphas"]))]
        ax.scatter(min_dists_at_step, r["alphas"], c=range(len(r["alphas"])),
                   cmap="viridis", s=8, alpha=0.6)
        ax.axvline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7, label="Collision")
        ax.set_title(f"{scen['name']} -- Alpha vs Obs Dist", fontsize=11)
        ax.set_xlabel("Min Distance to Obstacle Surface (m)")
        ax.set_ylabel("Alpha")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

        # --- Column 5: Phi vs Obstacle Distance ---
        ax = axs[i, 4]
        ax.scatter(min_dists_at_step, r["phis"], c=range(len(r["phis"])),
                   cmap="magma", s=8, alpha=0.6)
        ax.axvline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7, label="Collision")
        ax.set_title(f"{scen['name']} -- Phi vs Obs Dist", fontsize=11)
        ax.set_xlabel("Min Distance to Obstacle Surface (m)")
        ax.set_ylabel("Phi")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

        # --- Column 6: Policy Map (alpha+phi scatter) ---
        ax = axs[i, 5]
        sc = ax.scatter(r["alphas"], r["phis"], c=min_dists_at_step,
                        cmap="RdYlGn", s=10, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Min Obs Dist (m)", fontsize=8)
        ax.set_title(f"{scen['name']} -- Policy Map", fontsize=11)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Phi")
        ax.grid(True, alpha=0.3)

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
        result = run_episode(env, model, scen, bias=scen["bias"],
                             radius_error=scen["radius_error"])
        agg["success"] += int(result["reached_target"])
        agg["collisions"] += int(result["collided"])
        agg["min_clearances"].append(result["min_clearance"])
        agg["efficiencies"].append(result["path_efficiency"])
        agg["steps"].append(result["steps"])
        agg["avg_speeds"].append(np.mean(result["speeds"]))

    # =====================================================================
    # AGGREGATE METRICS PLOT
    # =====================================================================
    fig, axs_agg = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"LEVEL 4: ISSf-CBF (alpha+phi) -- {N_RANDOM_SCENARIOS} Random Scenarios "
                 f"(bias={EVAL_BIAS_MAGNITUDE}, radius_err={EVAL_RADIUS_ERROR})",
                 fontsize=16)

    ax = axs_agg[0, 0]
    val = agg["success"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
    ax.set_ylim(0, 110)
    ax.text(0, val + 2, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs_agg[0, 1]
    val = agg["collisions"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collisions (phi should help!)")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
    ax.set_ylim(0, max(val * 1.3, 10))
    ax.text(0, val + 0.5, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs_agg[0, 2]
    val = np.mean(agg["min_clearances"])
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    ax.text(0, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs_agg[1, 0]
    val = np.mean(agg["efficiencies"])
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs_agg[1, 1]
    val = np.mean(agg["steps"])
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
    ax.text(0, val + 1, f"{val:.0f}", ha="center", fontsize=10)

    ax = axs_agg[1, 2]
    val = np.mean(agg["avg_speeds"])
    ax.bar([0], [val], color="forestgreen", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Speed")
    ax.set_xticks([0]); ax.set_xticklabels(["ISSf-CBF"], fontsize=9)
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
    print(f"HAND-PICKED SCENARIOS (bias={EVAL_BIAS_MAGNITUDE} upward, radius_error={EVAL_RADIUS_ERROR})")
    print(f"{'='*80}")
    print(f"{'Scenario':<25} {'Reached':>8} {'Collided':>9} {'MinDist':>9} "
          f"{'Steps':>7} {'AvgSpd':>8} {'PathEff':>9}")
    print(f"{'-'*80}")

    for data in all_scenarios:
        scen, r = data["scen"], data["result"]
        print(f"{scen['name']:<25} "
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

    # =====================================================================
    # Conclusion
    # =====================================================================
    conclusion = f"""Level 4: ISSf-CBF with Alpha + Phi -- Evaluation Results
=========================================================

Setup:
  - Action: [kx, ky, alpha, phi] -- full ISSf-CBF with learnable robustness margin
  - ISSf-CBF constraint: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)
  - Constant bias disturbance: magnitude={EVAL_BIAS_MAGNITUDE}, hand-picked=upward (pi/2)
  - Radius estimation error: hand-picked={EVAL_RADIUS_ERROR}m (true=5.0, estimated=4.0)
  - Random scenarios: bias angle U(0,2pi), radius error U(-1,0)
  - Combined uncertainty: dynamics mismatch + perception error
  - {N_RANDOM_SCENARIOS} random scenarios + {len(SCENARIOS)} hand-picked scenarios

Aggregate Results ({N_RANDOM_SCENARIOS} random scenarios):
  - Success rate:      {agg['success']}/{N_RANDOM_SCENARIOS} ({agg['success']/N_RANDOM_SCENARIOS*100:.0f}%)
  - Collision rate:    {agg['collisions']}/{N_RANDOM_SCENARIOS} ({agg['collisions']/N_RANDOM_SCENARIOS*100:.0f}%)
  - Avg min clearance: {np.mean(agg['min_clearances']):.3f}m
  - Avg steps:         {np.mean(agg['steps']):.1f}
  - Avg speed:         {np.mean(agg['avg_speeds']):.2f} m/s
  - Avg path eff:      {np.mean(agg['efficiencies']):.2f}

Key Question: Does phi finally show adaptive behavior with both sources of uncertainty?
  - Collision rate: {agg['collisions']/N_RANDOM_SCENARIOS*100:.0f}%
  - The ISSf-CBF adds a robustness margin: (||Lgh||^2 * phi) / h(x).
  - This margin grows when close to obstacles (small h) and when phi is large.
  - With BOTH dynamics mismatch (bias) AND perception error (wrong radius),
    the phi term should be essential for maintaining safety.
  - Compare to Level 3 (alpha-only with bias): does phi reduce collisions?
  - The policy should learn to increase phi near obstacles when uncertainty is high.
"""
    with open(os.path.join(save_dir, "conclusion.txt"), "w") as f:
        f.write(conclusion)
    print(f"Saved: plots/conclusion.txt")

    print(f"\nDone! Plots saved to {save_dir}")
