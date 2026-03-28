"""
Level 2: CBF with learnable alpha evaluation.
Action is [kx, ky, alpha]. QP enforces safety.
No disturbance, no radius noise. TRUE obs_radius = 5.0 everywhere.

Outputs:
  plots/combined_scenarios.png  (4-col: traj | alpha+obszone | speed | alpha vs dist)
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

from env_dynamic import Level2CBFAlphaEnv

# --- CONFIG ---
MODEL_PATH = "./models_dynamic/dynamic_5000k_model"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
OBS_RADIUS = 5.0

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


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, speeds = [], [], []
    alphas = []
    dist_list = []
    per_obs_dists = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - OBS_RADIUS
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])

        action, _ = model.predict(obs, deterministic=True)
        alphas.append(float(action[2]))
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
        "alphas": alphas,
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
        scenarios.append({
            "name": f"Random_{i}",
            "obs": obs_list,
            "target_pos": np.array([100.0, target_y]),
            "target_radius": target_radius,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = Level2CBFAlphaEnv()
    model = PPO.load(MODEL_PATH)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # =====================================================================
    # COMBINED PLOT: 4 columns — Trajectory | Alpha+ObsZone | Speed | Alpha vs Dist
    # =====================================================================
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 4, figsize=(36, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    fig.suptitle("LEVEL 2: CBF + Learnable Alpha (kx, ky, alpha) -- No disturbance, no radius noise",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]

        # --- Column 1: Trajectory with alpha colormap ---
        ax = axs[i, 0]
        for j in range(3):
            circle = plt.Circle(scen["obs"][j], OBS_RADIUS, color="red", alpha=0.2)
            ax.add_patch(circle)
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

        # --- Column 2: Alpha over time + obstacle zone shading ---
        ax = axs[i, 1]
        ax.plot(r["alphas"], color="blue", linewidth=1.5, label="alpha")
        # Shade regions where robot is close to each obstacle
        for oi in range(3):
            close_mask = np.array(r["per_obs_dists"][oi]) < 5.0
            for start_idx in range(len(close_mask)):
                if close_mask[start_idx]:
                    ax.axvspan(start_idx, start_idx + 1, color=OBS_COLORS[oi],
                               alpha=0.1)
        ax.set_title(f"{scen['name']} -- Alpha + Obs Zone", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Alpha")
        ax.set_ylim(0, 5.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

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
        ax.set_title(f"{scen['name']} -- Alpha vs Obs Distance", fontsize=11)
        ax.set_xlabel("Min Distance to Obstacle Surface (m)")
        ax.set_ylabel("Alpha")
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

    agg = {"success": 0, "collisions": 0, "min_clearances": [], "efficiencies": [],
           "steps": [], "avg_speeds": []}

    for idx, scen in enumerate(random_scenarios):
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{N_RANDOM_SCENARIOS}...")
        result = run_episode(env, model, scen)
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
    fig.suptitle(f"LEVEL 2: CBF + Alpha -- {N_RANDOM_SCENARIOS} Random Scenarios",
                 fontsize=16)

    ax = axs_agg[0, 0]
    val = agg["success"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
    ax.set_ylim(0, 110)
    ax.text(0, val + 2, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs_agg[0, 1]
    val = agg["collisions"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collisions (CBF should prevent!)")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
    ax.set_ylim(0, max(val * 1.3, 10))
    ax.text(0, val + 0.5, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs_agg[0, 2]
    val = np.mean(agg["min_clearances"])
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    ax.text(0, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs_agg[1, 0]
    val = np.mean(agg["efficiencies"])
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs_agg[1, 1]
    val = np.mean(agg["steps"])
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
    ax.text(0, val + 1, f"{val:.0f}", ha="center", fontsize=10)

    ax = axs_agg[1, 2]
    val = np.mean(agg["avg_speeds"])
    ax.bar([0], [val], color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Speed")
    ax.set_xticks([0]); ax.set_xticklabels(["CBF+Alpha"], fontsize=9)
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
    print(f"HAND-PICKED SCENARIOS")
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
    conclusion = f"""Level 2: CBF + Learnable Alpha -- Evaluation Results
=====================================================

Setup:
  - Action: [kx, ky, alpha] -- policy chooses navigation gains + CBF aggressiveness
  - Standard CBF constraint: Lgh @ u >= -alpha * h(x)
  - No disturbance, no radius noise. TRUE obs_radius = 5.0
  - {N_RANDOM_SCENARIOS} random scenarios + {len(SCENARIOS)} hand-picked scenarios

Aggregate Results ({N_RANDOM_SCENARIOS} random scenarios):
  - Success rate:      {agg['success']}/{N_RANDOM_SCENARIOS} ({agg['success']/N_RANDOM_SCENARIOS*100:.0f}%)
  - Collision rate:    {agg['collisions']}/{N_RANDOM_SCENARIOS} ({agg['collisions']/N_RANDOM_SCENARIOS*100:.0f}%)
  - Avg min clearance: {np.mean(agg['min_clearances']):.3f}m
  - Avg steps:         {np.mean(agg['steps']):.1f}
  - Avg speed:         {np.mean(agg['avg_speeds']):.2f} m/s
  - Avg path eff:      {np.mean(agg['efficiencies']):.2f}

Key Question: Does the standard CBF guarantee safety (0% collisions)?
  - Collision rate: {agg['collisions']/N_RANDOM_SCENARIOS*100:.0f}%
  - Under ideal conditions (no disturbance, no radius error), the CBF QP
    should theoretically guarantee h(x) >= 0 at all times.
  - If collisions = 0%, the CBF works perfectly in the clean setting.
  - This establishes the baseline: CBF is sufficient when model matches reality.
  - Next question (Level 3): what happens when disturbance breaks the model?
"""
    with open(os.path.join(save_dir, "conclusion.txt"), "w") as f:
        f.write(conclusion)
    print(f"Saved: plots/conclusion.txt")

    print(f"\nDone! Plots saved to {save_dir}")
