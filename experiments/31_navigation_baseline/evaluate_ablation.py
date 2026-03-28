"""
Navigation baseline evaluation — NO CBF, just [kx, ky].
Focus: can the policy learn to navigate around obstacles and reach the target?

Outputs:
  plots/combined_scenarios.png  (3-col: traj | speed | distance to obstacles)
  plots/aggregate_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env_dynamic import NavigationBaselineEnv

# --- CONFIG ---
MODEL_PATH = "./models_dynamic/dynamic_5000k_model"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
OBS_RADIUS = 5.0

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
    env = NavigationBaselineEnv()
    model = PPO.load(MODEL_PATH)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # =====================================================================
    # COMBINED PLOT: 3 columns — Trajectory | Speed | Distance to Obstacles
    # =====================================================================
    OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
    OBS_LABELS = ["Obs 1", "Obs 2", "Obs 3"]
    TIME_MARKER_INTERVAL = 50
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 3, figsize=(27, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1, 1]})
    fig.suptitle("NAVIGATION BASELINE (kx, ky only, NO CBF): Can the policy navigate?",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]

        # --- Column 1: Trajectory ---
        ax = axs[i, 0]
        for j in range(3):
            ax.add_patch(plt.Circle(scen["obs"][j], OBS_RADIUS, color="red", alpha=0.2))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                                color="green", alpha=0.3))

        ax.plot(r["traj_x"], r["traj_y"], color="black", linewidth=2, zorder=5)

        for t in range(0, len(r["traj_x"]) - 1, TIME_MARKER_INTERVAL):
            ax.plot(r["traj_x"][t], r["traj_y"][t], marker='s', color='black',
                    markersize=4, zorder=6)
            ax.text(r["traj_x"][t], r["traj_y"][t] + 0.8, f"t={t}", fontsize=7,
                    color='black', ha='center', zorder=7)

        status = ("REACHED" if r["reached_target"] else "FAIL") + \
                 (" COLLISION" if r["collided"] else "")
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

        # --- Column 2: Speed ---
        ax = axs[i, 1]
        ax.plot(r["speeds"], color="black", linewidth=2)
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, label="Max")
        ax.set_title(f"{scen['name']} -- Speed", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(-0.1, 4.5)
        ax.grid(True, alpha=0.3)

        # --- Column 3: Distance to obstacles ---
        ax = axs[i, 2]
        for oi in range(3):
            ax.plot(r["per_obs_dists"][oi], color=OBS_COLORS[oi], linewidth=1.5,
                    label=OBS_LABELS[oi])
        ax.axhline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7, label="Collision")
        ax.set_title(f"{scen['name']} -- Distance to Obstacles", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Distance to surface (m)")
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
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"NAVIGATION BASELINE (no CBF): {N_RANDOM_SCENARIOS} Random Scenarios",
                 fontsize=16)

    ax = axs[0, 0]
    val = agg["success"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
    ax.set_ylim(0, 110)
    ax.text(0, val + 2, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 1]
    val = agg["collisions"] / N_RANDOM_SCENARIOS * 100
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collisions (no CBF protection!)")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
    ax.set_ylim(0, max(val * 1.3, 10))
    ax.text(0, val + 0.5, f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 2]
    val = np.mean(agg["min_clearances"])
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    ax.text(0, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 0]
    val = np.mean(agg["efficiencies"])
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 1]
    val = np.mean(agg["steps"])
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
    ax.text(0, val + 1, f"{val:.0f}", ha="center", fontsize=10)

    ax = axs[1, 2]
    val = np.mean(agg["avg_speeds"])
    ax.bar([0], [val], color="black", alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Speed")
    ax.set_xticks([0]); ax.set_xticklabels(["No CBF"], fontsize=9)
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

    print(f"\nDone! Plots saved to {save_dir}")
