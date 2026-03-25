"""
2-obstacle ablation: fixed-alpha vs dynamic-alpha.
Key question: does alpha recover between obstacles?

Outputs:
  plots/1_trajectories.png       — trajectory overlay per scenario
  plots/2_cbf_intervention.png   — ||k_nom - safe_u|| over time
  plots/3_aggregate_metrics.png  — success/collision/clearance/efficiency (100 random)
  plots/4_alpha_adaptation.png   — alpha + dist-to-nearest-obstacle over time
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env_dynamic import MultiObstacleDynamicEnv
from env_fixed_alpha import MultiObstacleFixedAlphaEnv

# --- CONFIG ---
DYNAMIC_MODEL_PATH = "./models_dynamic/dynamic_900k_model"
FIXED_ALPHAS = [0.1, 0.5, 1.0, 5.0]
_cmap = plt.cm.plasma
FIXED_ALPHA_COLORS = {a: _cmap((a - 0.1) / (5.0 - 0.1)) for a in FIXED_ALPHAS}
DYNAMIC_COLOR = "black"
MAX_STEPS = 200
N_RANDOM_SCENARIOS = 100

# Hand-picked 2-obstacle scenarios
SCENARIOS = [
    {"name": "Straight Gauntlet",
     "obs1": np.array([3.5, 0.0]), "obs2": np.array([7.0, 0.0]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Staggered",
     "obs1": np.array([3.0, 1.0]), "obs2": np.array([7.0, -1.0]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Zigzag",
     "obs1": np.array([3.0, -1.5]), "obs2": np.array([6.5, 1.5]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Tight Corridor",
     "obs1": np.array([4.0, 1.2]), "obs2": np.array([4.0, -1.2]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Near-Far",
     "obs1": np.array([2.5, 0.0]), "obs2": np.array([8.0, 0.0]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 0.7},
    {"name": "Diagonal Block",
     "obs1": np.array([3.5, 1.0]), "obs2": np.array([6.5, -1.0]),
     "target_pos": np.array([10.0, -2.0]), "target_radius": 0.8},
    {"name": "Wall Squeeze",
     "obs1": np.array([4.0, 0.5]), "obs2": np.array([7.0, 0.5]),
     "target_pos": np.array([10.0, 3.5]), "target_radius": 0.6},
    {"name": "Funnel",
     "obs1": np.array([3.0, 2.0]), "obs2": np.array([3.0, -2.0]),
     "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
]


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.obstacle1_pos = scen["obs1"].copy()
    env.obstacle2_pos = scen["obs2"].copy()
    env.obstacle1_radius = 1.0
    env.obstacle2_radius = 1.0
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_dynamic_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y, alphas = [], [], []
    dist1_list, dist2_list, min_dist_list = [], [], []
    cbf_interventions = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        d1 = np.linalg.norm(env.robot_pos - env.obstacle1_pos) - env.obstacle1_radius
        d2 = np.linalg.norm(env.robot_pos - env.obstacle2_pos) - env.obstacle2_radius
        dist1_list.append(d1)
        dist2_list.append(d2)
        min_dist_list.append(min(d1, d2))

        action, _ = model.predict(obs, deterministic=True)
        alphas.append(action[0])
        k_nom = np.array([action[1], action[2]])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        safe_u = info.get("safe_u", k_nom)
        cbf_interventions.append(np.linalg.norm(k_nom - safe_u))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(min_dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "alphas": alphas,
        "dist1": dist1_list, "dist2": dist2_list, "min_dist": min_dist_list,
        "cbf_interventions": cbf_interventions,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(min_dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def run_fixed_episode(env, model, scen):
    obs = setup_scenario(env, scen)
    traj_x, traj_y = [], []
    dist1_list, dist2_list, min_dist_list = [], [], []
    cbf_interventions = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        d1 = np.linalg.norm(env.robot_pos - env.obstacle1_pos) - env.obstacle1_radius
        d2 = np.linalg.norm(env.robot_pos - env.obstacle2_pos) - env.obstacle2_radius
        dist1_list.append(d1)
        dist2_list.append(d2)
        min_dist_list.append(min(d1, d2))

        action, _ = model.predict(obs, deterministic=True)
        k_nom = np.array([action[0], action[1]])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        safe_u = info.get("safe_u", k_nom)
        cbf_interventions.append(np.linalg.norm(k_nom - safe_u))

        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    collided = min(min_dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y,
        "dist1": dist1_list, "dist2": dist2_list, "min_dist": min_dist_list,
        "cbf_interventions": cbf_interventions,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(min_dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def generate_random_scenarios(n, seed=42):
    rng = np.random.RandomState(seed)
    scenarios = []
    for i in range(n):
        obs1_x = rng.uniform(2.0, 5.0)
        obs1_y = rng.uniform(-2.0, 2.0)
        obs2_x = rng.uniform(5.5, 8.5)
        obs2_y = rng.uniform(-2.0, 2.0)
        target_y = rng.uniform(-4.0, 4.0)
        target_radius = rng.uniform(0.3, 2.0)
        scenarios.append({
            "name": f"Random_{i}",
            "obs1": np.array([obs1_x, obs1_y]),
            "obs2": np.array([obs2_x, obs2_y]),
            "target_pos": np.array([10.0, target_y]),
            "target_radius": target_radius,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dynamic-alpha model...")
    dyn_env = MultiObstacleDynamicEnv()
    dyn_model = PPO.load(DYNAMIC_MODEL_PATH)

    fixed_models = {}
    fixed_envs = {}
    for alpha in FIXED_ALPHAS:
        path = f"./models_fixed/fixed_alpha_{alpha}_900k_model"
        print(f"Loading fixed alpha={alpha} from {path}...")
        fixed_models[alpha] = PPO.load(path)
        fixed_envs[alpha] = MultiObstacleFixedAlphaEnv(alpha=alpha)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        dyn = run_dynamic_episode(dyn_env, dyn_model, scen)
        fixed_results = {}
        for alpha in FIXED_ALPHAS:
            fixed_results[alpha] = run_fixed_episode(fixed_envs[alpha], fixed_models[alpha], scen)
        all_scenarios.append({"scen": scen, "dyn": dyn, "fixed": fixed_results})

    # =====================================================================
    # PLOT 1: Trajectories
    # =====================================================================
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(12, 5 * len(SCENARIOS)))
    fig.suptitle(r"2-Obstacle: Fixed $\alpha$ vs Dynamic $\alpha$", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        ax = axs[i]

        ax.add_patch(plt.Circle(scen["obs1"], 1.0, color="red", alpha=0.3))
        ax.add_patch(plt.Circle(scen["obs2"], 1.0, color="red", alpha=0.3))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"], color="green", alpha=0.3))

        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["traj_x"], r["traj_y"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.5, linestyle="--", alpha=0.6, label=rf"Fixed $\alpha$={fa}")

        dyn_x, dyn_y = dyn["traj_x"], dyn["traj_y"]
        ax.plot(dyn_x, dyn_y, color="gray", linestyle="--", alpha=0.3)
        sc = ax.scatter(dyn_x[:-1], dyn_y[:-1], c=dyn["alphas"], cmap="plasma",
                        vmin=0.1, vmax=5.0, s=25, zorder=5, label=r"Dynamic $\alpha$")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$ Value")

        status = "REACHED" if dyn["reached_target"] else ("COLLISION" if dyn["collided"] else "TIMEOUT")
        ax.set_title(f"{scen['name']} | {status} | Steps: {dyn['steps']}", fontsize=11)
        ax.set_xlim(-1, 12)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "1_trajectories.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/1_trajectories.png")

    # =====================================================================
    # PLOT 2: CBF intervention
    # =====================================================================
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(10, 4 * len(SCENARIOS)))
    fig.suptitle(r"CBF Intervention: $\|k_{nom} - u_{safe}\|$ Over Time", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        ax = axs[i]
        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["cbf_interventions"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.2, linestyle="--", alpha=0.6, label=rf"Fixed $\alpha$={fa}")
        ax.plot(dyn["cbf_interventions"], color=DYNAMIC_COLOR, linewidth=2, label=r"Dynamic $\alpha$")
        ax.set_title(scen["name"], fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(r"$\|k_{nom} - u_{safe}\|$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "2_cbf_intervention.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/2_cbf_intervention.png")

    # =====================================================================
    # PLOT 4: Alpha adaptation — with dist to BOTH obstacles
    # =====================================================================
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(10, 4 * len(SCENARIOS)))
    fig.suptitle(r"Dynamic $\alpha$ Adaptation Over Time (2 Obstacles)", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn = data["scen"], data["dyn"]
        ax = axs[i]

        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.plot(dyn["alphas"], color="purple", linewidth=2, label=r"$\alpha$")
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_dist = ax.twinx()
        ax_dist.set_ylabel("Distance to Obstacle (m)", color="darkorange")
        ax_dist.plot(dyn["dist1"], color="darkorange", linewidth=1.5, linestyle="-.", label="Dist Obs 1")
        ax_dist.plot(dyn["dist2"], color="coral", linewidth=1.5, linestyle=":", label="Dist Obs 2")
        ax_dist.tick_params(axis="y", labelcolor="darkorange")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for fa in FIXED_ALPHAS:
            ax.axhline(fa, color=FIXED_ALPHA_COLORS[fa], linewidth=1, linestyle=":", alpha=0.4)

        ax.set_title(scen["name"], fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "4_alpha_adaptation.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/4_alpha_adaptation.png")

    # =====================================================================
    # BATCH: 100 random scenarios
    # =====================================================================
    print(f"\nRunning {N_RANDOM_SCENARIOS} random scenarios...")
    random_scenarios = generate_random_scenarios(N_RANDOM_SCENARIOS)

    methods = [rf"$\alpha$={a}" for a in FIXED_ALPHAS] + [r"Dynamic $\alpha$"]
    method_colors = [FIXED_ALPHA_COLORS[a] for a in FIXED_ALPHAS] + [DYNAMIC_COLOR]
    agg = {m: {"success": 0, "collisions": 0, "min_clearances": [], "efficiencies": [], "steps": []}
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

        for j, fa in enumerate(FIXED_ALPHAS):
            r = run_fixed_episode(fixed_envs[fa], fixed_models[fa], scen)
            m = methods[j]
            agg[m]["success"] += int(r["reached_target"])
            agg[m]["collisions"] += int(r["collided"])
            agg[m]["min_clearances"].append(r["min_clearance"])
            agg[m]["efficiencies"].append(r["path_efficiency"])
            agg[m]["steps"].append(r["steps"])

    # =====================================================================
    # PLOT 3: Aggregate metrics
    # =====================================================================
    n_methods = len(methods)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Aggregate Metrics — 2 Obstacles ({N_RANDOM_SCENARIOS} Random Scenarios)", fontsize=16)
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

    ax = axs[1, 0]
    vals = [np.mean(agg[m]["min_clearances"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin (higher = safer)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    ax = axs[1, 1]
    vals = [np.mean(agg[m]["efficiencies"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line Dist")
    ax.set_title("Path Efficiency (closer to 1.0 = more direct)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "3_aggregate_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/3_aggregate_metrics.png")

    # =====================================================================
    # Console summary
    # =====================================================================
    print(f"\n{'='*110}")
    print("HAND-PICKED SCENARIO RESULTS")
    print(f"{'='*110}")
    print(f"{'Scenario':<22} {'Method':<18} {'Reached':>8} {'Collided':>9} {'MinDist':>9} {'Steps':>7} {'PathEff':>9}")
    print(f"{'-'*110}")

    for data in all_scenarios:
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        for alpha in FIXED_ALPHAS:
            r = fixed_results[alpha]
            print(f"{scen['name']:<22} {'Fixed a=' + str(alpha):<18} "
                  f"{'Yes' if r['reached_target'] else 'No':>8} "
                  f"{'YES' if r['collided'] else 'No':>9} "
                  f"{r['min_clearance']:>9.3f} {r['steps']:>7} {r['path_efficiency']:>9.2f}")
        print(f"{scen['name']:<22} {'Dynamic':<18} "
              f"{'Yes' if dyn['reached_target'] else 'No':>8} "
              f"{'YES' if dyn['collided'] else 'No':>9} "
              f"{dyn['min_clearance']:>9.3f} {dyn['steps']:>7} {dyn['path_efficiency']:>9.2f}")
        print()

    print(f"{'='*110}")
    print(f"AGGREGATE ({N_RANDOM_SCENARIOS} RANDOM SCENARIOS)")
    print(f"{'='*110}")
    print(f"{'Method':<18} {'Success':>10} {'Collisions':>11} {'AvgMinDist':>11} {'AvgSteps':>9} {'AvgPathEff':>11}")
    print(f"{'-'*110}")
    for m in methods:
        a = agg[m]
        print(f"{m:<18} {a['success']:>8}/{N_RANDOM_SCENARIOS} {a['collisions']:>9}/{N_RANDOM_SCENARIOS} "
              f"{np.mean(a['min_clearances']):>11.3f} {np.mean(a['steps']):>9.1f} {np.mean(a['efficiencies']):>11.2f}")

    print(f"\nDone! Plots saved to {save_dir}")
