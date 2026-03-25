"""
Proper ablation: each fixed-alpha model was trained with its own k_x,k_y policy.
Compare them against the dynamic-alpha model on identical scenarios.

Outputs:
  plots/1_trajectories.png       — hand-picked scenario trajectories
  plots/2_cbf_intervention.png   — ||k_nom - safe_u|| over time
  plots/3_aggregate_metrics.png  — success/collision/clearance/efficiency (100 random scenarios)
  plots/4_alpha_adaptation.png   — dynamic alpha vs distance over time
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import sys

from env_dynamic_optimized import AdaptiveCBFEnvOptimized
from env_fixed_alpha import FixedAlphaCBFEnv

# --- CONFIG ---
DYNAMIC_MODEL_PATH = "./models_dynamic_optimized/dynamic_optimized_900000_model"
FIXED_ALPHAS = [0.1, 0.5, 1.0, 5.0]
FIXED_ALPHA_COLORS = {0.1: "orange", 0.5: "green", 1.0: "blue", 5.0: "purple"}
DYNAMIC_COLOR = "red"
MAX_STEPS = 150
N_RANDOM_SCENARIOS = 100

# Hand-picked scenarios (includes harder ones)
SCENARIOS = [
    {"name": "Center Park",       "obs_pos": np.array([4.0, 0.1]),  "target_pos": np.array([9.0, 0.0]),  "target_radius": 1.0},
    {"name": "High Offset",       "obs_pos": np.array([5.0, -0.5]), "target_pos": np.array([8.0, 3.0]),  "target_radius": 1.5},
    {"name": "Tight Low Corner",  "obs_pos": np.array([3.0, 0.5]),  "target_pos": np.array([9.0, -2.5]), "target_radius": 0.8},
    {"name": "Dead Center Block", "obs_pos": np.array([5.0, 0.0]),  "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Early Dodge",       "obs_pos": np.array([2.0, 0.0]),  "target_pos": np.array([8.0, 4.0]),  "target_radius": 1.5},
    # --- Harder scenarios ---
    {"name": "Needle Thread",     "obs_pos": np.array([4.5, 0.0]),  "target_pos": np.array([9.0, 0.0]),  "target_radius": 0.5},
    {"name": "Close Shave",       "obs_pos": np.array([3.0, 0.0]),  "target_pos": np.array([6.0, 0.0]),  "target_radius": 0.5},
    {"name": "Double Back",       "obs_pos": np.array([4.0, -1.0]), "target_pos": np.array([9.0, -3.5]), "target_radius": 0.6},
    {"name": "Wall Hug",          "obs_pos": np.array([5.0, 2.0]),  "target_pos": np.array([9.0, 4.0]),  "target_radius": 0.7},
    {"name": "Gauntlet",          "obs_pos": np.array([3.5, 0.0]),  "target_pos": np.array([10.0, 0.0]), "target_radius": 0.5},
]


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.obstacle_pos = scen["obs_pos"].copy()
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
    """Run the dynamic-alpha model (outputs [alpha, k_x, k_y])."""
    obs = setup_scenario(env, scen)
    traj_x, traj_y, alphas, distances = [], [], [], []
    cbf_interventions = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dist = np.linalg.norm(env.robot_pos - env.obstacle_pos) - env.obstacle_radius
        distances.append(dist)

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
    collided = min(distances) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "alphas": alphas,
        "distances": distances, "cbf_interventions": cbf_interventions,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_dist": min(distances), "path_length": plen,
        "path_efficiency": efficiency,
    }


def run_fixed_episode(env, model, scen):
    """Run a fixed-alpha model (outputs [k_x, k_y] only, alpha is in the env)."""
    obs = setup_scenario(env, scen)
    traj_x, traj_y, distances = [], [], []
    cbf_interventions = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dist = np.linalg.norm(env.robot_pos - env.obstacle_pos) - env.obstacle_radius
        distances.append(dist)

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
    collided = min(distances) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y,
        "distances": distances, "cbf_interventions": cbf_interventions,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_dist": min(distances), "path_length": plen,
        "path_efficiency": efficiency,
    }


def generate_random_scenarios(n, seed=42):
    """Generate n random scenarios with varying difficulty."""
    rng = np.random.RandomState(seed)
    scenarios = []
    for i in range(n):
        obs_x = rng.uniform(2.0, 8.0)
        obs_y = rng.uniform(-2.0, 2.0)
        target_y = rng.uniform(-4.0, 4.0)
        target_radius = rng.uniform(0.3, 2.0)
        scenarios.append({
            "name": f"Random_{i}",
            "obs_pos": np.array([obs_x, obs_y]),
            "target_pos": np.array([9.0, target_y]),
            "target_radius": target_radius,
        })
    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    # Load dynamic model
    print("Loading dynamic-alpha model...")
    dyn_env = AdaptiveCBFEnvOptimized()
    dyn_model = PPO.load(DYNAMIC_MODEL_PATH)

    # Load fixed-alpha models
    fixed_models = {}
    fixed_envs = {}
    for alpha in FIXED_ALPHAS:
        path = f"./models_fixed/fixed_alpha_{alpha}_900k_model"
        print(f"Loading fixed alpha={alpha} model from {path}...")
        fixed_models[alpha] = PPO.load(path)
        fixed_envs[alpha] = FixedAlphaCBFEnv(alpha=alpha)

    # --- Collect hand-picked scenario results ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for i, scen in enumerate(SCENARIOS):
        dyn = run_dynamic_episode(dyn_env, dyn_model, scen)
        fixed_results = {}
        for alpha in FIXED_ALPHAS:
            fixed_results[alpha] = run_fixed_episode(fixed_envs[alpha], fixed_models[alpha], scen)
        all_scenarios.append({"scen": scen, "dyn": dyn, "fixed": fixed_results})

    # =====================================================================
    # PLOT 1: Trajectory overview (one row per scenario)
    # =====================================================================
    fig_traj, axs_traj = plt.subplots(len(SCENARIOS), 1, figsize=(10, 5 * len(SCENARIOS)))
    fig_traj.suptitle(r"Trajectory Comparison: Fixed $\alpha$ vs Dynamic $\alpha$", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        ax = axs_traj[i]

        ax.add_patch(plt.Circle(scen["obs_pos"], 1.0, color="red", alpha=0.3))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"], color="green", alpha=0.3))

        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["traj_x"], r["traj_y"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.5, linestyle="--", alpha=0.6,
                    label=rf"Fixed $\alpha$={fa}")

        dyn_x, dyn_y = dyn["traj_x"], dyn["traj_y"]
        ax.plot(dyn_x, dyn_y, color="gray", linestyle="--", alpha=0.3)
        sc = ax.scatter(dyn_x[:-1], dyn_y[:-1], c=dyn["alphas"], cmap="coolwarm",
                        vmin=0.1, vmax=5.0, s=25, zorder=5, label=r"Dynamic $\alpha$")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$ Value")

        status = "REACHED" if dyn["reached_target"] else ("COLLISION" if dyn["collided"] else "TIMEOUT")
        ax.set_title(f"{scen['name']} | {status} | Steps: {dyn['steps']}", fontsize=11)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

    fig_traj.tight_layout()
    fig_traj.savefig(os.path.join(save_dir, "1_trajectories.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_traj)
    print("Saved: plots/1_trajectories.png")

    # =====================================================================
    # PLOT 2: CBF intervention over time (hand-picked scenarios)
    # =====================================================================
    fig_cbf, axs_cbf = plt.subplots(len(SCENARIOS), 1, figsize=(10, 4 * len(SCENARIOS)))
    fig_cbf.suptitle(r"CBF Intervention: $\|k_{nom} - u_{safe}\|$ Over Time", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        ax = axs_cbf[i]

        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax.plot(r["cbf_interventions"], color=FIXED_ALPHA_COLORS[fa],
                    linewidth=1.2, linestyle="--", alpha=0.6,
                    label=rf"Fixed $\alpha$={fa}")

        ax.plot(dyn["cbf_interventions"], color=DYNAMIC_COLOR, linewidth=2,
                label=r"Dynamic $\alpha$")

        ax.set_title(f"{scen['name']}", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(r"$\|k_{nom} - u_{safe}\|$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

    fig_cbf.tight_layout()
    fig_cbf.savefig(os.path.join(save_dir, "2_cbf_intervention.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_cbf)
    print("Saved: plots/2_cbf_intervention.png")

    # =====================================================================
    # PLOT 4: Dynamic alpha vs distance (hand-picked scenarios)
    # =====================================================================
    fig_alpha, axs_alpha = plt.subplots(len(SCENARIOS), 1, figsize=(10, 4 * len(SCENARIOS)))
    fig_alpha.suptitle(r"Dynamic $\alpha$ Adaptation Over Time", fontsize=16, y=1.01)

    for i, data in enumerate(all_scenarios):
        scen, dyn = data["scen"], data["dyn"]
        ax = axs_alpha[i]

        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.plot(range(len(dyn["alphas"])), dyn["alphas"], color="purple", linewidth=2, label=r"$\alpha$")
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_dist = ax.twinx()
        ax_dist.set_ylabel("Distance to Obstacle (m)", color="darkorange")
        ax_dist.plot(range(len(dyn["distances"])), dyn["distances"], color="darkorange",
                     linewidth=2, linestyle="-.", label="Dist to Obs")
        ax_dist.tick_params(axis="y", labelcolor="darkorange")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for fa in FIXED_ALPHAS:
            ax.axhline(fa, color=FIXED_ALPHA_COLORS[fa], linewidth=1, linestyle=":", alpha=0.4)

        ax.set_title(f"{scen['name']}", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    fig_alpha.tight_layout()
    fig_alpha.savefig(os.path.join(save_dir, "4_alpha_adaptation.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_alpha)
    print("Saved: plots/4_alpha_adaptation.png")

    # =====================================================================
    # BATCH: 100 random scenarios for statistical comparison
    # =====================================================================
    print(f"\nRunning {N_RANDOM_SCENARIOS} random scenarios for aggregate stats...")
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
        agg[m_dyn]["min_clearances"].append(dyn["min_dist"])
        agg[m_dyn]["efficiencies"].append(dyn["path_efficiency"])
        agg[m_dyn]["steps"].append(dyn["steps"])

        for j, fa in enumerate(FIXED_ALPHAS):
            r = run_fixed_episode(fixed_envs[fa], fixed_models[fa], scen)
            m = methods[j]
            agg[m]["success"] += int(r["reached_target"])
            agg[m]["collisions"] += int(r["collided"])
            agg[m]["min_clearances"].append(r["min_dist"])
            agg[m]["efficiencies"].append(r["path_efficiency"])
            agg[m]["steps"].append(r["steps"])

    # =====================================================================
    # PLOT 3: Aggregate metrics (from 100 random scenarios)
    # =====================================================================
    n_methods = len(methods)
    fig_agg, axs_agg = plt.subplots(2, 2, figsize=(14, 10))
    fig_agg.suptitle(f"Aggregate Metrics ({N_RANDOM_SCENARIOS} Random Scenarios)", fontsize=16)

    x = np.arange(n_methods)

    # (a) Success rate
    ax = axs_agg[0, 0]
    success_rates = [agg[m]["success"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, success_rates, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    # (b) Collision rate
    ax = axs_agg[0, 1]
    collision_rates = [agg[m]["collisions"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, collision_rates, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, max(max(collision_rates) * 1.3, 10))
    for bar, val in zip(bars, collision_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    # (c) Min clearance (avg across scenarios)
    ax = axs_agg[1, 0]
    avg_clearance = [np.mean(agg[m]["min_clearances"]) for m in methods]
    bars = ax.bar(x, avg_clearance, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin (higher = safer)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    for bar, val in zip(bars, avg_clearance):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    # (d) Path efficiency (avg across scenarios)
    ax = axs_agg[1, 1]
    avg_eff = [np.mean(agg[m]["efficiencies"]) for m in methods]
    bars = ax.bar(x, avg_eff, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line Dist")
    ax.set_title("Path Efficiency (closer to 1.0 = more direct)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, avg_eff):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)

    fig_agg.tight_layout()
    fig_agg.savefig(os.path.join(save_dir, "3_aggregate_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_agg)
    print("Saved: plots/3_aggregate_metrics.png")

    # =====================================================================
    # Console summary
    # =====================================================================
    print(f"\n{'='*105}")
    print("HAND-PICKED SCENARIO RESULTS")
    print(f"{'='*105}")
    print(f"{'Scenario':<22} {'Method':<18} {'Reached':>8} {'Collided':>9} {'MinDist':>9} {'Steps':>7} {'PathEff':>9}")
    print(f"{'-'*105}")

    for data in all_scenarios:
        scen, dyn, fixed_results = data["scen"], data["dyn"], data["fixed"]
        for alpha in FIXED_ALPHAS:
            r = fixed_results[alpha]
            reached_str = "Yes" if r["reached_target"] else "No"
            collided_str = "YES" if r["collided"] else "No"
            print(f"{scen['name']:<22} {'Fixed a=' + str(alpha):<18} {reached_str:>8} {collided_str:>9} {r['min_dist']:>9.3f} {r['steps']:>7} {r['path_efficiency']:>9.2f}")
        reached_str = "Yes" if dyn["reached_target"] else "No"
        collided_str = "YES" if dyn["collided"] else "No"
        print(f"{scen['name']:<22} {'Dynamic':<18} {reached_str:>8} {collided_str:>9} {dyn['min_dist']:>9.3f} {dyn['steps']:>7} {dyn['path_efficiency']:>9.2f}")
        print()

    print(f"{'='*105}")
    print(f"AGGREGATE ({N_RANDOM_SCENARIOS} RANDOM SCENARIOS)")
    print(f"{'='*105}")
    print(f"{'Method':<18} {'Success':>10} {'Collisions':>11} {'AvgMinDist':>11} {'AvgSteps':>9} {'AvgPathEff':>11}")
    print(f"{'-'*105}")

    for m in methods:
        a = agg[m]
        avg_min = np.mean(a["min_clearances"])
        avg_steps = np.mean(a["steps"])
        avg_eff = np.mean(a["efficiencies"])
        print(f"{m:<18} {a['success']:>8}/{N_RANDOM_SCENARIOS} {a['collisions']:>9}/{N_RANDOM_SCENARIOS} {avg_min:>11.3f} {avg_steps:>9.1f} {avg_eff:>11.2f}")

    print(f"\nDone! Plots saved to {save_dir}")
