"""
Proper ablation: each fixed-alpha model was trained with its own k_x,k_y policy.
Compare them against the dynamic-alpha model on identical scenarios.
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import AdaptiveCBFEnv
from env_fixed_alpha import FixedAlphaCBFEnv

# --- CONFIG ---
DYNAMIC_MODEL_PATH = "../model/run4_900000_model"  # Change to your best run
FIXED_ALPHAS = [0.1, 0.5, 1.0, 5.0]
FIXED_ALPHA_COLORS = {0.1: "orange", 0.5: "green", 1.0: "blue", 5.0: "purple"}
DYNAMIC_COLOR = "red"
MAX_STEPS = 150

SCENARIOS = [
    {"name": "Center Park",       "obs_pos": np.array([4.0, 0.1]),  "target_pos": np.array([9.0, 0.0]),  "target_radius": 1.0},
    {"name": "High Offset",       "obs_pos": np.array([5.0, -0.5]), "target_pos": np.array([8.0, 3.0]),  "target_radius": 1.5},
    {"name": "Tight Low Corner",  "obs_pos": np.array([3.0, 0.5]),  "target_pos": np.array([9.0, -2.5]), "target_radius": 0.8},
    {"name": "Dead Center Block", "obs_pos": np.array([5.0, 0.0]),  "target_pos": np.array([10.0, 0.0]), "target_radius": 1.0},
    {"name": "Early Dodge",       "obs_pos": np.array([2.0, 0.0]),  "target_pos": np.array([8.0, 4.0]),  "target_radius": 1.5},
]


def setup_scenario(env, scen):
    env.reset()
    env.robot_pos = np.array([0.0, 0.0])
    env.obstacle_pos = scen["obs_pos"].copy()
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    return env._get_obs()


def run_dynamic_episode(env, model, scen):
    """Run the dynamic-alpha model (outputs [alpha, k_x, k_y])."""
    obs = setup_scenario(env, scen)
    traj_x, traj_y, alphas, distances = [], [], [], []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dist = np.linalg.norm(env.robot_pos - env.obstacle_pos) - env.obstacle_radius
        distances.append(dist)

        action, _ = model.predict(obs, deterministic=True)
        alphas.append(action[0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    return {"traj_x": traj_x, "traj_y": traj_y, "alphas": alphas, "distances": distances,
            "total_reward": total_reward, "steps": step + 1, "reached_target": reached,
            "collided": min(distances) < 0, "min_dist": min(distances)}


def run_fixed_episode(env, model, scen):
    """Run a fixed-alpha model (outputs [k_x, k_y] only, alpha is in the env)."""
    obs = setup_scenario(env, scen)
    traj_x, traj_y, distances = [], [], []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dist = np.linalg.norm(env.robot_pos - env.obstacle_pos) - env.obstacle_radius
        distances.append(dist)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            traj_x.append(env.robot_pos[0])
            traj_y.append(env.robot_pos[1])
            break

    reached = np.linalg.norm(env.robot_pos - env.target_pos) < env.target_radius
    return {"traj_x": traj_x, "traj_y": traj_y, "distances": distances,
            "total_reward": total_reward, "steps": step + 1, "reached_target": reached,
            "collided": min(distances) < 0, "min_dist": min(distances)}


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    # Load dynamic model
    print("Loading dynamic-alpha model...")
    dyn_model = PPO.load(DYNAMIC_MODEL_PATH)
    dyn_env = AdaptiveCBFEnv()

    # Load fixed-alpha models
    fixed_models = {}
    fixed_envs = {}
    for alpha in FIXED_ALPHAS:
        path = f"./models_fixed/fixed_alpha_{alpha}_900k_model"
        print(f"Loading fixed alpha={alpha} model from {path}...")
        fixed_models[alpha] = PPO.load(path)
        fixed_envs[alpha] = FixedAlphaCBFEnv(alpha=alpha)

    # --- Plot: 3 columns now ---
    fig, axs = plt.subplots(len(SCENARIOS), 3, figsize=(24, 5 * len(SCENARIOS)))
    fig.suptitle(r"Ablation: Fixed $\alpha$ (separately trained) vs Dynamic $\alpha$", fontsize=18)

    summary_rows = []

    for i, scen in enumerate(SCENARIOS):
        ax_traj = axs[i, 0]
        ax_metrics = axs[i, 1]
        ax_corr = axs[i, 2]

        # Run dynamic
        dyn = run_dynamic_episode(dyn_env, dyn_model, scen)

        # Run each fixed-alpha
        fixed_results = {}
        for alpha in FIXED_ALPHAS:
            fixed_results[alpha] = run_fixed_episode(fixed_envs[alpha], fixed_models[alpha], scen)

        # --- Left: trajectory plot ---
        obs_circle = plt.Circle(scen["obs_pos"], 1.0, color="red", alpha=0.3)
        ax_traj.add_patch(obs_circle)
        target_circle = plt.Circle(scen["target_pos"], scen["target_radius"], color="green", alpha=0.3)
        ax_traj.add_patch(target_circle)

        # Fixed alpha trajectories as dashed lines
        for fa in FIXED_ALPHAS:
            r = fixed_results[fa]
            ax_traj.plot(r["traj_x"], r["traj_y"], color=FIXED_ALPHA_COLORS[fa],
                         linewidth=1.5, linestyle="--", alpha=0.6,
                         label=rf"Fixed $\alpha$={fa}")

        # Dynamic trajectory: color-coded by alpha value (like your existing eval plots)
        dyn_x = dyn["traj_x"]
        dyn_y = dyn["traj_y"]
        ax_traj.plot(dyn_x, dyn_y, color="gray", linestyle="--", alpha=0.3)
        sc = ax_traj.scatter(dyn_x[:-1], dyn_y[:-1], c=dyn["alphas"], cmap="coolwarm",
                             vmin=0.1, vmax=5.0, s=25, zorder=5, label=rf"Dynamic $\alpha$ (RL)")
        cbar = plt.colorbar(sc, ax=ax_traj, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$ Value")

        # Timestep markers every 10 steps
        time_interval = 10
        for t in range(0, len(dyn_x) - 1, time_interval):
            ax_traj.plot(dyn_x[t], dyn_y[t], marker="s", color="black", markersize=4, zorder=6)
            ax_traj.text(dyn_x[t], dyn_y[t] + 0.3, f"t={t}", fontsize=7, color="black", ha="center", zorder=7)

        ax_traj.set_title(f"{scen['name']} | Steps: {dyn['steps']} | Reward: {dyn['total_reward']:.1f}", fontsize=11)
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.set_xlim(-1, 11)
        ax_traj.set_ylim(-5, 5)
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.grid(True, alpha=0.3)
        if i == 0:
            ax_traj.legend(loc="upper left", fontsize=7)

        # --- Right: bar chart ---
        labels = [rf"$\alpha$={a}" for a in FIXED_ALPHAS] + ["Dynamic"]
        rewards = [fixed_results[a]["total_reward"] for a in FIXED_ALPHAS] + [dyn["total_reward"]]
        min_dists = [fixed_results[a]["min_dist"] for a in FIXED_ALPHAS] + [dyn["min_dist"]]
        steps = [fixed_results[a]["steps"] for a in FIXED_ALPHAS] + [dyn["steps"]]
        bar_colors = [FIXED_ALPHA_COLORS[a] for a in FIXED_ALPHAS] + [DYNAMIC_COLOR]

        x = np.arange(len(labels))
        width = 0.35

        ax_metrics.bar(x - width / 2, rewards, width, color=bar_colors, alpha=0.7, label="Total Reward")
        ax_metrics.set_ylabel("Total Reward")
        ax_metrics.set_title(f"{scen['name']} - Metrics", fontsize=13)
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(labels, fontsize=9)

        ax_dist = ax_metrics.twinx()
        ax_dist.bar(x + width / 2, min_dists, width, color=bar_colors, alpha=0.4,
                     edgecolor="black", linestyle="--", label="Min Dist to Obs")
        ax_dist.set_ylabel("Min Distance to Obstacle (m)")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":")

        # Reached/collided markers
        all_results = [fixed_results[a] for a in FIXED_ALPHAS] + [dyn]
        for j, r in enumerate(all_results):
            marker = "O" if r["reached_target"] else "X"
            color = "green" if r["reached_target"] else ("red" if r["collided"] else "gray")
            ax_metrics.text(j, ax_metrics.get_ylim()[1] * 0.95, marker,
                            ha="center", fontsize=12, fontweight="bold", color=color)

        if i == 0:
            ax_metrics.legend(loc="upper left", fontsize=7)
            ax_dist.legend(loc="upper right", fontsize=7)
        ax_metrics.grid(True, alpha=0.3, axis="y")

        # --- Third column: alpha vs distance over time (dynamic only) ---
        ax_corr.set_title(f"{scen['name']} - " + r"Dynamic $\alpha$ vs Distance", fontsize=13)
        ax_corr.set_xlabel("Time Step")

        # Plot alpha on left y-axis
        ax_corr.set_ylabel(r"$\alpha$ Value", color="purple")
        ax_corr.plot(range(len(dyn["alphas"])), dyn["alphas"], color="purple", linewidth=2, label=r"$\alpha$")
        ax_corr.tick_params(axis="y", labelcolor="purple")
        ax_corr.set_ylim(0, 5.5)

        # Plot distance on right y-axis
        ax_corr_dist = ax_corr.twinx()
        ax_corr_dist.set_ylabel("Distance to Obstacle (m)", color="darkorange")
        ax_corr_dist.plot(range(len(dyn["distances"])), dyn["distances"], color="darkorange",
                          linewidth=2, linestyle="-.", label="Dist to Obs")
        ax_corr_dist.tick_params(axis="y", labelcolor="darkorange")
        ax_corr_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        # Show where fixed alphas would sit as horizontal reference lines
        for fa in FIXED_ALPHAS:
            ax_corr.axhline(fa, color=FIXED_ALPHA_COLORS[fa], linewidth=1, linestyle=":", alpha=0.4)

        if i == 0:
            lines1, labels1 = ax_corr.get_legend_handles_labels()
            lines2, labels2 = ax_corr_dist.get_legend_handles_labels()
            ax_corr.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
        ax_corr.grid(True, alpha=0.3)

        # Collect summary
        for alpha in FIXED_ALPHAS:
            r = fixed_results[alpha]
            summary_rows.append((scen["name"], f"Fixed a={alpha}", r["total_reward"], r["min_dist"], r["reached_target"], r["collided"], r["steps"]))
        summary_rows.append((scen["name"], "Dynamic", dyn["total_reward"], dyn["min_dist"], dyn["reached_target"], dyn["collided"], dyn["steps"]))

    plt.tight_layout()
    save_path = os.path.join(save_dir, "ablation_proper.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # --- Summary table ---
    print(f"\n{'='*95}")
    print("ABLATION SUMMARY (per scenario)")
    print(f"{'='*95}")
    print(f"{'Scenario':<22} {'Method':<16} {'Reward':>10} {'MinDist':>10} {'Steps':>8} {'Reached':>9} {'Collided':>9}")
    print(f"{'-'*95}")
    for row in summary_rows:
        scen_name, method, reward, min_dist, reached, collided, steps = row
        reached_str = 'Yes' if reached else 'No'
        collided_str = 'YES' if collided else 'No'
        print(f"{scen_name:<22} {method:<16} {reward:>10.1f} {min_dist:>10.3f} {steps:>8} {reached_str:>9} {collided_str:>9}")

    # Aggregate
    print(f"\n{'='*95}")
    print("AGGREGATE (averaged across 5 scenarios)")
    print(f"{'='*95}")
    print(f"{'Method':<16} {'Avg Reward':>12} {'Avg MinDist':>12} {'Avg Steps':>10} {'Reached':>10} {'Collided':>10}")
    print(f"{'-'*95}")

    for method_label in [f"Fixed a={a}" for a in FIXED_ALPHAS] + ["Dynamic"]:
        rows = [r for r in summary_rows if r[1] == method_label]
        avg_reward = np.mean([r[2] for r in rows])
        avg_min_dist = np.mean([r[3] for r in rows])
        avg_steps = np.mean([r[6] for r in rows])
        total_reached = sum(r[4] for r in rows)
        total_collided = sum(r[5] for r in rows)
        print(f"{method_label:<16} {avg_reward:>12.1f} {avg_min_dist:>12.3f} {avg_steps:>10.1f} {total_reached:>8}/5 {total_collided:>8}/5")

    print(f"\nDone!")
