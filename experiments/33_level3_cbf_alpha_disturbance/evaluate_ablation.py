"""
Level 3: CBF with learnable alpha + constant bias disturbance.
Action is [kx, ky, alpha]. QP enforces safety.
Constant bias breaks the CBF model assumption (x_dot = u, but reality is x_dot = u + bias).

Outputs:
  plots/combined_scenarios.png  (5-col: traj | alpha+obszone | speed | alpha vs dist | policy map)
  plots/aggregate_policy_map.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
import os

from env_dynamic import Level3CBFDisturbanceEnv

# --- CONFIG ---
MODEL_PATH = "./models_dynamic/dynamic_5000k_model"
MAX_STEPS = 600
OBS_RADIUS = 5.0
EVAL_BIAS_MAGNITUDE = 0.7

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


def setup_scenario(env, scen, bias=None):
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
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_episode(env, model, scen, bias=None):
    obs = setup_scenario(env, scen, bias=bias)
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



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = Level3CBFDisturbanceEnv()
    model = PPO.load(MODEL_PATH)

    # Hand-picked bias: upward (pi/2)
    hand_picked_bias = EVAL_BIAS_MAGNITUDE * np.array([np.cos(np.pi / 2), np.sin(np.pi / 2)])

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        result = run_episode(env, model, scen, bias=hand_picked_bias)
        all_scenarios.append({"scen": scen, "result": result})

    # =====================================================================
    # COMBINED PLOT: 5 columns — Trajectory | Alpha+ObsZone | Speed | Alpha vs Dist | Policy Map
    # =====================================================================
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 5, figsize=(45, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1, 1, 1, 1]})
    fig.suptitle("LEVEL 3: CBF + Alpha + Disturbance (kx, ky, alpha) -- bias=0.7 upward",
                 fontsize=18, y=1.005)

    # Collect all alpha-vs-dist data for policy map
    all_alpha_dist_pairs = []

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]

        # --- Column 1: Trajectory with bias arrow ---
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

        # Bias arrow annotation
        ax.annotate("", xy=(5, 12), xytext=(5, 8),
                     arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
        ax.text(5, 13, f"bias={EVAL_BIAS_MAGNITUDE:.1f}", fontsize=8,
                color="purple", ha="center", fontweight="bold")

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

        # Collect for policy map
        for t in range(len(r["alphas"])):
            d = min(r["per_obs_dists"][oi][t] for oi in range(3))
            all_alpha_dist_pairs.append((d, r["alphas"][t]))

        # --- Column 5: Policy Map (alpha vs distance scatter, all scenarios combined) ---
        ax = axs[i, 4]
        ax.scatter(min_dists_at_step, r["alphas"], c=r["alphas"],
                   cmap="coolwarm", vmin=0.1, vmax=5.0, s=10, alpha=0.5)
        ax.axvline(0, color="red", linewidth=1.5, linestyle=":", alpha=0.7)
        ax.set_title(f"{scen['name']} -- Policy Map", fontsize=11)
        ax.set_xlabel("Min Distance to Obstacle Surface (m)")
        ax.set_ylabel("Alpha")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_scenarios.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/combined_scenarios.png")

    # =====================================================================
    # AGGREGATE POLICY MAP: Alpha vs Distance from ALL hand-picked scenarios
    # =====================================================================
    fig_agg, ax_agg = plt.subplots(figsize=(10, 8))
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, len(all_scenarios)))

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        ax_agg.scatter(
            r["dist"], r["alphas"],
            color=scenario_colors[i], s=12, alpha=0.6, label=scen["name"],
        )

    ax_agg.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Collision boundary")
    ax_agg.set_xlabel("Min Distance to Obstacle Surface (m)", fontsize=12)
    ax_agg.set_ylabel("Alpha Value", fontsize=12)
    ax_agg.set_title(
        "Level 3 (CBF + alpha + bias): Alpha vs Distance to Nearest Obstacle (all scenarios)",
        fontsize=13,
    )
    ax_agg.legend(fontsize=9, loc="best")
    ax_agg.grid(True, alpha=0.3)
    fig_agg.tight_layout()
    fig_agg.savefig(os.path.join(save_dir, "aggregate_policy_map.png"), bbox_inches="tight", dpi=150)
    plt.close(fig_agg)
    print("Saved: plots/aggregate_policy_map.png")

    # =====================================================================
    # Console summary table
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"HAND-PICKED SCENARIOS (bias={EVAL_BIAS_MAGNITUDE}, upward)")
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

    print(f"\nDone! Plots saved to {save_dir}")
