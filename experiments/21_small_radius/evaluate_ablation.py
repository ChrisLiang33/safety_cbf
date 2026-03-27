"""
Phi-CBF + SMALL Radius ablation:
  Dynamic [kx,ky,alpha,phi] (ISSf-CBF) vs Dynamic [kx,ky,alpha] (standard CBF)
Both trained and evaluated with high noise (eval=0.3).

Goal: show that phi provides genuine safety value under high disturbance,
not just a comfort margin.

Outputs:
  plots/combined_scenarios.png  (7-col: traj | alpha+phi+obs_zone | speed | alpha+dist | phi+dist | speed+alpha+phi | policy map)
  plots/aggregate_metrics.png
"""
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from env_dynamic import PhiCBFSmallRadiusEnv
from env_alpha_only import AlphaOnlySmallRadiusEnv

# --- CONFIG ---
PHI_MODEL_PATH = "./models_dynamic/dynamic_2000k_model"
ALPHA_ONLY_MODEL_PATH = "./models_alpha_only/alpha_only_2000k_model"
MAX_STEPS = 600
N_RANDOM_SCENARIOS = 100
OBS_RADIUS = 3.0
EVAL_NOISE = 0.3

PHI_COLOR = "black"
ALPHA_ONLY_COLOR = "#2196F3"  # blue

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
    for i in range(3):
        env.obs_pos[i] = scen["obs"][i].copy()
        env.obs_radius[i] = OBS_RADIUS
    env.target_pos = scen["target_pos"].copy()
    env.target_radius = scen["target_radius"]
    env.prev_dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
    env.velocity = np.zeros(2)
    return env._get_obs()


def path_length(traj_x, traj_y):
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    return np.sum(np.sqrt(dx**2 + dy**2))


def run_phi_episode(env, model, scen):
    """Run episode with dynamic [kx, ky, alpha, phi] (ISSf-CBF)."""
    obs = setup_scenario(env, scen)
    env.disturbance_scale = EVAL_NOISE
    traj_x, traj_y, alphas, phis, speeds = [], [], [], [], []
    k_nom_speeds, safe_u_speeds = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    h_vals_list = [[], [], []]
    alpha_h_list = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.obs_radius[i]
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])
            x_diff = env.robot_pos - env.obs_pos[oi]
            h = np.sum(x_diff**2) - env.obs_radius[oi]**2
            h_vals_list[oi].append(h)

        action, _ = model.predict(obs, deterministic=True)
        alpha_val = float(action[2])
        phi_val = float(action[3])
        alphas.append(alpha_val)
        phis.append(phi_val)

        for oi in range(3):
            alpha_h_list[oi].append(alpha_val * h_vals_list[oi][-1])

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
    collided = min(dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "alphas": alphas, "phis": phis,
        "speeds": speeds, "k_nom_speeds": k_nom_speeds,
        "safe_u_speeds": safe_u_speeds, "dist": dist_list,
        "per_obs_dists": per_obs_dists,
        "h_vals": h_vals_list, "alpha_h": alpha_h_list,
        "total_reward": total_reward, "steps": step + 1,
        "reached_target": reached, "collided": collided,
        "min_clearance": min(dist_list), "path_length": plen,
        "path_efficiency": efficiency,
    }


def run_alpha_only_episode(env, model, scen):
    """Run episode with dynamic [kx, ky, alpha] (standard CBF, no phi)."""
    obs = setup_scenario(env, scen)
    env.disturbance_scale = EVAL_NOISE
    traj_x, traj_y, alphas, speeds = [], [], [], []
    k_nom_speeds, safe_u_speeds = [], []
    dist_list = []
    per_obs_dists = [[], [], []]
    total_reward = 0.0

    for step in range(MAX_STEPS):
        traj_x.append(env.robot_pos[0])
        traj_y.append(env.robot_pos[1])
        dists = [np.linalg.norm(env.robot_pos - env.obs_pos[i]) - env.obs_radius[i]
                 for i in range(3)]
        dist_list.append(min(dists))
        for oi in range(3):
            per_obs_dists[oi].append(dists[oi])

        action, _ = model.predict(obs, deterministic=True)
        alpha_val = float(action[2])
        alphas.append(alpha_val)

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
    collided = min(dist_list) < 0
    straight_line = np.linalg.norm(scen["target_pos"] - np.array([0.0, 0.0]))
    plen = path_length(traj_x, traj_y)
    efficiency = plen / straight_line if straight_line > 0 else 1.0

    return {
        "traj_x": traj_x, "traj_y": traj_y, "alphas": alphas,
        "speeds": speeds, "k_nom_speeds": k_nom_speeds,
        "safe_u_speeds": safe_u_speeds, "dist": dist_list,
        "per_obs_dists": per_obs_dists,
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

    print(f"Loading models... (eval noise={EVAL_NOISE})")
    phi_env = PhiCBFSmallRadiusEnv()
    phi_model = PPO.load(PHI_MODEL_PATH)

    ao_env = AlphaOnlySmallRadiusEnv()
    ao_model = PPO.load(ALPHA_ONLY_MODEL_PATH)

    # --- Hand-picked scenarios ---
    print(f"\nRunning {len(SCENARIOS)} hand-picked scenarios...")
    all_scenarios = []
    for scen in SCENARIOS:
        phi_result = run_phi_episode(phi_env, phi_model, scen)
        ao_result = run_alpha_only_episode(ao_env, ao_model, scen)
        all_scenarios.append({"scen": scen, "phi": phi_result, "ao": ao_result})

    # =====================================================================
    # COMBINED PLOT: 7 columns
    # Traj | Alpha+Phi+ObsZone | Speed | Alpha+Dist | Phi+Dist | Speed+Alpha+Phi | Policy Map
    # =====================================================================
    TIME_MARKER_INTERVAL = 50
    OBS_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
    OBS_LABELS = ["Obs 1", "Obs 2", "Obs 3"]
    n_scen = len(SCENARIOS)
    fig, axs = plt.subplots(n_scen, 7, figsize=(63, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1.2, 1, 1, 1, 1, 1]})
    fig.suptitle(rf"SMALL Radius (r=3.0) (noise={EVAL_NOISE}): Dynamic $\alpha$+$\varphi$ (ISSf-CBF) vs Dynamic $\alpha$-only (standard CBF)",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, phi, ao = data["scen"], data["phi"], data["ao"]

        # --- Column 1: Trajectory ---
        ax = axs[i, 0]
        for j in range(3):
            ax.add_patch(plt.Circle(scen["obs"][j], OBS_RADIUS, color="red", alpha=0.2))
        ax.add_patch(plt.Circle(scen["target_pos"], scen["target_radius"],
                                color="green", alpha=0.3))

        # Alpha-only trajectory
        ax.plot(ao["traj_x"], ao["traj_y"], color=ALPHA_ONLY_COLOR,
                linewidth=2, linestyle="--", alpha=0.7, label=r"$\alpha$-only (no $\varphi$)")

        # Phi trajectory with alpha colormap
        phi_x, phi_y = phi["traj_x"], phi["traj_y"]
        step_skip = 2
        sc = ax.scatter(phi_x[:-1:step_skip], phi_y[:-1:step_skip],
                        c=phi["alphas"][::step_skip], cmap="coolwarm",
                        vmin=0.1, vmax=5.0, s=14, zorder=5, label=r"$\alpha$+$\varphi$ (ISSf)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$")

        for t in range(0, len(phi_x) - 1, TIME_MARKER_INTERVAL):
            ax.plot(phi_x[t], phi_y[t], marker='s', color='black', markersize=4, zorder=6)
            ax.text(phi_x[t], phi_y[t] + 0.8, f"t={t}", fontsize=7, color='black',
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

        # Collision/success annotations
        phi_status = ("REACHED" if phi["reached_target"] else "FAIL") + (" COLLISION" if phi["collided"] else "")
        ao_status = ("REACHED" if ao["reached_target"] else "FAIL") + (" COLLISION" if ao["collided"] else "")
        metrics_lines = [
            f"{'Method':<16} {'Steps':>5} {'Reward':>7} {'Status'}",
            "-" * 42,
            f"{'a-only':<16} {ao['steps']:>5} {ao['total_reward']:>7.0f} {ao_status}",
            f"{'a+phi':<16} {phi['steps']:>5} {phi['total_reward']:>7.0f} {phi_status}",
        ]
        metrics_text = "\n".join(metrics_lines)
        ax.text(0.02, -0.15, metrics_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

        # --- Column 2: Alpha & Phi with Obstacle Zones ---
        ax = axs[i, 1]
        ax_phi2 = ax.twinx()

        # Obstacle zone shading
        for oi in range(3):
            d = phi["per_obs_dists"][oi]
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.08)
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            ax.text(t_min, 5.3, f"{OBS_LABELS[oi]}", fontsize=6,
                    color=OBS_COLORS[oi], ha="center", va="top", fontweight="bold")

        # Alpha on left axis
        ax.plot(phi["alphas"], color="purple", linewidth=2.5, label=r"$\alpha$", zorder=5)
        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        # Phi on right axis
        ax_phi2.plot(phi["phis"], color="darkorange", linewidth=2.5, linestyle="-.",
                     label=r"$\varphi$", zorder=5)
        ax_phi2.set_ylabel(r"$\varphi$ Value", color="darkorange")
        ax_phi2.tick_params(axis="y", labelcolor="darkorange")
        ax_phi2.set_ylim(-0.5, 10.5)

        ax.set_title(f"{scen['name']} — Alpha & Phi (Obs Zones)", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_phi2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 3: Speed ---
        ax = axs[i, 2]
        ax.plot(ao["speeds"], color=ALPHA_ONLY_COLOR,
                linewidth=1.5, linestyle="--", alpha=0.7, label=r"$\alpha$-only")
        ax.plot(phi["speeds"], color=PHI_COLOR, linewidth=2, label=r"$\alpha$+$\varphi$")
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4, label="Max (3 m/s)")
        ax.set_title(f"{scen['name']} — Robot Speed", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(-0.1, 4.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)

        # --- Column 4: Alpha + Per-Obstacle Distance ---
        ax = axs[i, 3]
        ax.set_ylabel(r"$\alpha$ Value", color="purple")
        ax.plot(phi["alphas"], color="purple", linewidth=2.5, label=r"$\alpha$ (ISSf)", zorder=5)
        ax.plot(ao["alphas"], color=ALPHA_ONLY_COLOR, linewidth=1.5, linestyle="--",
                alpha=0.7, label=r"$\alpha$ (std CBF)", zorder=4)
        ax.tick_params(axis="y", labelcolor="purple")
        ax.set_ylim(0, 5.5)

        ax_dist = ax.twinx()
        ax_dist.set_ylabel("Dist to Obs (m)", color="gray")
        ax_dist.tick_params(axis="y", labelcolor="gray")
        ax_dist.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for oi in range(3):
            d = phi["per_obs_dists"][oi]
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

        ax.set_title(f"{scen['name']} — Alpha vs Obstacle Proximity", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 5: Phi + Per-Obstacle Distance ---
        ax = axs[i, 4]
        ax.set_ylabel(r"$\varphi$ Value", color="darkorange")
        ax.plot(phi["phis"], color="darkorange", linewidth=2.5, label=r"$\varphi$", zorder=5)
        ax.tick_params(axis="y", labelcolor="darkorange")
        ax.set_ylim(-0.7, 10.5)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

        ax_dist2 = ax.twinx()
        ax_dist2.set_ylabel("Dist to Obs (m)", color="gray")
        ax_dist2.tick_params(axis="y", labelcolor="gray")
        ax_dist2.axhline(0, color="red", linewidth=1, linestyle=":", alpha=0.5)

        for oi in range(3):
            d = phi["per_obs_dists"][oi]
            ax_dist2.plot(d, color=OBS_COLORS[oi], linewidth=1.2, linestyle="-.",
                          alpha=0.7, label=OBS_LABELS[oi])
            t_min = int(np.argmin(d))
            ax.axvline(t_min, color=OBS_COLORS[oi], linewidth=1.5, linestyle="--", alpha=0.6)
            ax.text(t_min, 10.0, f"{OBS_LABELS[oi]}\nt={t_min}", fontsize=6,
                    color=OBS_COLORS[oi], ha="center", va="top", fontweight="bold")
            in_zone = np.array(d) < 10.0
            for t_idx in range(len(d)):
                if in_zone[t_idx]:
                    ax.axvspan(t_idx, t_idx + 1, color=OBS_COLORS[oi], alpha=0.04)

        ax.set_title(f"{scen['name']} — Phi vs Obstacle Proximity", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_dist2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 6: Speed + Alpha + Phi overlay ---
        ax = axs[i, 5]
        ax.set_ylabel("Speed (m/s)", color="teal")
        ax.plot(phi["safe_u_speeds"], color="teal", linewidth=2, label=r"$\|u_{safe}\|$")
        ax.plot(phi["k_nom_speeds"], color="teal", linewidth=1, linestyle=":",
                alpha=0.5, label=r"$\|k_{nom}\|$")
        ax.axhline(3.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.tick_params(axis="y", labelcolor="teal")
        ax.set_ylim(-0.1, 5.0)

        ax_ap = ax.twinx()
        ax_ap.set_ylabel(r"$\alpha$ / $\varphi$")
        ax_ap.plot(phi["alphas"], color="purple", linewidth=1.5, linestyle="--",
                   alpha=0.7, label=r"$\alpha$")
        ax_ap.plot(phi["phis"], color="darkorange", linewidth=1.5, linestyle="-.",
                   alpha=0.7, label=r"$\varphi$")
        ax_ap.set_ylim(-1, 10.5)

        for t_idx in range(len(phi["dist"])):
            if phi["dist"][t_idx] < 8.0:
                ax.axvspan(t_idx, t_idx + 1, color="red", alpha=0.05)

        ax.set_title(f"{scen['name']} — Speed + Alpha + Phi", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_ap.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

        # --- Column 7: Policy Map (alpha & phi vs distance) ---
        ax = axs[i, 6]
        min_dists_per_t = np.array(phi["dist"])
        alphas_arr = np.array(phi["alphas"])
        phis_arr = np.array(phi["phis"])

        ax.scatter(min_dists_per_t, alphas_arr,
                   c="purple", s=10, alpha=0.5, zorder=3, label=r"$\alpha$")
        ax.scatter(min_dists_per_t, phis_arr,
                   c="darkorange", s=10, alpha=0.5, marker="^", zorder=3, label=r"$\varphi$")

        ax.set_xlabel("Min Dist to Obs Surface (m)")
        ax.set_ylabel(r"$\alpha$ / $\varphi$")
        ax.set_ylim(-1, 10.5)
        ax.set_xlim(-1, max(min_dists_per_t) * 1.05)
        ax.axvline(0, color="red", linewidth=1, linestyle=":", alpha=0.5, label="Collision boundary")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.3)
        ax.set_title(f"{scen['name']} — Policy Map (alpha & phi)", fontsize=11)
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

    methods = [r"$\alpha$-only (std CBF)", r"$\alpha$+$\varphi$ (ISSf-CBF)"]
    method_colors = [ALPHA_ONLY_COLOR, PHI_COLOR]
    agg = {m: {"success": 0, "collisions": 0, "min_clearances": [], "efficiencies": [],
               "steps": [], "avg_speeds": []}
           for m in methods}

    for idx, scen in enumerate(random_scenarios):
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{N_RANDOM_SCENARIOS}...")

        phi_r = run_phi_episode(phi_env, phi_model, scen)
        m_phi = methods[1]
        agg[m_phi]["success"] += int(phi_r["reached_target"])
        agg[m_phi]["collisions"] += int(phi_r["collided"])
        agg[m_phi]["min_clearances"].append(phi_r["min_clearance"])
        agg[m_phi]["efficiencies"].append(phi_r["path_efficiency"])
        agg[m_phi]["steps"].append(phi_r["steps"])
        agg[m_phi]["avg_speeds"].append(np.mean(phi_r["speeds"]))

        ao_r = run_alpha_only_episode(ao_env, ao_model, scen)
        m_ao = methods[0]
        agg[m_ao]["success"] += int(ao_r["reached_target"])
        agg[m_ao]["collisions"] += int(ao_r["collided"])
        agg[m_ao]["min_clearances"].append(ao_r["min_clearance"])
        agg[m_ao]["efficiencies"].append(ao_r["path_efficiency"])
        agg[m_ao]["steps"].append(ao_r["steps"])
        agg[m_ao]["avg_speeds"].append(np.mean(ao_r["speeds"]))

    # =====================================================================
    # AGGREGATE METRICS PLOT
    # =====================================================================
    n_methods = len(methods)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(rf"Aggregate — SMALL Radius (r=3.0) (noise={EVAL_NOISE}): $\alpha$+$\varphi$ vs $\alpha$-only ({N_RANDOM_SCENARIOS} Random Scenarios)",
                 fontsize=16)
    x = np.arange(n_methods)

    ax = axs[0, 0]
    vals = [agg[m]["success"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Target Reached")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 1]
    vals = [agg[m]["collisions"] / N_RANDOM_SCENARIOS * 100 for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety Violations")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, max(max(vals) * 1.3, 10))
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axs[0, 2]
    vals = [np.mean(agg[m]["min_clearances"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Min Clearance (m)")
    ax.set_title("Safety Margin")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.axhline(0, color="red", linewidth=1, linestyle=":")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 0]
    vals = [np.mean(agg[m]["efficiencies"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Path Length / Straight-Line")
    ax.set_title("Path Efficiency")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=10)

    ax = axs[1, 1]
    vals = [np.mean(agg[m]["steps"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Steps")
    ax.set_title("Episode Length (lower = faster)")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=10)

    ax = axs[1, 2]
    vals = [np.mean(agg[m]["avg_speeds"]) for m in methods]
    bars = ax.bar(x, vals, color=method_colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Avg Speed (m/s)")
    ax.set_title("Average Robot Speed")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.axhline(3.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "aggregate_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots/aggregate_metrics.png")

    # =====================================================================
    # Console summary
    # =====================================================================
    print(f"\n{'='*100}")
    print(f"HAND-PICKED SCENARIO RESULTS (noise={EVAL_NOISE})")
    print(f"{'='*100}")
    print(f"{'Scenario':<25} {'Method':<20} {'Reached':>8} {'Collided':>9} "
          f"{'MinDist':>9} {'Steps':>7} {'AvgSpd':>8} {'PathEff':>9}")
    print(f"{'-'*100}")

    for data in all_scenarios:
        scen, phi, ao = data["scen"], data["phi"], data["ao"]
        print(f"{scen['name']:<25} {'alpha-only':<20} "
              f"{'Yes' if ao['reached_target'] else 'No':>8} "
              f"{'YES' if ao['collided'] else 'No':>9} "
              f"{ao['min_clearance']:>9.3f} {ao['steps']:>7} "
              f"{np.mean(ao['speeds']):>8.2f} {ao['path_efficiency']:>9.2f}")
        print(f"{scen['name']:<25} {'alpha+phi':<20} "
              f"{'Yes' if phi['reached_target'] else 'No':>8} "
              f"{'YES' if phi['collided'] else 'No':>9} "
              f"{phi['min_clearance']:>9.3f} {phi['steps']:>7} "
              f"{np.mean(phi['speeds']):>8.2f} {phi['path_efficiency']:>9.2f}")
        print()

    print(f"{'='*100}")
    print(f"AGGREGATE ({N_RANDOM_SCENARIOS} RANDOM SCENARIOS)")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'Success':>10} {'Collisions':>11} {'AvgMinDist':>11} "
          f"{'AvgSteps':>9} {'AvgSpeed':>9} {'AvgPathEff':>11}")
    print(f"{'-'*100}")
    for m in methods:
        a = agg[m]
        print(f"{m:<25} {a['success']:>8}/{N_RANDOM_SCENARIOS} "
              f"{a['collisions']:>9}/{N_RANDOM_SCENARIOS} "
              f"{np.mean(a['min_clearances']):>11.3f} {np.mean(a['steps']):>9.1f} "
              f"{np.mean(a['avg_speeds']):>9.2f} {np.mean(a['efficiencies']):>11.2f}")

    print(f"\nDone! Plots saved to {save_dir}")
