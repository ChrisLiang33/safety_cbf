"""
Exp 45: Generate combined plot with ONLY columns 1 (trajectory) and 2 (alpha+phi time series).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from stable_baselines3 import PPO
import os

from env_dynamic import HigherSpeedEnv
from evaluate_randomized import (
    generate_scenarios, setup_scenario, run_episode,
    plot_trajectory, plot_alpha_phi_time,
    MODEL_PATH, MAX_STEPS
)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_dir = "./plots_randomized/"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading model...")
    env = HigherSpeedEnv()
    model = PPO.load(MODEL_PATH)

    scenarios = generate_scenarios(n=10, seed=42)

    print(f"\nRunning {len(scenarios)} scenarios...")
    all_scenarios = []
    for scen in scenarios:
        result = run_episode(env, model, scen)
        all_scenarios.append({"scen": scen, "result": result})

    # --- COMBINED PLOT: only columns 1 and 2 ---
    n_scen = len(scenarios)
    fig, axs = plt.subplots(n_scen, 2, figsize=(28, 7 * n_scen),
                            gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Exp 45: Trajectory + Alpha/Phi (6 m/s, fully randomized eval)",
                 fontsize=18, y=1.005)

    for i, data in enumerate(all_scenarios):
        scen, r = data["scen"], data["result"]
        plot_trajectory(axs[i, 0], scen, r, fontsize_title=14, fontsize_label=12,
                        fontsize_metrics=9, fontsize_cbar=10, fontsize_bias=10,
                        fontsize_radius=9, fontsize_time=9, markersize=5)
        plot_alpha_phi_time(axs[i, 1], scen, r, fontsize_title=14, fontsize_label=12,
                            fontsize_legend=10)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "combined_traj_alpha_phi.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: plots_randomized/combined_traj_alpha_phi.png")
