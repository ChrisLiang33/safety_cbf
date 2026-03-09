import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

def solve_strict_cbf(robot_pos, obs_pos, radius, k_nom, alpha):
    """
    The strict CBF filter requested by your advisor (no epsilon term).
    """
    # 1. Calculate distances
    x_diff = robot_pos - obs_pos
    h_x = np.sum(x_diff**2) - (radius**2)
    L_g_h = 2 * x_diff
    
    # 2. Setup CVXPY
    u = cp.Variable(2)
    cost = cp.Minimize(0.5 * cp.sum_squares(u - k_nom))
    
    # 3. The Strict Constraint: L_g_h * u >= -alpha * h_x
    constraints = [
        L_g_h @ u >= -alpha * h_x,
        u >= np.array([-2.0, -2.0]), # Hardware velocity limits
        u <= np.array([2.0, 2.0])
    ]
    
    prob = cp.Problem(cost, constraints)
    
    # 4. Solve safely
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if u.value is not None:
            return u.value
    except Exception:
        pass
    
    # Fallback to zero velocity if the solver crashes
    return np.array([0.0, 0.0])

if __name__ == "__main__":
    print("Loading Trained Pure RL Driver...")
    # Load the model you saved in Step 1
    model = PPO.load("pure_rl_driver_model")
    obs_pos = np.array([4.0, 0.1]) 
    target_pos = np.array([9.0, 0.0]) # ADD THIS LINE
    
    # The parameters for our Ablation Study
    alphas_to_test = [0.1, 0.5, 1.0, 5.0]
    colors = ['orange', 'green', 'blue', 'purple']
    
    dt = 0.1
    obs_radius = 1.0
    
    # Place the obstacle directly in the robot's path to force a dodge
    obs_pos = np.array([4.0, 0.1]) 
    
    plt.figure(figsize=(12, 6))
    
    print("Running alpha sweep simulations...")
    for alpha, color in zip(alphas_to_test, colors):
        robot_pos = np.array([0.0, 0.0])
        traj_x, traj_y = [robot_pos[0]], [robot_pos[1]]
        
        # Simulate for 100 timesteps
        for step in range(100):
            # 1. Ask the AI what it wants to do (k_nom)
            # Ask the AI what it wants to do (k_nom)
            obs_array = np.array([
                float(robot_pos[0]), float(robot_pos[1]),
                float(obs_pos[0]), float(obs_pos[1]), float(obs_radius),
                float(target_pos[0]), float(target_pos[1]), float(target_raduis)
            ], dtype=np.float32)
            
            action, _ = model.predict(obs_array, deterministic=True)
            k_nom = np.array([action[0], action[1]])
            
            # 2. Pass k_nom through the strict CBF math
            u_safe = solve_strict_cbf(robot_pos, obs_pos, obs_radius, k_nom, alpha)
            
            # 3. Move the robot using the filtered velocity
            robot_pos += u_safe * dt
            
            traj_x.append(robot_pos[0])
            traj_y.append(robot_pos[1])
            
            # Stop early if we crossed the finish line
            if robot_pos[0] > 9.0:
                break
                
        plt.plot(traj_x, traj_y, color=color, linewidth=2, label=f"Alpha = {alpha}")

    # --- Plotting the Environment ---
    print("Generating Ablation Study Graph...")
    circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='red', alpha=0.4, label='Obstacle')
    plt.gca().add_patch(circle)
    
    # Pure RL (No CBF) Baseline
    plt.plot([0, 9], [0, 0], color='black', linestyle='--', alpha=0.3, label='Requested Trajectory (Crash)')

    plt.title(r"Control Barrier Function Ablation Study: Alpha ($\alpha$) Sweep", fontsize=16)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(-1, 10)
    plt.ylim(-4, 4)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    eval_dir = "./eval_plots/"
    os.makedirs(eval_dir, exist_ok=True)
    plt.savefig(os.path.join(eval_dir, "alpha_sweep_ablation.png"), bbox_inches='tight')
    plt.show()
    print("Done! Graph saved to eval_plots/alpha_sweep_ablation.png")