from env import AdaptiveCBFEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

if __name__ == "__main__":
    log_dir = "./cbf_logs/"
    
    if os.path.exists(log_dir):
        print(f"Clearing old logs from {log_dir}...")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    n_envs = 8
    print(f"Spinning up {n_envs} parallel environments...")

    # Create the parallel vectorized environment
    vec_env = make_vec_env(
        AdaptiveCBFEnv, 
        n_envs=n_envs, 
        vec_env_cls=DummyVecEnv, 
        monitor_dir=log_dir
    )

    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")  

    print("Starting Training...")
    total_timesteps = 150000
    model.learn(total_timesteps=total_timesteps) 
    
    os.makedirs("./model/", exist_ok=True)
    print("Training finished! Saving model...")
    model.save(f"./model/{total_timesteps}_model")

    print("Plotting Learning Curve...")
    dataframes = []
    
    for file in os.listdir(log_dir):
        if file.endswith("monitor.csv"):
            df_part = pd.read_csv(os.path.join(log_dir, file), skiprows=1)
            df_part['timestep'] = df_part['l'].cumsum()
            dataframes.append(df_part)
            
    if dataframes:
        df = pd.concat(dataframes)
        df = df.sort_values(by='timestep').reset_index(drop=True)
        
        plt.figure(figsize=(10, 4))
        
        # Lowered window to 100 since we dropped to 150k timesteps
        plt.plot(df['timestep'], df['r'].rolling(window=100).mean(), label='Rolling Average Reward', color='green')
        
        plt.title("RL Training: Chronological Reward Learning Curve")
        plt.xlabel("Total Training Timesteps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        
        eval_dir = "./eval_plots/"
        os.makedirs(eval_dir, exist_ok=True)
        save_path = os.path.join(eval_dir, "reward_learning_curve.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Reward plot saved to {save_path}")
        
        plt.show()