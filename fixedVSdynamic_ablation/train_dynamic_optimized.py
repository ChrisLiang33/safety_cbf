"""
Train dynamic alpha PPO models using the optimized parametric QP env.
Logs training time so you can compare against original training speed.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import shutil
import time

from env_dynamic_optimized import AdaptiveCBFEnvOptimized

TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    model_dir = "./models_dynamic_optimized/"
    os.makedirs(model_dir, exist_ok=True)

    log_dir = "./logs_dynamic_optimized/"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    vec_env = make_vec_env(
        AdaptiveCBFEnvOptimized,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        monitor_dir=log_dir
    )

    print(f"Training dynamic alpha (optimized) for {TOTAL_TIMESTEPS} timesteps...")
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = os.path.join(model_dir, f"dynamic_optimized_{TOTAL_TIMESTEPS}_model")
    model.save(save_path)
    print(f"\nSaved: {save_path}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
