"""
Train separate PPO models for each fixed alpha value.
Each model only learns [k_x, k_y] with alpha baked into the env.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import shutil
import time

from env_fixed_alpha import FixedAlphaCBFEnv

ALPHAS = [0.1, 0.5, 1.0, 5.0]
TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    model_dir = "./models_fixed/"
    os.makedirs(model_dir, exist_ok=True)

    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"Training fixed alpha = {alpha} for {TOTAL_TIMESTEPS} timesteps...")
        print(f"{'='*60}")

        log_dir = f"./logs_fixed/alpha_{alpha}/"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        vec_env = make_vec_env(
            lambda: FixedAlphaCBFEnv(alpha=alpha),
            n_envs=N_ENVS,
            vec_env_cls=DummyVecEnv,
            monitor_dir=log_dir
        )

        model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")
        start = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        elapsed = time.time() - start

        save_path = os.path.join(model_dir, f"fixed_alpha_{alpha}_900k_model")
        model.save(save_path)
        print(f"Saved: {save_path}")
        print(f"Training time for alpha={alpha}: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print("\nAll fixed-alpha models trained!")
