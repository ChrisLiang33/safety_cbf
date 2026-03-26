"""Train fixed-alpha models for 2-obstacle environment with CBF penalty."""
import time
import numpy as np
from stable_baselines3 import PPO
from env_fixed_alpha import MultiObstacleFixedAlphaEnv

ALPHAS = [0.1, 0.5, 1.0, 5.0]
TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"  Training fixed alpha = {alpha} (CBF penalty)")
        print(f"{'='*60}")

        env_fns = [lambda a=alpha: MultiObstacleFixedAlphaEnv(alpha=a) for _ in range(N_ENVS)]

        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv(env_fns)

        model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

        start = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        elapsed = time.time() - start

        save_path = f"./models_fixed/fixed_alpha_{alpha}_900k_model"
        model.save(save_path)
        vec_env.close()

        print(f"\n>> alpha={alpha}: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
