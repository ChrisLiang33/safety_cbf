"""Train alpha-only model on large map with big obstacle squeeze gate."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import AlphaOnlyLongSqueezeDynamicEnv

TOTAL_TIMESTEPS = 1_500_000  # more steps for longer episodes
N_ENVS = 8

if __name__ == "__main__":
    print("Training alpha-only long squeeze model (100m map, r=5 obstacles)...")

    env_fns = [lambda: AlphaOnlyLongSqueezeDynamicEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_1500k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Long squeeze model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
