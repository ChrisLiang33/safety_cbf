"""Train dynamic-alpha model for single obstacle with alpha reward bonus."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic_optimized import AdaptiveCBFEnvOptimized

TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training dynamic-alpha model (1 obstacle + alpha reward)...")

    env_fns = [lambda: AdaptiveCBFEnvOptimized() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic_optimized/dynamic_optimized_900000_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Dynamic model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
