"""Train alpha-only model with 2-obstacle squeeze gate."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import AlphaOnly2ObsDynamicEnv

TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training alpha-only squeeze gate model...")

    env_fns = [lambda: AlphaOnly2ObsDynamicEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_900k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Squeeze gate model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
