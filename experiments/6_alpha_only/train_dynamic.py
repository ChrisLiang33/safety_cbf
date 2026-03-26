"""Train alpha-only dynamic model. Agent only controls alpha, k_nom is fixed."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import AlphaOnlyDynamicEnv

TOTAL_TIMESTEPS = 900_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training alpha-only dynamic model...")

    env_fns = [lambda: AlphaOnlyDynamicEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_900k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Alpha-only model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
