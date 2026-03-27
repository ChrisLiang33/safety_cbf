"""Train 3-obstacle weave model: agent controls [kx, ky, alpha]."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import ThreeObsWeaveDynamicEnv

TOTAL_TIMESTEPS = 2_000_000  # more steps — 3 obstacles + navigation
N_ENVS = 8

if __name__ == "__main__":
    print("Training 3-obstacle weave model (kx, ky, alpha)...")

    env_fns = [lambda: ThreeObsWeaveDynamicEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_2000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> 3-obs weave model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
