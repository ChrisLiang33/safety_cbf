"""Train navigation baseline: agent controls [kx, ky] only. No CBF."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import NavigationBaselineEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training NAVIGATION BASELINE (kx, ky only, no CBF)...")
    print("  Direct control: robot_pos += [kx, ky] * dt")
    print("  No alpha, no phi, no QP safety filter")
    print("  Gentler randomization: obstacles in x-bands, y randomized")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps")

    env_fns = [lambda: NavigationBaselineEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Navigation baseline: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
