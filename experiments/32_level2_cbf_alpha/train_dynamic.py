"""Level 2: Train CBF + alpha model [kx, ky, alpha]. No disturbance, no radius noise."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import Level2CBFAlphaEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Level 2: Training CBF + alpha (kx, ky, alpha)...")
    print("  Standard CBF: Lgh @ u >= -alpha * h(x)")
    print("  No disturbance, no radius noise")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")

    env_fns = [lambda: Level2CBFAlphaEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Level 2 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
