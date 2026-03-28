"""Level 4: Train ISSf-CBF [kx, ky, alpha, phi] with disturbance + radius noise."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import Level4ISSfCBFEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Level 4: Training ISSf-CBF (kx, ky, alpha, phi)...")
    print("  ISSf-CBF: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)")
    print("  Constant bias ~ U(0.0, 1.0), radius error ~ U(-1.0, +1.0)")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps")

    env_fns = [lambda: Level4ISSfCBFEnv(
        radius_error_range=(-1.0, 1.0),
        bias_magnitude_range=(0.0, 1.0),
    ) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Level 4 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
