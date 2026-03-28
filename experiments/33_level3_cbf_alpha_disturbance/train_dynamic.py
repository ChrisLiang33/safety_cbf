"""Level 3: Train CBF + alpha + constant bias [kx, ky, alpha]. No radius noise."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import Level3CBFDisturbanceEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Level 3: Training CBF + alpha + CONSTANT BIAS (kx, ky, alpha)...")
    print("  Standard CBF: Lgh @ u >= -alpha * h(x)")
    print("  Constant bias ~ U(0.0, 1.0) per episode")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps")

    env_fns = [lambda: Level3CBFDisturbanceEnv(bias_magnitude_range=(0.0, 1.0)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Level 3 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
