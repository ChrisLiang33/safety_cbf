"""Exp 38: Train ISSf-CBF with harder conditions."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import HarderTrainingEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 38: Training ISSf-CBF + HARDER CONDITIONS (kx, ky, alpha, phi)...")
    print("  Strong bias U(0.5,1.5), always-underestimate radius error U(-2,0)")
    print("  Random obs radius U(3,7)")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")

    env_fns = [lambda: HarderTrainingEnv(
        radius_error_range=(-2.0, 0.0),
        bias_magnitude_range=(0.5, 1.5),
        obs_radius_range=(3.0, 7.0),
    ) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Exp 38 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
