"""Exp 37: Train ISSf-CBF with reward bonus for safety margin."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import RewardPhiEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 37: Training ISSf-CBF + REWARD PHI (kx, ky, alpha, phi)...")
    print("  Safety margin bonus when near obstacles")
    print("  Random obs radius U(3,7), bias U(0,1), radius error U(-1,+1)")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")

    env_fns = [lambda: RewardPhiEnv(
        radius_error_range=(-1.0, 1.0),
        bias_magnitude_range=(0.0, 1.0),
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
    print(f"\n>> Exp 37 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
