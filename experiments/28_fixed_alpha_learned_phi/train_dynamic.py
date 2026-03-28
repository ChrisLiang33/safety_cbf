"""Train FIXED alpha + learned phi model: agent controls [kx, ky, phi]."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import FixedAlphaLearnedPhiEnv, FIXED_ALPHA

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print(f"Training FIXED alpha={FIXED_ALPHA} + LEARNED phi (kx, ky, phi)...")
    print(f"  ISSf-CBF: Lgh @ u >= -{FIXED_ALPHA} * h(x) + (||Lgh||^2 * phi) / h(x)")
    print("  Radius error ~ U(-1.0, +1.0), bias magnitude ~ U(0.0, 1.0)")
    print("  Randomized obstacle placement")

    env_fns = [lambda: FixedAlphaLearnedPhiEnv(
        radius_error_range=(-1.0, 1.0),
        bias_magnitude_range=(0.0, 1.0),
    ) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_2000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Fixed-alpha model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
