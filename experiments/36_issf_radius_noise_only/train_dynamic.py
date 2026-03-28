"""Train ISSf-CBF with radius noise only (no bias): [kx, ky, alpha, phi]."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import ISSfRadiusNoiseOnlyEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 36: Training ISSf-CBF + RADIUS NOISE ONLY (kx, ky, alpha, phi)...")
    print("  ISSf-CBF: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)")
    print("  Radius error ~ U(-1.0, +1.0), NO dynamics disturbance")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")

    env_fns = [lambda: ISSfRadiusNoiseOnlyEnv(radius_error_range=(-1.0, 1.0)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Exp 36 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
