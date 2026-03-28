"""Train alpha-only model (no phi): agent controls [kx, ky, alpha] with radius estimation noise."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_alpha_only import AlphaOnlyRadiusNoiseEnv

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training alpha-only + RADIUS ESTIMATION NOISE model (kx, ky, alpha)...")
    print("  Standard CBF: Lgh @ u >= -alpha * h(x)")
    print("  Radius error: per-obstacle, sampled ONCE per episode ~ U(-1.0, +1.0)")
    print("  TRUE radius=5.0 for collision, ESTIMATED radius for CBF")

    env_fns = [lambda: AlphaOnlyRadiusNoiseEnv(radius_error_range=(-1.0, 1.0)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_alpha_only/alpha_only_2000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Alpha-only+radius-noise model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
