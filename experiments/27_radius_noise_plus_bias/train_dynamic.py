"""Train ISSf-CBF model with radius noise + constant bias + randomized obstacles."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import PhiCBFRadiusNoisePlusBiasEnv

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training ISSf-CBF + RADIUS NOISE + BIAS + RANDOMIZED OBSTACLES (kx, ky, alpha, phi)...")
    print("  ISSf-CBF: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)")
    print("  Radius error ~ U(-1.0, +1.0) per obstacle per episode")
    print("  Constant bias magnitude ~ U(0.0, 1.0), random direction per episode")
    print("  Obstacles: fully randomized placement (no slalom)")

    env_fns = [lambda: PhiCBFRadiusNoisePlusBiasEnv(
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

    print(f"\n>> ISSf-CBF model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
