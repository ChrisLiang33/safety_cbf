"""Train ISSf-CBF model with adversarial bias + radius noise + randomized obstacles."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import PhiCBFAdversarialBiasEnv

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training ISSf-CBF + ADVERSARIAL BIAS (kx, ky, alpha, phi)...")
    print("  Bias always pushes toward nearest obstacle")
    print("  Bias magnitude ~ U(0.0, 1.0) per episode, direction = adversarial each step")
    print("  Radius error ~ U(-1.0, +1.0), randomized obstacles")

    env_fns = [lambda: PhiCBFAdversarialBiasEnv(
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

    print(f"\n>> Adversarial bias model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
