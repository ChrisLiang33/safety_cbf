"""Train phi-CBF + tight obstacles model: agent controls [kx, ky, alpha, phi]."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import PhiCBFTightObsEnv

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training phi-CBF + TIGHT obstacles model (kx, ky, alpha, phi)...")
    print("  ISSf-CBF: Lgh @ u >= -alpha * h(x) + (||Lgh||^2 * phi) / h(x)")
    print("  Domain randomization: noise scale ~ U(0.0, 0.5) per episode")

    env_fns = [lambda: PhiCBFTightObsEnv(disturbance_range=(0.0, 0.5)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_2000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Phi-CBF+high-noise model: {elapsed:.1f}s ({elapsed/60:.1f}min) — saved to {save_path}")
