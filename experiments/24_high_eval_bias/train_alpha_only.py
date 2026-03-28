"""Train alpha-only model (no phi): agent controls [kx, ky, alpha] with constant bias."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_alpha_only import AlphaOnlyHighEvalBiasEnv

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8

if __name__ == "__main__":
    print("Training alpha-only + CONSTANT BIAS model (kx, ky, alpha)...")
    print("  Standard CBF: Lgh @ u >= -alpha * h(x)")
    print("  Bias: fixed random direction per episode, magnitude ~ U(0.0, 1.0)")
    print("  NOTE: training same as exp 22; eval will use bias magnitude 1.5")

    env_fns = [lambda: AlphaOnlyHighEvalBiasEnv(disturbance_range=(0.0, 1.0)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start

    save_path = "./models_alpha_only/alpha_only_2000k_model"
    model.save(save_path)
    vec_env.close()

    print(f"\n>> Alpha-only+constant-bias model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
