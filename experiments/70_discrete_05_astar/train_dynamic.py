"""Exp 70: Discrete A* + moving obstacles (step=0.5), 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import DiscreteAStarMovingObsEnv05

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 70: Discrete A* + moving obstacles (step=0.5)")
    print("  Discrete(9) action space: 3 alpha x 3 phi")
    print("  ALPHA_STEP=0.5, PHI_STEP=0.5")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: DiscreteAStarMovingObsEnv05() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/dynamic_10000k_model")
    vec_env.close()
    print(f"\n>> Exp 70: {elapsed:.1f}s ({elapsed/60:.1f}min)")
