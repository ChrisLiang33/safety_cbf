"""Exp 62 Phase 1: Train nav with moving obstacles."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_nav_moving import NavMovingObsEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 62 Phase 1: Nav with moving obstacles (kx, ky)")
    print("  Obstacles move at 0.3-1.0 m/s")
    print("  No CBF, no QP")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: NavMovingObsEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/nav_moving_model")
    vec_env.close()
    print(f"\n>> Phase 1 (nav moving): {elapsed:.1f}s ({elapsed/60:.1f}min)")
