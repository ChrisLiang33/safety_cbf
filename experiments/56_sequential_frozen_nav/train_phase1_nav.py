"""Exp 56 Phase 1: Train navigation-only policy (kx, ky). No CBF."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_nav_only import NavOnlyEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 56 Phase 1: Train navigation only (kx, ky)")
    print("  No CBF, no QP, no alpha, no phi")
    print("  Max speed: 6 m/s, target at x=150, 800 max steps")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: NavOnlyEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/nav_only_model")
    vec_env.close()
    print(f"\n>> Phase 1 (nav): {elapsed:.1f}s ({elapsed/60:.1f}min)")
