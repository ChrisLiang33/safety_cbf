"""Exp 58 Phase 1: Train easy navigation (no obstacles)."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_nav_easy import NavEasyEnv

TOTAL_TIMESTEPS = 5_000_000  # fewer steps needed — no obstacles to learn about
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 58 Phase 1: Train easy navigation (kx, ky)")
    print("  NO obstacles — learn to beeline to target")
    print("  Max speed: 6 m/s, target at x=150")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: NavEasyEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/nav_easy_model")
    vec_env.close()
    print(f"\n>> Phase 1 (easy nav): {elapsed:.1f}s ({elapsed/60:.1f}min)")
