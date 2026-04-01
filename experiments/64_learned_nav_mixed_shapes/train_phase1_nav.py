"""Exp 64 Phase 1: Train nav with mixed obstacle shapes."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_nav_shapes import NavMixedShapesEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 64 Phase 1: Nav with mixed shapes (kx, ky)")
    print("  Circles, rectangles, line segments")
    print("  No CBF, no QP")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: NavMixedShapesEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/nav_shapes_model")
    vec_env.close()
    print(f"\n>> Phase 1 (nav shapes): {elapsed:.1f}s ({elapsed/60:.1f}min)")
