"""Exp 65: A* (stochastic) + tighter mixed obstacle shapes, 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import AStarTightMixedShapesEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 65: Stochastic A* + tighter mixed shapes (alpha, phi)")
    print("  Obstacles: 5 tightly spaced circles, rectangles, line segments")
    print("  y-range [-6, 6], 5 x-bands with narrow gaps")
    print("  A* with waypoint perturbation (noise=2.0m)")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: AStarTightMixedShapesEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/dynamic_10000k_model")
    vec_env.close()
    print(f"\n>> Exp 65: {elapsed:.1f}s ({elapsed/60:.1f}min)")
