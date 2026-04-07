"""Exp 69: Discrete A* + moving obstacles, 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import DiscreteAStarMovingObsEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 69: Discrete A* + moving obstacles (alpha/phi step=0.2)")
    print("  9 discrete actions: 3 alpha x 3 phi (down/stay/up)")
    print("  Obstacles move at 0.3-1.0 m/s, bounce off boundaries")
    print("  A* replans every 10 steps")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: DiscreteAStarMovingObsEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/dynamic_10000k_model")
    vec_env.close()
    print(f"\n>> Exp 69: {elapsed:.1f}s ({elapsed/60:.1f}min)")
