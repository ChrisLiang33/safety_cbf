"""Exp 74 Phase 1: Learn phi (alpha fixed=2.0), 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_phase1_phi import PhiOnlyEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 74 Phase 1: Learn phi (alpha fixed=2.0)")
    print("  3 discrete actions: phi_down / phi_stay / phi_up (step=0.5)")
    print("  Alpha constant at 2.0 -- phi must handle safety alone")
    print("  A* replans every 30 steps, INFLATION_MARGIN=0.0")
    print("  Obstacles move at 0.3-1.0 m/s")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: PhiOnlyEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/phi_only_model")
    vec_env.close()
    print(f"\n>> Exp 74 Phase 1: {elapsed:.1f}s ({elapsed/60:.1f}min)")
