"""Exp 74 Phase 2: Learn alpha (phi frozen from Phase 1), 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_phase2_alpha import AlphaWithFrozenPhiEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 74 Phase 2: Learn alpha (phi frozen from Phase 1)")
    print("  3 discrete actions: alpha_down / alpha_stay / alpha_up (step=0.5)")
    print("  Phi driven by frozen Phase 1 model")
    print("  A* replans every 30 steps, INFLATION_MARGIN=0.0")
    print("  Obstacles move at 0.3-1.0 m/s")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: AlphaWithFrozenPhiEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/alpha_frozen_phi_model")
    vec_env.close()
    print(f"\n>> Exp 74 Phase 2: {elapsed:.1f}s ({elapsed/60:.1f}min)")
