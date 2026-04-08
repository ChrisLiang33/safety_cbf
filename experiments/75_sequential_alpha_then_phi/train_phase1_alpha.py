"""Exp 75, Phase 1: Learn alpha (phi fixed=0.5), 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_phase1_alpha import AlphaOnlyEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 75 Phase 1: Learn alpha (phi fixed=0.5)")
    print("  3 discrete actions: alpha down/stay/up (step=0.5)")
    print("  PHI_FIXED = 0.5, ALPHA_INIT = 2.0")
    print("  A* replans every 30 steps, INFLATION_MARGIN=0.0")
    print("  Obstacles move at 0.3-1.0 m/s")
    print("  SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: AlphaOnlyEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/alpha_only_model")
    vec_env.close()
    print(f"\n>> Exp 75 Phase 1: {elapsed:.1f}s ({elapsed/60:.1f}min)")
