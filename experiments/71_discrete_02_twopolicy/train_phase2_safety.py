"""Exp 71 Phase 2: Train discrete safety filter with frozen nav."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_safety_discrete import SafetyDiscreteMovingObsEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 71 Phase 2: Discrete safety filter (alpha/phi up/stay/down)")
    print("  Frozen nav policy, SHIELD SDF CBF, collision -250")
    print("  ALPHA_STEP=0.2, PHI_STEP=0.2, Discrete(9)")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: SafetyDiscreteMovingObsEnv(
        nav_model_path="./models_dynamic/nav_moving_discrete_model_v2"
    ) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/safety_discrete_model")
    vec_env.close()
    print(f"\n>> Phase 2 (discrete safety): {elapsed:.1f}s ({elapsed/60:.1f}min)")
