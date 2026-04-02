"""Exp 62 Phase 2: Train safety filter with moving obstacles."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_safety_moving import SafetyMovingObsEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 62 Phase 2: Safety filter with moving obstacles (alpha, phi)")
    print("  Frozen nav policy, SHIELD SDF CBF, collision -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: SafetyMovingObsEnv(
        nav_model_path="./models_dynamic/nav_moving_model_v2"
    ) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/safety_filter_model")
    vec_env.close()
    print(f"\n>> Phase 2 (safety moving): {elapsed:.1f}s ({elapsed/60:.1f}min)")
