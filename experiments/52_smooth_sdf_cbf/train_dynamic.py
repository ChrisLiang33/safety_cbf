"""Exp 52: Smooth SDF-based CBF, 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import SmoothSDFCBFEnv

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 52: Smooth SDF-based CBF (kx, ky, alpha, phi)")
    print("  h = lambda*(1 - e^{-gamma*sdf}), lambda=1.0, gamma=0.5")
    print("  Max speed: 6 m/s, target at x=150, 800 max steps")
    print("  Systematic bias U(-0.6,0.4), jitter U(-0.2,0.2)")
    print("  Dynamics bias U(0.3,1.0), obs radius U(3,7)")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: SmoothSDFCBFEnv() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/dynamic_10000k_model")
    vec_env.close()
    print(f"\n>> Exp 52: {elapsed:.1f}s ({elapsed/60:.1f}min)")
