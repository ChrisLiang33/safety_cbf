"""Exp 54: SHIELD SDF + collision -250, 10M steps."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env_dynamic import ShieldCollision250Env

TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 16

if __name__ == "__main__":
    print("Exp 54: SHIELD SDF + collision -250 (kx, ky, alpha, phi)")
    print("  SHIELD pipeline Eq 19-22, single constraint")
    print("  Max speed: 6 m/s, target at x=150, 800 max steps")
    print("  Collision penalty: -250")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")
    env_fns = [lambda: ShieldCollision250Env() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)
    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.time() - start
    model.save("./models_dynamic/dynamic_10000k_model")
    vec_env.close()
    print(f"\n>> Exp 54: {elapsed:.1f}s ({elapsed/60:.1f}min)")
