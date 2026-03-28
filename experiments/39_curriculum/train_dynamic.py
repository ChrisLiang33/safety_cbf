"""Exp 39: Train ISSf-CBF with curriculum learning (3 phases)."""
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_dynamic import CurriculumEnv

TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16

# Phase transitions
PHASE_1_END = 1_000_000   # 0-1M: clean
PHASE_2_END = 3_000_000   # 1M-3M: moderate
# 3M-5M: hard


class CurriculumCallback(BaseCallback):
    """Callback to advance curriculum phase based on timesteps."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_phase = 1

    def _on_step(self) -> bool:
        total = self.num_timesteps
        if total < PHASE_1_END:
            new_phase = 1
        elif total < PHASE_2_END:
            new_phase = 2
        else:
            new_phase = 3

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            # Update all environments
            for env_idx in range(self.training_env.num_envs):
                self.training_env.env_method("set_phase", new_phase, indices=[env_idx])
            print(f"\n>> CURRICULUM: Phase {new_phase} at {total/1e6:.1f}M steps")

        return True


if __name__ == "__main__":
    print("Exp 39: Training ISSf-CBF + CURRICULUM (kx, ky, alpha, phi)...")
    print("  Phase 1 (0-1M): clean — no noise, no bias")
    print("  Phase 2 (1M-3M): moderate — bias U(0,0.5), error U(-0.5,0.5)")
    print("  Phase 3 (3M-5M): hard — bias U(0,1.0), error U(-1.0,1.0)")
    print("  Random obs radius U(3,7) throughout")
    print(f"  {TOTAL_TIMESTEPS/1e6:.0f}M timesteps, {N_ENVS} envs")

    env_fns = [lambda: CurriculumEnv(obs_radius_range=(3.0, 7.0)) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096)

    callback = CurriculumCallback()

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - start

    save_path = "./models_dynamic/dynamic_5000k_model"
    model.save(save_path)
    vec_env.close()
    print(f"\n>> Exp 39 model: {elapsed:.1f}s ({elapsed/60:.1f}min) -- saved to {save_path}")
