# 1 Obstacle + Alpha Reward Bonus

## What's different from `fixedVSdynamic_ablation`?

One line in `env_dynamic_optimized.py`:
```python
reward += 0.3 * alpha  # incentivize keeping alpha high when safe
```

Same single-obstacle setup, same scenarios, same evaluation — only the dynamic env's reward is changed.

## Purpose

Isolate whether the alpha reward bonus alone fixes the decay problem, independent of adding more obstacles. Compare this folder's `plots/4_alpha_adaptation.png` against `fixedVSdynamic_ablation/plots/4_alpha_adaptation.png`.

## How to run

```bash
cd single_obstacle_alpha_reward
python run_all.py
```

~75 min on GPU.
