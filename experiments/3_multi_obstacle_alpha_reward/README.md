# 2 Obstacles + Alpha Reward Bonus

## What's different from `multi_obstacle_ablation`?

One line in `env_dynamic.py`:
```python
reward += 0.3 * alpha  # incentivize keeping alpha high when safe
```

This gives the agent a small reward for keeping alpha high. The collision penalty (-100) still dominates near obstacles, so the agent will lower alpha when needed. But in safe regions, this bonus pushes alpha back up — fixing the decay problem.

## How to run

```bash
cd multi_obstacle_alpha_reward
python run_all.py
```

~75-90 min on GPU.

## What to look for

Compare `plots/4_alpha_adaptation.png` from this folder vs `multi_obstacle_ablation/plots/4_alpha_adaptation.png`:
- Without alpha reward: alpha drops near obstacles and stays low
- With alpha reward: alpha should dip near obstacles then **recover** in between

## Reward breakdown (dynamic env only)

| Term | Value | Purpose |
|------|-------|---------|
| Progress | `+50 * Δdist` | Move toward target |
| Target reached | `+100` | Goal bonus |
| Collision | `-100` | Safety penalty |
| Time penalty | `-0.5` | Don't dawdle |
| Lateral penalty | `-0.01 * \|k_y\|` | Prefer straight paths |
| **Alpha bonus** | **`+0.3 * α`** | **Keep alpha high when safe** |
