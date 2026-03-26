# Ablation Study: 2 Obstacles — Fixed vs Dynamic Alpha

## Why 2 obstacles?
With 1 obstacle, alpha drops near it (correct) but never recovers after passing (broken). This is because alpha has no effect when far from any obstacle — the CBF constraint is trivially satisfied regardless of alpha value, so there's zero learning signal to raise it back up.

With 2 obstacles, the robot must:
1. Lower alpha near obstacle 1
2. Recover alpha in the gap (otherwise CBF over-intervenes on approach to obstacle 2)
3. Lower alpha near obstacle 2

This creates a reward signal to raise alpha between obstacles.

## How to run

```bash
cd multi_obstacle_ablation
python run_all.py
```

This trains all models then evaluates. Expect ~75-90 min total on GPU.

## What it does (in order)

1. **Trains 4 fixed-alpha models** (α = 0.1, 0.5, 1.0, 5.0) — 900k steps each
2. **Trains 1 dynamic-alpha model** — 900k steps
3. **Evaluates all 5 models** on 8 hand-picked + 100 random scenarios

## Output plots (in `plots/`)

| File | What it shows |
|------|---------------|
| `1_trajectories.png` | Trajectory overlay per scenario, dynamic colored by alpha |
| `2_cbf_intervention.png` | How much CBF corrects the agent's action over time |
| `3_aggregate_metrics.png` | Success rate, collisions, safety margin, path efficiency (100 random) |
| `4_alpha_adaptation.png` | Alpha vs distance to BOTH obstacles — the key plot |

## What to look for in the results

**Plot 4 is the money plot.** You want to see:
- Alpha dips when distance to obstacle 1 is small
- Alpha rises in the gap between obstacles
- Alpha dips again when distance to obstacle 2 is small

**Plot 3** should show dynamic alpha matching or beating the best fixed alpha across all metrics.

## Files

- `env_dynamic.py` — 2-obstacle env, agent outputs [α, k_x, k_y], QP has 2 CBF constraints
- `env_fixed_alpha.py` — 2-obstacle env, agent outputs [k_x, k_y], alpha is fixed
- `train_fixed_alphas.py` — Trains one model per fixed alpha value
- `train_dynamic.py` — Trains the dynamic alpha model
- `evaluate_ablation.py` — Loads all models, runs scenarios, generates plots + console table
- `run_all.py` — Orchestrates train → train → evaluate

## Environment details

- Observation (11-dim): `[rel_obs1_x, rel_obs1_y, obs1_r, rel_obs2_x, rel_obs2_y, obs2_r, rel_tgt_x, rel_tgt_y, tgt_r, vel_x, vel_y]`
- Obstacle 1 spawns at x ∈ [2, 5], obstacle 2 at x ∈ [5.5, 8.5]
- Target at x = 10.0
- Both obstacles have radius 1.0
