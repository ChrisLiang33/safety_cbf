# Ablation Study: Fixed Alpha vs Dynamic Alpha

## Purpose
Compare separately-trained fixed-alpha CBF policies against the dynamic-alpha policy to prove that learning alpha is beneficial.

## Steps

### 1. Train the fixed-alpha baselines (~4x normal training time)
```bash
cd safety_cbf/ablation
python train_fixed_alphas.py
```
This trains 4 PPO models (alpha = 0.1, 0.5, 1.0, 5.0), each for 900k steps.
Each model only learns [k_x, k_y] with alpha baked into the env.
Models saved to `ablation/models_fixed/`.

### 2. Evaluate everything
```bash
python evaluate_ablation.py
```
Runs all 5 fixed-alpha models + your dynamic-alpha model (run4 @ 900k) on 5 scenarios.
Output saved to `ablation/plots/ablation_proper.png`.

## What the output shows (3 columns per scenario)
- **Left**: Trajectory overlay — all methods on same map
- **Middle**: Bar chart — total reward + min obstacle distance
- **Right**: Alpha vs distance over time (dynamic only) — proves alpha correlates with obstacle proximity

## Files
- `env_fixed_alpha.py` — Env where alpha is fixed, action is [k_x, k_y] only
- `train_fixed_alphas.py` — Trains one model per fixed alpha
- `evaluate_ablation.py` — Loads all models, runs scenarios, plots + prints summary table

## Config
To change the dynamic model being compared, edit `DYNAMIC_MODEL_PATH` at the top of `evaluate_ablation.py`.
