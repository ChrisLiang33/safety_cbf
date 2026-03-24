# Ablation Study: Fixed Alpha vs Dynamic Alpha

## Purpose
Compare separately-trained fixed-alpha CBF policies against the dynamic-alpha policy to prove that learning alpha is beneficial.

## Steps

### 1. Train the fixed-alpha baselines
```bash
cd fixedVSdynamic_ablation
python train_fixed_alphas.py
```
Trains 4 PPO models (alpha = 0.1, 0.5, 1.0, 5.0), each for 900k steps on GPU.
Each model only learns [k_x, k_y] with alpha baked into the env.
Models saved to `models_fixed/`. Training time is logged per alpha.

### 2. (Optional) Retrain dynamic model with optimized env
```bash
python train_dynamic_optimized.py
```
Trains a dynamic-alpha model using the optimized parametric QP env.
Logs training time so you can compare speed against the original `train.py`.
Model saved to `models_dynamic_optimized/`.

### 3. Evaluate everything
```bash
python evaluate_ablation.py
```
Runs all fixed-alpha models + your dynamic-alpha model on 5 scenarios.
Output saved to `plots/ablation_proper.png`.

## What the output shows (3 columns per scenario)
- **Left**: Trajectory overlay — all methods on same map
- **Middle**: Bar chart — total reward + min obstacle distance
- **Right**: Alpha vs distance over time (dynamic only) — proves alpha correlates with obstacle proximity

## Files
- `env_fixed_alpha.py` — Env where alpha is fixed, action is [k_x, k_y] only (parametric QP)
- `env_dynamic_optimized.py` — Same as main env.py but with parametric QP for faster training
- `train_fixed_alphas.py` — Trains one model per fixed alpha, logs time
- `train_dynamic_optimized.py` — Trains dynamic model with optimized env, logs time
- `evaluate_ablation.py` — Loads all models, runs scenarios, plots + prints summary table

## Config
To change the dynamic model being compared, edit `DYNAMIC_MODEL_PATH` at the top of `evaluate_ablation.py`.
