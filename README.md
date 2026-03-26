pip install 'stable-baselines3[extra]'

# Safety CBF — Dynamic Alpha for Robot Navigation

## Project Structure

```
experiments/
├── 1_baseline_single_obstacle/      # Baseline: 1 obs, no alpha reward
├── 2_multi_obstacle/                # 2 obs, no alpha reward
├── 3_multi_obstacle_alpha_reward/   # 2 obs + reward += 0.3 * alpha
├── 4_single_obstacle_alpha_reward/  # 1 obs + reward += 0.3 * alpha
└── 5_cbf_penalty/                   # 2 obs + reward -= ||safe_u - k_nom||
legacy/                              # Old standalone scripts & artifacts
```

## How to run an experiment
```bash
cd experiments/5_cbf_penalty
python run_all.py
```

## Key Concepts

**CBF constraint:** `L_g_h @ u >= -alpha * h(x)`

- **High alpha** near obstacle = loose constraint = CBF barely intervenes = tight path
- **Low alpha** near obstacle = tight constraint = CBF steers away = wide detour
- **Ideal dynamic alpha:** high when safe (efficient), low near obstacles (safe), recovers after passing

**Alpha decay problem:** Alpha drops near obstacles but never recovers — because when far from obstacles, the CBF constraint is trivially satisfied regardless of alpha (no learning signal).

**Experiments test different fixes:**
- Alpha reward bonus (exp 3, 4): `reward += 0.3 * alpha` — works but feels artificial
- CBF intervention penalty (exp 5): `reward -= ||safe_u - k_nom||` — penalizes unnecessary safety corrections
