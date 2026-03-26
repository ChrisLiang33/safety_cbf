# CBF Penalty Ablation (2 Obstacles)

## Hypothesis
Penalizing CBF intervention magnitude (`reward -= 1.0 * ||safe_u - k_nom||`) creates a natural incentive for the agent to raise alpha between obstacles, without the artificial `reward += 0.3 * alpha` bonus.

## Why this should work
- Low alpha near an upcoming obstacle → CBF intervenes early and heavily → big penalty
- High alpha when safe → CBF stays quiet → no penalty
- Between obstacles: low alpha causes premature intervention for the next obstacle → penalty → incentive to recover alpha

## Key change from `multi_obstacle_ablation`
In both `env_dynamic.py` and `env_fixed_alpha.py`:
```python
cbf_cost = float(np.linalg.norm(safe_u - k_nom))
reward -= 1.0 * cbf_cost  # penalize CBF intervention
```

## What to look for in results
- **Plot 4 (alpha adaptation):** Does alpha dip near obstacles and recover between them?
- **Plot 2 (CBF intervention):** Does dynamic alpha have lower total intervention than fixed alphas?
- **Plot 3 (aggregate):** Does dynamic alpha maintain 100% success + 0% collisions?

## Run
```bash
cd cbf_penalty_ablation
python run_all.py
```
Trains 5 models (4 fixed + 1 dynamic) at 900k steps each, then evaluates.
