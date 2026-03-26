# Experiment 6: Alpha-Only Control

## Hypothesis
When the agent can ONLY control alpha (not k_nom), it must learn optimal alpha scheduling to maximize reward. No escape route through path planning.

## Design
- **Agent action:** `[alpha]` (1D) — range [0.1, 5.0]
- **k_nom:** Fixed proportional controller, always heads toward target at 2 m/s
- **Fixed alpha baselines:** Fully deterministic (no training needed) — just run the simulation

## Why this is different
In experiments 1-5, the agent controls both alpha and k_nom. It can minimize CBF intervention by planning better paths, making alpha irrelevant. Here, k_nom is locked — the ONLY way to influence the trajectory is through alpha.

- Low alpha → CBF intervenes more → wider detour → less progress → less reward
- High alpha → CBF stays quiet → direct path → more progress → more reward
- Too high near obstacle → collision → -100 reward

## Expected alpha behavior
High → dip near obstacle → recover after passing. The progress reward alone should drive this, since low alpha directly costs reward via CBF-induced detours.

## Run
```bash
cd experiments/6_alpha_only
python run_all.py
```
Only trains 1 model (dynamic alpha). Fixed baselines need no training.
