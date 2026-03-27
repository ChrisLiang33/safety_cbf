# Experiment 8: Alpha-Only Long Squeeze

## Hypothesis
Scaling up the map and obstacle size amplifies the time cost of low alpha.
Big obstacles (r=5) create a ~30m CBF influence zone. Crawling through at 0.3 m/s
vs full speed at 3.0 m/s costs hundreds of extra steps — an undeniable penalty.

## Design
- **Map:** 100m (robot at origin, target at x=100)
- **Agent action:** `[alpha]` (1D) — range [0.1, 5.0]
- **k_nom:** Fixed proportional controller at 3 m/s
- **Obstacles:** radius=5, straddling direct path, surface gap 1–6m
- **Time penalty:** -1.0/step
- **Max steps:** 600
- **Training:** 1.5M steps (more for longer episodes)

## Why this is different from Experiment 7
Exp 7: 10m map, r=1 obstacles, ~4m influence zone.
Exp 8: 100m map, r=5 obstacles, ~30m influence zone.

The squeeze zone is 7x longer. At alpha=0.1:
- Robot crawls at ~0.3 m/s through 30m → ~1000 steps (timeout!)
- At alpha=5.0: full speed 3 m/s → ~100 steps

The time penalty alone: -1000 vs -100 = 900 reward difference.

## Run
```bash
cd experiments/8_alpha_only_long_squeeze
python run_all.py
```
