# Experiment 7: Alpha-Only Squeeze Gate

## Hypothesis
When two obstacles straddle the direct path, low alpha makes BOTH CBFs squeeze the feasible velocity set simultaneously, forcing the robot to crawl. The time cost of crawling should create a strong, sustained incentive to raise alpha.

## Design
- **Agent action:** `[alpha]` (1D) — range [0.1, 5.0]
- **k_nom:** Fixed proportional controller toward target at 2 m/s
- **Obstacles:** Two obstacles straddling the direct path, surface gap 0.6–2.0m
- **Time penalty:** -1.0/step (doubled from exp 6)

## Why this is different from Experiment 6
Exp 6 had 1 obstacle — low alpha just deflects direction, robot still moves at ~2 m/s.
Here, 2 obstacles create a **gate**. Low alpha = both CBFs active = velocity set shrinks to a crawl.

### The math
At the gate approach (robot at x=4, obstacles at x=5, y=±1.5):
- `ux <= alpha * h / 2` (approx, from both constraints)
- alpha=0.1 → ux ≈ 0.11 m/s (crawling)
- alpha=5.0 → ux ≈ 5.6 m/s (unrestricted)
- That's ~50x speed difference through the gate!

## Run
```bash
cd experiments/7_alpha_only_squeeze
python run_all.py
```
