"""
One-shot: train both models then evaluate.
  1. Train alpha-only model (standard CBF, no phi)
  2. Train alpha+phi model (ISSf-CBF)
  3. Evaluate ablation (compare both)

Usage:
    cd experiments/18_high_noise_phi
    python run_all.py
"""
import subprocess
import sys
import time

SCRIPTS = [
    ("Train alpha-only model (kx, ky, alpha)", [sys.executable, "train_alpha_only.py"]),
    ("Train alpha+phi model (kx, ky, alpha, phi)", [sys.executable, "train_dynamic.py"]),
    ("Evaluate ablation", [sys.executable, "evaluate_ablation.py"]),
]

if __name__ == "__main__":
    total_start = time.time()
    results = []

    for name, cmd in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  STEP: {name}")
        print(f"{'='*60}\n")

        step_start = time.time()
        ret = subprocess.run(cmd)
        step_elapsed = time.time() - step_start

        status = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        results.append((name, step_elapsed, status))
        print(f"\n>> {name}: {status} ({step_elapsed:.1f}s / {step_elapsed/60:.1f}min)")

        if ret.returncode != 0:
            print(f"\n!! {name} failed — stopping early.")
            break

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  ALL DONE — Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")
    for name, elapsed, status in results:
        print(f"  {status:>12s}  {elapsed:7.1f}s  {name}")
    print()
