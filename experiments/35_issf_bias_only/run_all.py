"""
Experiment 35: ISSf-CBF with ONLY constant bias (no radius noise).
Isolates dynamics uncertainty to test if phi compensates for unmodeled force.

Usage:
    cd experiments/35_issf_bias_only
    python run_all.py
"""
import subprocess, sys, time

SCRIPTS = [
    ("Train ISSf-CBF (kx, ky, alpha, phi)", [sys.executable, "train_dynamic.py"]),
    ("Evaluate", [sys.executable, "evaluate_ablation.py"]),
]

if __name__ == "__main__":
    total_start = time.time()
    results = []
    for name, cmd in SCRIPTS:
        print(f"\n{'='*60}\n  STEP: {name}\n{'='*60}\n")
        step_start = time.time()
        ret = subprocess.run(cmd)
        step_elapsed = time.time() - step_start
        status = "OK" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        results.append((name, step_elapsed, status))
        print(f"\n>> {name}: {status} ({step_elapsed:.1f}s / {step_elapsed/60:.1f}min)")
        if ret.returncode != 0:
            print(f"\n!! {name} failed -- stopping early.")
            break
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}\n  ALL DONE -- Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n{'='*60}")
    for name, elapsed, status in results:
        print(f"  {status:>12s}  {elapsed:7.1f}s  {name}")
    print()
