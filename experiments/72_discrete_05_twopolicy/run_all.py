"""Run all: Phase 1 (nav) -> eval nav success -> Phase 2 (discrete safety) -> evaluate."""
import subprocess, sys, time
SCRIPTS = [
    ("Phase 1: Nav Moving Obs", [sys.executable, "train_phase1_nav.py"]),
    ("Nav Success Check", [sys.executable, "eval_nav_success.py", "--episodes", "500", "--threshold", "0.7"]),
    ("Phase 2: Discrete Safety", [sys.executable, "train_phase2_safety.py"]),
    ("Evaluate", [sys.executable, "evaluate_randomized.py"]),
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
            if name == "Nav Success Check":
                print("\n!! Navigation policy failed success check. Aborting pipeline. !!")
            break
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}\n  ALL DONE -- {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n{'='*60}")
    for name, elapsed, status in results:
        print(f"  {status:>12s}  {elapsed:7.1f}s  {name}")
