"""Evaluate navigation policy success rate with moving obstacles."""
import argparse
import numpy as np
from stable_baselines3 import PPO

NAV_MODEL_PATH = "./models_dynamic/nav_moving_model_v2"


def run_eval(model_path, env_cls, n_episodes=500, seed=0):
    model = PPO.load(model_path)
    env = env_cls()
    reached = 0
    collided = 0
    timeout = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        dist2target = np.linalg.norm(env.robot_pos - env.target_pos)
        if dist2target < env.target_radius:
            reached += 1
        elif terminated and not truncated:
            collided += 1
        else:
            timeout += 1
    return reached, collided, timeout, n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--model", type=str, default=NAV_MODEL_PATH)
    args = parser.parse_args()

    from env_nav_moving import NavMovingObsEnv
    env_cls = NavMovingObsEnv

    print(f"Evaluating nav policy: {args.model}")
    print(f"  {args.episodes} episodes, threshold={args.threshold:.0%}")
    reached, collided, timeout, total = run_eval(args.model, env_cls, n_episodes=args.episodes)
    success_rate = reached / total

    print(f"\n{'='*50}")
    print(f"  Navigation Success Rate: {success_rate:.1%} ({reached}/{total})")
    print(f"  Collision Rate:          {collided/total:.1%} ({collided}/{total})")
    print(f"  Timeout Rate:            {timeout/total:.1%} ({timeout}/{total})")
    print(f"{'='*50}")

    if success_rate >= args.threshold:
        print(f"\n  PASS — {success_rate:.1%} >= {args.threshold:.0%}")
        exit(0)
    else:
        print(f"\n  FAIL — {success_rate:.1%} < {args.threshold:.0%}")
        exit(1)
