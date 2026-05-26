"""PPO training entrypoint for the cobar-2026 RL controller.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/rl/train.py --total-steps 50000

Checkpoints land in `miniproject/rl/checkpoints/`. Tensorboard logs in
`miniproject/rl/tb/`. Trained policy is saved as `policy.zip` in the
checkpoint directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import datetime
from pathlib import Path

# Ensure miniproject root on sys.path so `from rl.cobar_env` works
THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.cobar_env import CobarEnv
from rl.policy_config import POLICY_KWARGS


def make_env(level: int, seed_pool: list[int] | None, max_steps: int,
             smooth_alpha: float = 0.7):
    def _thunk():
        return CobarEnv(
            level=level, seed_pool=seed_pool, max_steps=max_steps,
            action_smooth_alpha=smooth_alpha,
        )
    return _thunk


def build_fresh_ppo(env, args, tb_dir):
    """A PPO with our tuned hyperparameters and the shared policy architecture.

    Used both for the from-scratch baseline and for the BC warm start (where
    the policy weights are then overwritten from the BC zip) — so the warm
    start runs with these hyperparameters, not whatever defaults the BC zip
    happened to be saved with.
    """
    return PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=str(tb_dir),
        policy_kwargs=POLICY_KWARGS,
        device="auto",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=2,
                   help="Cobar level to train on (0..4). Default 2.")
    p.add_argument("--total-steps", type=int, default=50_000,
                   help="Total training timesteps.")
    p.add_argument("--max-episode-steps", type=int, default=20_000,
                   help="Max steps per training episode.")
    p.add_argument("--seed-pool", type=str, default="random",
                   help="Comma-separated seeds (e.g. '1,67,777') or 'random'.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Number of parallel envs (CPU only). Keep low (1-2) "
                        "to fit memory; vision rendering is heavy.")
    p.add_argument("--out-dir", type=str, default="miniproject/rl/checkpoints",
                   help="Where to save checkpoints / final policy.")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512,
                   help="Rollout length per update.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=5000,
                   help="Save a periodic checkpoint every N steps.")
    p.add_argument("--eval-freq", type=int, default=10000,
                   help="Evaluate (and save best) every N steps.")
    p.add_argument("--n-eval-episodes", type=int, default=5,
                   help="Episodes per evaluation.")
    p.add_argument("--eval-max-steps", type=int, default=30000,
                   help="Max steps per eval episode (longer than train cap).")
    p.add_argument("--init-from", type=str, default=None,
                   help="Path to a previously trained policy.zip to start "
                        "from (curriculum learning). Loads weights, then "
                        "continues PPO training on the current --level.")
    p.add_argument("--smooth-alpha", type=float, default=0.7,
                   help="Action low-pass smoothing: applied = (1-alpha)*prev "
                        "+ alpha*action. 1.0 = no smoothing, 0.3 = heavy. "
                        "Default 0.7 = light smoothing.")
    p.add_argument("--init-from-bc", type=str, default=None,
                   help="Path to a BC-trained policy .zip (from train_bc.py). "
                        "The BC zip is itself a full PPO model, so this loads "
                        "via PPO.load() — same path as --init-from, no weight "
                        "mapping. If both are given, --init-from wins.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tb"

    # Parse seed pool
    if args.seed_pool == "random":
        seed_pool = None
    else:
        seed_pool = [int(s) for s in args.seed_pool.split(",")]

    print(f"Training PPO on cobar level {args.level}")
    print(f"  total_steps={args.total_steps}")
    print(f"  max_episode_steps={args.max_episode_steps}")
    print(f"  seed_pool={seed_pool}")
    print(f"  n_envs={args.n_envs}")
    print(f"  out_dir={out_dir}")

    # Build env(s)
    if args.n_envs == 1:
        env = make_env(args.level, seed_pool, args.max_episode_steps, args.smooth_alpha)()
    else:
        env = DummyVecEnv([
            make_env(args.level, seed_pool, args.max_episode_steps, args.smooth_alpha)
            for _ in range(args.n_envs)
        ])

    # Three ways to start:
    #   --init-from     curriculum continuation: genuinely resume a PPO run
    #                   (keeps its policy, value function and optimizer state).
    #   --init-from-bc  BC warm start: a fresh PPO with our tuned hyperparameters,
    #                   then ONLY the policy weights are copied from the BC zip
    #                   (clean state_dict load — identical MultiInputPolicy
    #                   architecture). The BC value head is random; PPO learns
    #                   it during fine-tuning. Hyperparameters come from args,
    #                   not from whatever the BC zip happened to be saved with.
    #   neither         from-scratch baseline.
    if args.init_from is not None:
        print(f"Loading policy from {args.init_from} (curriculum continuation)")
        model = PPO.load(
            args.init_from,
            env=env,
            tensorboard_log=str(tb_dir),
            device="auto",
        )
        # Use the requested learning rate (typically lower for fine-tune).
        model.learning_rate = args.learning_rate
    elif args.init_from_bc is not None:
        print(f"BC warm start: loading policy weights from {args.init_from_bc}")
        model = build_fresh_ppo(env, args, tb_dir)
        bc_model = PPO.load(args.init_from_bc, device="auto")
        model.policy.load_state_dict(bc_model.policy.state_dict())
        del bc_model
    else:
        model = build_fresh_ppo(env, args, tb_dir)

    # Eval env for best-model tracking. Separate from the training env so
    # the eval doesn't contaminate the rollout buffer. Uses a longer step
    # budget than training to give policies a chance to actually reach the
    # banana (success requires ~30k steps for the heuristic).
    eval_env = make_env(args.level, seed_pool, args.eval_max_steps, args.smooth_alpha)()
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(out_dir),
            name_prefix="ppo_cobar",
        ),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(out_dir / "eval_logs"),
            eval_freq=max(args.eval_freq, args.n_steps),  # SB3 needs >= rollout
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]

    # Wall-clock timing for the whole training run
    t_start = time.perf_counter()
    start_iso = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[TIMER] training started at {start_iso}")
    print(f"[TIMER] target steps = {args.total_steps:,}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=False,
    )

    t_end = time.perf_counter()
    elapsed_s = t_end - t_start
    end_iso = datetime.datetime.now().isoformat(timespec="seconds")
    avg_fps = args.total_steps / elapsed_s if elapsed_s > 0 else 0.0
    h, rem = divmod(elapsed_s, 3600)
    m, s = divmod(rem, 60)
    print(f"[TIMER] training ended at {end_iso}")
    print(f"[TIMER] total elapsed: {int(h):d}h {int(m):d}m {int(s):d}s ({elapsed_s:.0f}s)")
    print(f"[TIMER] average fps: {avg_fps:.1f}")

    final_path = out_dir / "policy.zip"
    model.save(str(final_path))
    print(f"Saved final policy: {final_path}")

    # Also write a small summary file so we can read it later without
    # rummaging through stdout.
    summary_path = out_dir / "timing.txt"
    summary_path.write_text(
        f"start: {start_iso}\n"
        f"end:   {end_iso}\n"
        f"elapsed_s: {elapsed_s:.1f}\n"
        f"elapsed_human: {int(h)}h {int(m)}m {int(s)}s\n"
        f"total_steps: {args.total_steps}\n"
        f"avg_fps: {avg_fps:.2f}\n"
    )


if __name__ == "__main__":
    main()
