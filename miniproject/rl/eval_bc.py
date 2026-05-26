"""Evaluate a policy `.zip` as a cobar controller and report outcome +
minimum distance to banana per (level, seed). Works for any PPO `.zip` —
the BC warm-start (`bc_policy.zip` from train_bc.py) or a fine-tuned PPO
policy (`policy.zip` from train.py).

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/rl/eval_bc.py --bc miniproject/rl/bc_policy.zip
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from stable_baselines3 import PPO
from rl.cobar_env import _build_observation
from submission import movement_correction


def run_one(model, level: int, seed: int,
            max_steps: int = 100_000, success_radius: float = 3.0,
            smooth_alpha: float = 0.7):
    sim = MiniprojectSimulation(level=level, seed=seed)
    turning = TurningController(sim.timestep)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=np.float32)
    last_action = np.zeros(2, dtype=np.float32)
    smoothed = np.zeros(2, dtype=np.float32)

    min_dist = float("inf")
    outcome = "timeout"

    for step in range(max_steps):
        fly_name = sim.fly.name
        olfaction = sim.get_olfaction(fly_name)
        quat = sim.get_body_rotations(fly_name)[0]
        raw_vision = sim.get_raw_vision(fly_name)
        contact_forces_world = sim.get_external_force(
            fly_name, subtract_adhesion_force=True
        )
        try:
            antenna_data = sim.get_antenna_data(fly_name)
        except Exception:
            antenna_data = None
        fly_xy = np.asarray(sim.get_body_positions(fly_name)[0][:2])
        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )

        obs = _build_observation(
            olfaction, quat, pitch, roll, raw_vision,
            contact_forces_world, antenna_data, last_action,
        )

        action, _state = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.8, 1.8).astype(np.float32)
        # Smooth identically to CobarEnv.step(); last_action fed into the
        # next obs is the smoothed value, as in the env.
        smoothed = (1.0 - smooth_alpha) * smoothed + smooth_alpha * action
        last_action = smoothed

        joint_angles, adhesion = turning.step(smoothed)
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(fly_name, ActuatorType.ADHESION, adhesion)
        sim.step()

        dist = float(np.linalg.norm(fly_xy - banana_xy))
        if dist < min_dist:
            min_dist = dist
        if dist <= success_radius:
            outcome = "success"
            return outcome, step + 1, min_dist
        if abs(pitch) > 90 or abs(roll) > 85:
            outcome = "flip"
            return outcome, step + 1, min_dist

    return outcome, max_steps, min_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc", type=str, default="miniproject/rl/bc_policy.zip",
                   help="Path to a policy .zip (BC warm-start or fine-tuned PPO).")
    p.add_argument("--levels", type=str, default="0,1,2",
                   help="Comma-separated levels.")
    p.add_argument("--seeds", type=str, default="1,67,777",
                   help="Comma-separated seeds.")
    p.add_argument("--max-steps", type=int, default=100_000)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading policy from {args.bc}")
    model = PPO.load(args.bc, device="auto")

    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print()
    print(f"{'L':>2} {'seed':>4}  {'outcome':>8}  {'steps':>6}  min_dist")
    print("-" * 40)
    for level in levels:
        for seed in seeds:
            t0 = time.perf_counter()
            outcome, steps, mind = run_one(model, level, seed, max_steps=args.max_steps)
            t = time.perf_counter() - t0
            print(f"{level:>2} {seed:>4}  {outcome:>8}  {steps:>6}  {mind:6.2f}  ({t:5.0f}s)")


if __name__ == "__main__":
    main()
