"""Run the heuristic controller and record (vision, scalars, action) tuples
in the env's exact Dict-observation format. The output is a behavior-cloning
training set used to warm-start the PPO policy network.

The controller steps every frame (its state machine needs continuity) but
transitions are only recorded every `--subsample` frames — adjacent vision
frames are nearly identical, so this shrinks the dataset with little loss.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/rl/collect_demos.py

Defaults: levels 0+1, the 3 known seeds, subsample 3. Outputs
miniproject/rl/demos_vision.npz with arrays `vision` (N, 6, 64, 64) uint8,
`scalars` (N, 8) float32, `actions` (N, 2) float32.
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
from submission.controller import Controller as HeuristicController
from submission import movement_correction
from rl.cobar_env import _build_observation


def collect_episode(level: int, seed: int, max_steps: int,
                    success_radius: float, subsample: int = 1):
    """Run the heuristic on (level, seed); return (vision_list, scalar_list,
    action_list, outcome).

    The controller steps every frame (its state machine needs continuity),
    but transitions are only RECORDED every `subsample` frames.
    """
    sim = MiniprojectSimulation(level=level, seed=seed)
    controller = HeuristicController(sim)
    banana_xy = np.asarray(sim.world.banana_xy, dtype=np.float32)

    vision_list = []
    scalar_list = []
    action_list = []
    outcome = "timeout"

    last_action = np.zeros(2, dtype=np.float32)

    for step in range(max_steps):
        fly_name = sim.fly.name
        # Gather all sensor data we'll need both for the observation and to
        # let the controller do its thing.
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

        # Run the heuristic; it caches debug_info / _history with the drive
        joint_angles, adhesion = controller.step(sim)
        last_step = controller._history[-1]
        action = np.array([last_step["drive_L"], last_step["drive_R"]], dtype=np.float32)

        # Record every `subsample`-th frame. `obs` was built with the
        # previous frame's action, `action` is this frame's — a consistent
        # (obs, action) pair regardless of subsampling, because last_action
        # is updated every real frame just below.
        if step % subsample == 0:
            vision_list.append(obs["vision"])
            scalar_list.append(obs["scalars"])
            action_list.append(action)

        last_action = action

        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(fly_name, ActuatorType.ADHESION, adhesion)
        sim.step()

        # Termination
        dist = float(np.linalg.norm(fly_xy - banana_xy))
        if dist <= success_radius:
            outcome = "success"
            break
        if abs(pitch) > 90 or abs(roll) > 85:
            outcome = "flip"
            break

    return vision_list, scalar_list, action_list, outcome


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=str, default="0,1",
                   help="Comma-separated levels to collect from. Default 0,1 "
                        "(where the heuristic succeeds reliably).")
    p.add_argument("--seeds", type=str, default="1,67,777",
                   help="Comma-separated seeds. Default = the 3 known seeds.")
    p.add_argument("--max-steps", type=int, default=80000,
                   help="Max steps per episode (heuristic L0 typical: 30-50k).")
    p.add_argument("--subsample", type=int, default=3,
                   help="Record every Nth frame. Adjacent vision frames are "
                        "nearly identical, so subsampling shrinks the dataset "
                        "with little information loss.")
    p.add_argument("--out", type=str, default="miniproject/rl/demos_vision.npz",
                   help="Output file.")
    p.add_argument("--include-flips", action="store_true",
                   help="Keep transitions even from flipped episodes (default: drop).")
    return p.parse_args()


def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    all_vision = []
    all_scalars = []
    all_actions = []
    summary = []

    t_start = time.perf_counter()
    for level in levels:
        for seed in seeds:
            t_ep_start = time.perf_counter()
            vision_list, scalar_list, action_list, outcome = collect_episode(
                level=level,
                seed=seed,
                max_steps=args.max_steps,
                success_radius=3.0,
                subsample=args.subsample,
            )
            t_ep = time.perf_counter() - t_ep_start
            n = len(vision_list)
            keep = (outcome != "flip") or args.include_flips
            tag = "[KEEP]" if keep else "[DROP]"
            print(f"L{level} s{seed:4d}: {outcome:7s}  recorded={n:6d}  {t_ep:5.0f}s  {tag}")
            summary.append((level, seed, outcome, n, t_ep))
            if keep:
                all_vision.extend(vision_list)
                all_scalars.extend(scalar_list)
                all_actions.extend(action_list)

    vision_arr = np.array(all_vision, dtype=np.uint8)
    scalar_arr = np.array(all_scalars, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vision=vision_arr,
        scalars=scalar_arr,
        actions=act_arr,
        summary=np.array(summary, dtype=object),
    )

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Saved {len(act_arr):,} transitions to {out}  (subsample={args.subsample})")
    print(f"  vision  {vision_arr.shape} {vision_arr.dtype}  "
          f"({vision_arr.nbytes / 1e6:.0f} MB in RAM)")
    print(f"  scalars {scalar_arr.shape} {scalar_arr.dtype}")
    print(f"  actions {act_arr.shape} {act_arr.dtype}")
    print(f"Total wall time: {elapsed:.0f}s")
    if len(act_arr):
        print(f"Action stats: drive_L mean={act_arr[:, 0].mean():+.3f} std={act_arr[:, 0].std():.3f}, "
              f"drive_R mean={act_arr[:, 1].mean():+.3f} std={act_arr[:, 1].std():.3f}")
        print(f"Action range: L=[{act_arr[:, 0].min():+.2f}, {act_arr[:, 0].max():+.2f}], "
              f"R=[{act_arr[:, 1].min():+.2f}, {act_arr[:, 1].max():+.2f}]")


if __name__ == "__main__":
    main()
