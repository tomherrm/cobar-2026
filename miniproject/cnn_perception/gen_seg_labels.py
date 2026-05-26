"""Generate the CNN segmentation training set: (raw vision, class mask) pairs
from the fly's eye cameras across several (level, seed) episodes.

The fly is driven on a **beeline straight at the banana**. The grass band is
placed geometrically *between* the fly's spawn and the banana, so heading
straight at the banana guarantees the fly traverses the grass on every L2 seed.
(Earlier drives — a blind forward-wander, then the heuristic controller — both
left some episodes grass-free: the wander missed the band when the banana
spawned behind the fly, and the heuristic is itself unreliable on L2 and
sometimes never reaches the grass.) L0/L1 episodes (no grass blades) are
included as negatives, so the CNN learns that green *terrain* is not a grass
*blade* — the exact distinction the old RGB heuristic could not make.

Output: seg_data.npz with
    vision (N, 2, res, res, 3) uint8   — both eyes, downsampled
    mask   (N, 2, res, res)    uint8   — class labels {0 bg, 1 grass, 2 banana}
    meta   (N, 3) int                 — (level, seed, step) per frame

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/cnn_perception/gen_seg_labels.py
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from cnn_perception.seg_utils import SegRenderer
from submission import movement_correction


def _yaw_deg(quat):
    """Body Z-rotation in degrees — same convention as the heuristic controller."""
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    _, _, yaw = rot.as_euler("xyz", degrees=True)
    return float(yaw)


def _wrap_deg(x):
    return (float(x) + 180.0) % 360.0 - 180.0


def beeline_episode(level: int, seed: int, max_steps: int, subsample: int,
                    res: int):
    """Drive the fly straight at the banana; capture (vision, mask) every
    `subsample` steps. Returns (vision_list, mask_list, n_steps, min_dist) —
    min_dist is the closest the fly got to the banana, a sanity check that the
    beeline steering is actually pointed the right way."""
    sim = MiniprojectSimulation(level=level, seed=seed)
    turning = TurningController(sim.timestep)
    seg = SegRenderer(sim)
    fly_name = sim.fly.name
    banana_xy = np.asarray(sim.world.banana_xy, dtype=np.float64)

    vision_list, mask_list = [], []
    min_dist = float("inf")

    step = 0
    for step in range(max_steps):
        if step % subsample == 0:
            # get_raw_vision is cached per sim-step — no extra render cost.
            raw_vision = sim.get_raw_vision(fly_name)   # list of (H, W, 3)
            masks = seg.render()                        # list of (H, W)
            eyes_v, eyes_m = [], []
            for rgb, msk in zip(raw_vision, masks):
                eyes_v.append(
                    cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
                )
                eyes_m.append(
                    cv2.resize(msk, (res, res), interpolation=cv2.INTER_NEAREST)
                )
            vision_list.append(np.stack(eyes_v))        # (2, res, res, 3)
            mask_list.append(np.stack(eyes_m))          # (2, res, res)

        # Beeline steering: face the banana, drive forward. The yaw-correction
        # formula matches the heuristic's proven COMMIT mode.
        quat = sim.get_body_rotations(fly_name)[0]
        fly_xy = np.asarray(sim.get_body_positions(fly_name)[0][:2])
        target_bearing = np.degrees(np.arctan2(
            banana_xy[1] - fly_xy[1], banana_xy[0] - fly_xy[0]
        ))
        yaw_err = _wrap_deg(target_bearing - _yaw_deg(quat))
        yaw_corr = float(np.clip(yaw_err / 30.0, -0.6, 0.6))
        drive = np.array([1.2 - yaw_corr, 1.2 + yaw_corr], dtype=np.float32)

        joint_angles, adhesion = turning.step(drive)
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(fly_name, ActuatorType.ADHESION, adhesion)
        sim.step()

        dist = float(np.linalg.norm(fly_xy - banana_xy))
        min_dist = min(min_dist, dist)

        # Stop on flip — post-flip views are not deployment-relevant.
        quat = sim.get_body_rotations(fly_name)[0]
        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )
        if abs(pitch) > 90 or abs(roll) > 85:
            break

    # Free the MuJoCo model, renderers and GL contexts promptly.
    seg.close()
    del seg, sim, turning
    gc.collect()
    return vision_list, mask_list, step + 1, min_dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--l2-seeds", type=str, default="1,67,777,2,42",
                   help="Level-2 seeds (the target: grass-cluster obstacles).")
    p.add_argument("--l1-seeds", type=str, default="1,67",
                   help="Level-1 seeds (negatives: green hills, no blades).")
    p.add_argument("--l0-seeds", type=str, default="1",
                   help="Level-0 seeds (negatives: flat).")
    p.add_argument("--max-steps", type=int, default=10000,
                   help="Per-episode cap. The beeline reaches/enters the "
                        "grass band well within this.")
    p.add_argument("--subsample", type=int, default=25,
                   help="Capture every Nth step.")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--out", type=str,
                   default="miniproject/cnn_perception/seg_data.npz")
    return p.parse_args()


def main():
    args = parse_args()

    episodes = []
    episodes += [(2, int(s)) for s in args.l2_seeds.split(",") if s.strip()]
    episodes += [(1, int(s)) for s in args.l1_seeds.split(",") if s.strip()]
    episodes += [(0, int(s)) for s in args.l0_seeds.split(",") if s.strip()]

    print(f"Generating segmentation labels from {len(episodes)} episodes "
          f"(beeline-to-banana drive)")
    print(f"  res={args.res}  subsample={args.subsample}  max_steps={args.max_steps}")

    all_vision, all_mask, all_meta = [], [], []
    t_start = time.perf_counter()

    for level, seed in episodes:
        t0 = time.perf_counter()
        vis, msk, n_steps, min_dist = beeline_episode(
            level, seed, args.max_steps, args.subsample, args.res
        )
        dt = time.perf_counter() - t0
        steps = list(range(0, n_steps, args.subsample))[:len(vis)]
        for s in steps:
            all_meta.append((level, seed, s))
        all_vision.extend(vis)
        all_mask.extend(msk)
        grass_frac = float(np.mean([(m == 1).mean() for m in msk])) if msk else 0.0
        print(f"  L{level} s{seed:4d}: {n_steps:6d} steps -> {len(vis):4d} frames  "
              f"grass_frac={grass_frac:.3f}  min_dist={min_dist:5.1f}  ({dt:5.0f}s)")

    vision_arr = np.asarray(all_vision, dtype=np.uint8)   # (N, 2, res, res, 3)
    mask_arr = np.asarray(all_mask, dtype=np.uint8)       # (N, 2, res, res)
    meta_arr = np.asarray(all_meta, dtype=np.int32)       # (N, 3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vision=vision_arr, mask=mask_arr, meta=meta_arr)

    elapsed = time.perf_counter() - t_start
    counts = np.bincount(mask_arr.ravel(), minlength=3)
    frac = counts / max(counts.sum(), 1)
    print()
    print(f"Saved {len(vision_arr):,} frames to {out}")
    print(f"  vision {vision_arr.shape} {vision_arr.dtype}  "
          f"({vision_arr.nbytes / 1e6:.0f} MB in RAM)")
    print(f"  mask   {mask_arr.shape} {mask_arr.dtype}")
    print(f"  pixel class balance: bg={frac[0]:.3f}  grass={frac[1]:.3f}  "
          f"banana={frac[2]:.4f}")
    print(f"Total wall time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
