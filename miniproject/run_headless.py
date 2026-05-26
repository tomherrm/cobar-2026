"""
Headless run - no rendering, just simulation + death log.
Much faster than preview_debug.py for repeated testing.

Run from cobar-2026/:
    .venv/Scripts/python.exe miniproject/run_headless.py --level 2 --seed 67
    .venv/Scripts/python.exe miniproject/run_headless.py --level 2 -s 67 -s 1 -s 777
"""
import argparse
import datetime
import json
import math
import numpy as np
import tqdm

from flygym.compose import ActuatorType
from miniproject import MiniprojectSimulation
from submission.controller import Controller

MAX_NUM_STEPS = 100_000


def save_history(controller, reason, seed, level):
    if not controller._history:
        return
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'death_log_L{level}_s{seed}_{ts}.json'
    with open(path, 'w') as f:
        json.dump(list(controller._history), f, indent=2)
    print(f"  [{reason}] Saved {len(controller._history)} steps -> {path}")


def run_once(level, seed):
    print(f"\n=== Level {level}  Seed {seed} ===")
    sim = MiniprojectSimulation(level=level, seed=seed)
    controller = Controller(sim)

    banana_xy = sim.world.banana_xy

    for step in tqdm.tqdm(range(MAX_NUM_STEPS), ncols=80):
        fly_xy = np.array(sim.get_body_positions(sim.fly.name)[0][:2])

        # Success
        if np.linalg.norm(fly_xy - banana_xy) <= 3:
            save_history(controller, 'SUCCESS', seed, level)
            print(f"  SUCCESS at step {step}")
            return step, 'SUCCESS'

        joint_angles, adhesion = controller.step(sim)

        # Augment history entry with world position and step number
        if controller._history:
            controller._history[-1]['step'] = step
            controller._history[-1]['x'] = float(fly_xy[0])
            controller._history[-1]['y'] = float(fly_xy[1])

        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        # Flip detection
        d = getattr(controller, 'debug_info', {})
        roll  = d.get('roll', 0)
        pitch = d.get('pitch', 0)
        if abs(roll) > 85 or abs(pitch) > 90:
            save_history(controller, 'FLIP', seed, level)
            print(f"  FLIP at step {step}  roll={roll:.1f}°  pitch={pitch:.1f}°")
            return step, 'FLIP'

    save_history(controller, 'TIMEOUT', seed, level)
    print(f"  TIMEOUT at step {MAX_NUM_STEPS}")
    return MAX_NUM_STEPS, 'TIMEOUT'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", type=int, default=2)
    parser.add_argument("-s", "--seed",  type=int, action='append', dest='seeds')
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [67]
    return args


def main():
    args = parse_args()
    results = []
    for seed in args.seeds:
        steps, reason = run_once(args.level, seed)
        results.append((seed, steps, reason))

    if len(results) > 1:
        print("\n=== Summary ===")
        for seed, steps, reason in results:
            print(f"  seed={seed:4d}  steps={steps:6d}  {reason}")


if __name__ == "__main__":
    main()
