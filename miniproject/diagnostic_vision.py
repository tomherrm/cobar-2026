"""
Diagnostic script — verify the visual blade detection system.

Run from cobar-2026/ directory:
    .\.venv\Scripts\python.exe miniproject/diagnostic_vision.py

Saves forward-band images to miniproject/vision_debug/ and prints per-step
statistics so we can verify:
  1. Forward-band dark fraction baseline (~15%) in open terrain
  2. Dark fraction rises when grass blades enter the forward visual field
  3. detect_blade_edge triggers at the right moments (not in open terrain / on hills)
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from flygym.compose import ActuatorType
from miniproject.simulation import MiniprojectSimulation
from submission.controller import Controller
from submission import movement_correction
from flygym.vision.retina import Retina

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'vision_debug')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAPTURE_STEPS = [10, 100, 300, 600, 1000, 2000, 3000, 4000, 5000, 6000]
PRINT_EVERY = 50
MAX_STEPS = max(CAPTURE_STEPS) + 1


def save_frame(step, omm, retina, detect_result):
    grass_slow, grass_bias, curr_areas, triggered = detect_result

    # Full hex images for reference
    left_max  = retina.hex_pxls_to_human_readable(omm[0].max(-1), color_8bit=True)
    right_max = retina.hex_pxls_to_human_readable(omm[1].max(-1), color_8bit=True)

    # Forward rectangular band — the region detect_blade_edge actually uses
    rect = movement_correction._crop_hex_to_rect(omm, retina.ommatidia_id_map)
    # rect shape: (2, n_rows, n_cols), values in [0, 1]
    left_rect  = rect[0]   # (n_rows, n_cols)
    right_rect = rect[1]

    left_dark_mask  = (left_rect  < 0.15).astype(np.float32)
    right_dark_mask = (right_rect < 0.15).astype(np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f'Step {step} | triggered={triggered} | slow={grass_slow:.2f} '
        f'| fwd_dark=[{curr_areas[0]:.3f}, {curr_areas[1]:.3f}]',
        fontsize=13
    )

    axes[0, 0].imshow(left_max, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'LEFT max(-1) full hex\nmean={left_max.mean():.1f}')

    axes[0, 1].imshow(right_max, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'RIGHT max(-1) full hex\nmean={right_max.mean():.1f}')

    axes[0, 2].imshow(left_rect, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(
        f'LEFT forward band (raw max-1)\ndark<0.15: {left_dark_mask.mean():.1%}'
    )

    axes[0, 3].imshow(right_rect, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title(
        f'RIGHT forward band (raw max-1)\ndark<0.15: {right_dark_mask.mean():.1%}'
    )

    axes[1, 0].imshow(left_dark_mask, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'LEFT dark mask\n(pale-type on green = 1)')

    axes[1, 1].imshow(
        np.abs(np.diff(left_rect, axis=1)), cmap='hot', vmin=0, vmax=0.3
    )
    axes[1, 1].set_title(
        f'LEFT gradient\nmean={np.abs(np.diff(left_rect, axis=1)).mean():.3f}'
    )

    axes[1, 2].imshow(right_dark_mask, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'RIGHT dark mask\n(pale-type on green = 1)')

    axes[1, 3].imshow(
        np.abs(np.diff(right_rect, axis=1)), cmap='hot', vmin=0, vmax=0.3
    )
    axes[1, 3].set_title(
        f'RIGHT gradient\nmean={np.abs(np.diff(right_rect, axis=1)).mean():.3f}'
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'step_{step:05d}.png')
    plt.savefig(path, dpi=80)
    plt.close()
    print(f'  [saved] {path}')


def main():
    print('Initialising simulation (level=2, seed=1)...')
    sim = MiniprojectSimulation(level=2, seed=1)
    controller = Controller(sim)
    retina = Retina()

    print(f'Output dir: {OUTPUT_DIR}')
    print(f'Capture at steps: {CAPTURE_STEPS}')
    print(f'{"step":>6}  {"loom":>5}  {"slow":>5}  {"fwd_L":>7}  {"fwd_R":>7}  '
          f'{"growth_L":>9}  {"growth_R":>9}')
    print('-' * 65)

    prev_areas = np.zeros(2)

    for step in range(MAX_STEPS):
        omm = sim.get_ommatidia_readouts(sim.fly.name)

        detect_result = movement_correction.detect_blade_edge(
            omm, retina, prev_areas,
            growth_threshold=controller.grass_growth_threshold,
            proximity_threshold=controller.grass_proximity_threshold,
        )
        grass_slow, grass_bias, curr_areas, triggered = detect_result

        if step in CAPTURE_STEPS:
            save_frame(step, omm, retina, detect_result)

        if step % PRINT_EVERY == 0:
            growth = curr_areas - prev_areas
            print(f'{step:6d}  {str(triggered):>5}  {grass_slow:5.2f}  '
                  f'{curr_areas[0]:7.3f}  {curr_areas[1]:7.3f}  '
                  f'{growth[0]:9.4f}  {growth[1]:9.4f}')

        prev_areas = curr_areas

        # Step simulation
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations(sim.fly.name)[0]
        drives = controller.drive_logic(olfaction, quat, omm)
        joint_angles, adhesion = controller.turning_controller.step(drives)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

    print('\nDone. Open vision_debug/ to inspect the images.')


if __name__ == '__main__':
    main()
