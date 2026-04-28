import argparse

import numpy as np
import pygame

from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController

from miniproject.interactive import KeyboardControl, GameState
from miniproject import MiniprojectSimulation
from submission.controller import Controller

WINDOW_NAME = "COBAR 2026 Miniproject"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--keyboard-mode",
        choices=["hold", "sticky"],
        default="hold",
        help=(
            "Keyboard control mode: 'sticky' keeps last gait command until changed, "
            "'hold' only walks while movement keys are actively pressed."
        ),
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        help="The level of the simulation to run. Default is 0.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed for the simulation. Default is 0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Game state and control setup
    game_state = GameState()

    sim = MiniprojectSimulation(
        level=args.level,
        seed=args.seed,
    )
    controller = Controller(sim)

    pygame.init()

    display_size = (1024, 512)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption(WINDOW_NAME)

    print("Getting controllers")
    controls = KeyboardControl(game_state, control_mode=args.keyboard_mode)

    step = 0
    while not game_state.get_quit():
        events = pygame.event.get()
        controls.process_events(events)
        keys_pressed = pygame.key.get_pressed()
        if game_state.get_reset():
            pass  # not implemented yet
        if game_state.get_quit():
            break

        gain_left, gain_right = controls.get_actions(keys_pressed)
        joint_angles, adhesion_signals = controller.step(sim)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()

        if sim.render_as_needed():
            # Render the latest RGB frame from flygym into the pygame window.
            frame = np.concatenate(
                [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
            )
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if frame_surface.get_size() != display_size:
                frame_surface = pygame.transform.smoothscale(
                    frame_surface, display_size
                )
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

        step += 1

    controls.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
