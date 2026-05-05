import argparse

import numpy as np
import pygame

from flygym.compose import ActuatorType
from flygym.examples.locomotion import TurningController
from miniproject.interactive import GameState
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
        default=4,
        help="The level of the simulation to run. Default is 0.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="The random seed for the simulation. Default is 0.",
    )
    parser.add_argument(
        "--dont-use-pygame-rendering",
        action=argparse.BooleanOptionalAction,
        help=(
            "If experiencing rendering issues, set this option to use opencv rendering instead of pygame."
            "Also requires installing the pynput library."
        ),
    )
    parser.add_argument(
        "--render-fly-vision",
        action=argparse.BooleanOptionalAction,
        help="Whether to also render what the fly sees from its perspective.",
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

    print("Getting controllers")
    if args.dont_use_pygame_rendering:
        from miniproject.interactive.controls_pynput import KeyboardControlPynput
        import cv2

        controls = KeyboardControlPynput(game_state)
        cv2.namedWindow(
            WINDOW_NAME,
        )
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        def get_controller_gains():
            pygame.event.pump()
            gain_left, gain_right = controls.get_actions()
            return gain_left, gain_right

        def render(frame: np.ndarray):
            cv2.imshow(WINDOW_NAME, frame[..., ::-1])  # convert RGB to BGR for opencv
            cv2.waitKey(1)

    else:
        from miniproject.interactive import KeyboardControl

        controls = KeyboardControl(game_state)
        display_size = (1024, 1024 if args.render_fly_vision else 512)
        screen = pygame.display.set_mode(display_size)
        pygame.display.set_caption(WINDOW_NAME)

        def get_controller_gains():
            events = pygame.event.get()
            controls.process_events(events)
            keys_pressed = pygame.key.get_pressed()
            gain_left, gain_right = controls.get_actions(keys_pressed)
            return gain_left, gain_right

        def render(frame: np.ndarray):
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if frame_surface.get_size() != display_size:
                frame_surface = pygame.transform.smoothscale(
                    frame_surface, display_size
                )
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

    step = 0
    while not game_state.get_quit():
        gain_left, gain_right = get_controller_gains()

        if game_state.get_reset():
            pass  # not implemented yet
        if game_state.get_quit():
            break

        joint_angles, adhesion_signals = controller.step(
            sim
        )
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion_signals)
        sim.step()

        if sim.render_as_needed():
            # Render the latest RGB frame from flygym into the pygame window.
            frame = np.concatenate(
                [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
            )
            
            if args.render_fly_vision:
                # 1. Get Ommatidia Readouts instead of raw vision
                omm = sim.get_ommatidia_readouts(sim.fly.name)
                
                # 2. Convert to 2D human-readable images using the controller's retina tool
                left_img = controller.retina.hex_pxls_to_human_readable(omm[0].max(-1), color_8bit=True)
                right_img = controller.retina.hex_pxls_to_human_readable(omm[1].max(-1), color_8bit=True)
                
               # 3. Define the vertical band (skip top 1/4, stop at 1/2)
                h = left_img.shape[0]
                start_row = h // 4
                end_row = h // 2
                
                # HARDCODED HORIZONTAL CROP: 
                # Let's say we want to shave exactly 4 pixels off the left and right edges
                trim = 120
                
                # Apply the 2D Slice! [vertical_start : vertical_end, horizontal_start : horizontal_end]
                left_crop = left_img[start_row:end_row, trim:]
                right_crop = right_img[start_row:end_row, :-trim]
                
                # 4. Concatenate left and right eyes side-by-side
                fly_vision_gray = np.concatenate((left_crop, right_crop), axis=1)
                
                # 5. Convert 2D Grayscale to 3D RGB (so it can stack with the 3rd-person camera)
                # This stacks the grayscale array 3 times to make Red, Green, and Blue identical
                fly_vision_rgb = np.stack((fly_vision_gray, fly_vision_gray, fly_vision_gray), axis=-1)

                # 6. Pad the width to match the main camera frame perfectly
                pad_width = (frame.shape[1] - fly_vision_rgb.shape[1]) // 2
                fly_vision = np.pad(
                    fly_vision_rgb,
                    (
                        (0, 0),                 # Don't pad height
                        (pad_width, pad_width), # Pad width equally on left and right
                        (0, 0),                 # Don't pad color channels
                    ),
                    mode='constant'
                )
                frame = np.vstack((fly_vision, frame))
            render(frame)
        step += 1

    controls.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
