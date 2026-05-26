r"""
Debug preview — shows the fly's forward visual band and centering-response signals
in real time alongside the normal simulation view.

Run from cobar-2026/:
    .\.venv\Scripts\python.exe miniproject/preview_debug.py --level 2 --seed 67
"""
import argparse
import datetime
import json
import numpy as np
import tqdm
import pygame

from flygym.compose import ActuatorType
from miniproject import MiniprojectSimulation
from submission.controller import Controller

WINDOW_NAME = "COBAR 2026 — Debug Vision"
MAX_NUM_STEPS = 100_000

# Debug panel layout constants
MAIN_H = 512          # height of main sim view
MAIN_W = 1024
DEBUG_H = 310         # height of debug panel below
SCALE = 5             # forward band image scale (16×31 → 80×155)
IMG_H = 16 * SCALE    # 80
IMG_W = 31 * SCALE    # 155
HRC_W = 30 * SCALE    # 150  (HRC is 16×30, one less column)
THRESHOLD = 0.15
BULL_EYE_THRESHOLD = 0.20

# Row 1: Raw + Mask panels for both eyes
IMG_X = [10, IMG_W + 20, IMG_W * 2 + 40, IMG_W * 3 + 60]
IMG_Y = 30            # y offset within debug panel (relative to debug panel top)
# Row 2: HRC panels
HRC_X = [10, HRC_W + 20]
HRC_Y = IMG_Y + IMG_H + 15

# Colours
BG_DEBUG   = (20, 20, 30)
COL_WHITE  = (240, 240, 240)
COL_GRAY   = (140, 140, 140)
COL_ORANGE = (255, 140, 0)
COL_BLUE   = (80, 160, 255)
COL_RED    = (220, 60, 60)
COL_GREEN  = (60, 200, 80)
COL_BAR_BG = (50, 50, 60)


def make_gray_surface(arr2d):
    """Float (rows, cols) [0,1] → scaled pygame Surface (grayscale)."""
    gray = (arr2d * 255).clip(0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)           # (rows, cols, 3)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surf, (IMG_W, IMG_H))


def make_hrc_surface(arr2d):
    """Float (rows, cols) HRC signal → blue (dark moves right) / red (dark moves left) Surface."""
    scaled = np.clip(arr2d * 20, -1, 1)
    rgb = np.zeros((*arr2d.shape, 3), dtype=np.uint8)
    pos = scaled > 0.05
    neg = scaled < -0.05
    rgb[pos, 2] = (scaled[pos] * 220).astype(np.uint8)
    rgb[neg, 0] = (-scaled[neg] * 220).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surf, (HRC_W, IMG_H))


def make_mask_surface(arr2d, threshold=THRESHOLD):
    """Float (rows, cols) [0,1] → red/dark mask surface (dark pixels highlighted)."""
    mask = arr2d < threshold
    rgb = np.zeros((*arr2d.shape, 3), dtype=np.uint8)
    rgb[mask,  0] = 220   # red = dark pixel (possible blade)
    rgb[~mask, 2] = 40    # faint blue = bright pixel (sky/ground)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(surf, (IMG_W, IMG_H))


def draw_debug_panel(screen, font_sm, font_md, debug_info, step):
    y0 = MAIN_H   # top of debug panel

    # Background
    pygame.draw.rect(screen, BG_DEBUG, (0, y0, MAIN_W, DEBUG_H))
    pygame.draw.line(screen, COL_GRAY, (0, y0), (MAIN_W, y0), 1)

    if not debug_info:
        return

    rect            = debug_info.get('rect')
    left_dark       = debug_info.get('left_dark', 0.0)
    right_dark      = debug_info.get('right_dark', 0.0)
    asym            = debug_info.get('asym', 0.0)
    roll            = debug_info.get('roll', 0.0)
    pitch           = debug_info.get('pitch', 0.0)
    roll_att        = debug_info.get('roll_attenuation', 1.0)
    grass_centering = debug_info.get('grass_centering', np.zeros(2))
    min_dark        = debug_info.get('min_dark', 0.0)
    bull_eye        = debug_info.get('bull_eye', False)
    col_signal      = debug_info.get('col_signal', 0.0)
    gap_mode        = debug_info.get('gap_mode', False)
    escape_rem      = debug_info.get('escape_remaining', 0)
    escape_phase    = debug_info.get('escape_phase', 'none')
    edge_asym       = debug_info.get('edge_asym', 0.0)
    hrc_asym        = debug_info.get('hrc_asym', 0.0)
    combined_asym   = debug_info.get('combined_asym', 0.0)
    hrc             = debug_info.get('hrc', None)

    # ── Image panels ─────────────────────────────────────────────────────────
    labels = ['L-Raw', 'L-Mask', 'R-Raw', 'R-Mask']
    if rect is not None:
        surfs = [
            make_gray_surface(rect[0]),
            make_mask_surface(rect[0]),
            make_gray_surface(rect[1]),
            make_mask_surface(rect[1]),
        ]
        for i, (surf, label) in enumerate(zip(surfs, labels)):
            x = IMG_X[i]
            lbl = font_sm.render(label, True, COL_GRAY)
            screen.blit(lbl, (x, y0 + 8))
            screen.blit(surf, (x, y0 + IMG_Y))
            pygame.draw.rect(screen, COL_GRAY, (x, y0 + IMG_Y, IMG_W, IMG_H), 1)

    # ── HRC panels (row 2) ───────────────────────────────────────────────────
    if hrc is not None:
        hrc_labels = ['L-HRC', 'R-HRC']
        for i in range(2):
            x = HRC_X[i]
            lbl = font_sm.render(hrc_labels[i], True, COL_GRAY)
            screen.blit(lbl, (x, y0 + HRC_Y - 14))
            surf = make_hrc_surface(hrc[i])
            screen.blit(surf, (x, y0 + HRC_Y))
            pygame.draw.rect(screen, COL_GRAY, (x, y0 + HRC_Y, HRC_W, IMG_H), 1)

    # ── Asymmetry bar ─────────────────────────────────────────────────────────
    bar_x = IMG_X[3] + IMG_W + 20
    bar_y = y0 + IMG_Y
    bar_w = MAIN_W - bar_x - 15
    bar_h = 18

    # asym bar
    pygame.draw.rect(screen, COL_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
    mid = bar_x + bar_w // 2
    fill = int(asym * bar_w * 2.5)
    fill = max(-bar_w // 2, min(bar_w // 2, fill))
    col = COL_ORANGE if asym > 0 else COL_BLUE
    if fill != 0:
        pygame.draw.rect(screen, col,
                         (min(mid, mid + fill), bar_y, abs(fill), bar_h))
    pygame.draw.line(screen, COL_WHITE, (mid, bar_y), (mid, bar_y + bar_h), 2)
    asym_lbl = font_sm.render(f"asym {asym:+.3f}", True, COL_WHITE)
    screen.blit(asym_lbl, (bar_x, bar_y - 14))

    # centering bar (grass_centering[0] = left drive addition)
    bar_y2 = bar_y + bar_h + 14
    pygame.draw.rect(screen, COL_BAR_BG, (bar_x, bar_y2, bar_w, bar_h))
    cent_val = float(grass_centering[0]) if hasattr(grass_centering, '__len__') else 0.0
    fill2 = int(cent_val * bar_w * 1.5)
    fill2 = max(-bar_w // 2, min(bar_w // 2, fill2))
    col2 = COL_ORANGE if cent_val > 0 else COL_BLUE
    if fill2 != 0:
        pygame.draw.rect(screen, col2,
                         (min(mid, mid + fill2), bar_y2, abs(fill2), bar_h))
    pygame.draw.line(screen, COL_WHITE, (mid, bar_y2), (mid, bar_y2 + bar_h), 2)
    cent_lbl = font_sm.render(f"centering[L] {cent_val:+.3f}", True, COL_WHITE)
    screen.blit(cent_lbl, (bar_x, bar_y2 - 14))

    # ── Text values ──────────────────────────────────────────────────────────
    tx = bar_x
    ty = bar_y2 + bar_h + 14

    def txt(s, color=COL_WHITE):
        nonlocal ty
        surf = font_sm.render(s, True, color)
        screen.blit(surf, (tx, ty))
        ty += 17

    txt(f"L_dark={left_dark:.3f}   R_dark={right_dark:.3f}   min={min_dark:.3f}",
        COL_ORANGE if bull_eye else COL_WHITE)
    txt(f"roll={roll:.1f}°  pitch={pitch:.1f}°  roll_att={roll_att:.2f}",
        COL_RED if abs(roll) > 30 else COL_WHITE)
    txt(f"col_signal={col_signal:.3f}   GAP-MODE: {'YES' if gap_mode else 'no'}   step={step}",
        COL_RED if gap_mode else COL_GREEN)
    txt(f"edge_asym={edge_asym:+.3f}   hrc_asym={hrc_asym:+.3f}   combined_asym={combined_asym:+.3f}",
        COL_GRAY)

    # New state-machine info (only present from new controller)
    mode = debug_info.get('mode')
    if mode is not None:
        n_central = debug_info.get('n_central_blades', 0)
        blades_L = debug_info.get('blades_L', 0)
        blades_R = debug_info.get('blades_R', 0)
        top_L = debug_info.get('top_count_max_L', 0)
        top_R = debug_info.get('top_count_max_R', 0)
        txt(f"mode={mode.upper():<8}  n_central={n_central}  "
            f"rgb_blades=L{blades_L} R{blades_R}  top_max=L{top_L} R{top_R}",
            COL_GREEN if mode == 'navigate' else COL_ORANGE)
        # Bio-cue signals (LPLC1 + campaniform sensilla)
        wall_force = debug_info.get('wall_force', 0.0)
        is_wall = debug_info.get('is_wall_contact', False)
        clutter = debug_info.get('clutter_factor', 1.0)
        txt(f"wall_force={wall_force:+.2f}  WALL={'YES' if is_wall else 'no '}  "
            f"clutter_factor={clutter:.2f}",
            COL_RED if is_wall else COL_GRAY)
        cty = debug_info.get('commit_target_yaw')
        sby = debug_info.get('scan_best_yaw')
        sbs = debug_info.get('scan_best_score')
        yaw = debug_info.get('yaw', 0.0)
        if mode == 'commit' and cty is not None:
            txt(f"  yaw={yaw:+.1f}°  target={cty:+.1f}°  err={(cty-yaw):+.1f}°",
                COL_GREEN)
        elif mode == 'scan':
            best = f"{sby:+.1f}" if sby is not None else "?"
            score = sbs if sbs is not None else "?"
            txt(f"  yaw={yaw:+.1f}°  best_yaw={best}  best_score={score}",
                COL_ORANGE)

    if escape_phase == 'backup':
        txt(f"*** EMERGENCY BACKUP (severe tilt) ***", COL_RED)
    elif escape_phase == 'scan':
        txt(f"*** SCAN: {escape_rem} steps left ***", COL_ORANGE)
    elif escape_phase == 'commit':
        txt(f"*** COMMIT: {escape_rem} steps left ***", COL_GREEN)


def save_history(controller, reason):
    if not controller._history:
        return
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'death_log_{ts}.json'
    with open(path, 'w') as f:
        json.dump(list(controller._history), f, indent=2)
    print(f"[{reason}] Saved {len(controller._history)} steps → {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", type=int, default=2)
    parser.add_argument("-s", "--seed",  type=int, default=67)
    return parser.parse_args()


def main():
    args = parse_args()

    sim = MiniprojectSimulation(level=args.level, seed=args.seed)
    controller = Controller(sim)

    pygame.init()
    screen = pygame.display.set_mode((MAIN_W, MAIN_H + DEBUG_H))
    pygame.display.set_caption(WINDOW_NAME)
    font_sm = pygame.font.Font(None, 18)
    font_md = pygame.font.Font(None, 22)

    def got_to_food():
        banana_xy = sim.world.banana_xy
        fly_xy = np.array(sim.get_body_positions(sim.fly.name)[0][:2])
        return np.linalg.norm(fly_xy - banana_xy) <= 3

    for step in tqdm.tqdm(range(MAX_NUM_STEPS)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if got_to_food():
            save_history(controller, 'SUCCESS')
            print(f"Got to goal in {step} timesteps.")
            break

        joint_angles, adhesion = controller.step(sim)

        # Augment history entry with world position and step number
        if controller._history:
            fly_pos = sim.get_body_positions(sim.fly.name)[0]
            controller._history[-1]['step'] = step
            controller._history[-1]['x'] = float(fly_pos[0])
            controller._history[-1]['y'] = float(fly_pos[1])

        sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
        sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
        sim.step()

        # Flip detection: abs(roll) > 85° or abs(pitch) > 90° → fly is upside down
        d = getattr(controller, 'debug_info', {})
        if abs(d.get('roll', 0)) > 85 or abs(d.get('pitch', 0)) > 90:
            save_history(controller, 'FLIP')
            print(f"Flip at step {step}  roll={d.get('roll', 0):.1f}°  pitch={d.get('pitch', 0):.1f}°")
            break

        if sim.render_as_needed():
            # Main sim frame
            frame = np.concatenate(
                [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
            )
            frame_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            if frame_surf.get_size() != (MAIN_W, MAIN_H):
                frame_surf = pygame.transform.smoothscale(frame_surf, (MAIN_W, MAIN_H))
            screen.blit(frame_surf, (0, 0))

            # Debug panel
            draw_debug_panel(screen, font_sm, font_md,
                             getattr(controller, 'debug_info', {}), step)
            pygame.display.flip()
    else:
        save_history(controller, 'TIMEOUT')
        print("Took too long.")


if __name__ == "__main__":
    main()
