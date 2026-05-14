import numpy as np
from scipy.spatial.transform import Rotation
from miniproject.simulation import MiniprojectSimulation
from . import odor_attraction
from . import OLD_movement_correction
from flygym.vision.retina import Retina


# ═══════════════════════════════════════════════════════════════════════════════
#  TUNING GUIDE
# ─────────────────────────────────────────────────────────────────────────────
#  ALIGNMENT
#    align_bias_thr   : how aligned before walking (lower = stricter, 0.05–0.2)
#    align_drive      : turning drive during alignment (1.5–3.0)
#
#  ODOR
#    attractive_gain  : how aggressively to turn toward odor (500–2000)
#    speed_gain       : overall walk speed multiplier (1.5–3.5)
#
#  OBSTACLE DETECTION  (_get_vision)
#    obs_green_thr    : min green dominance g/total to count pixel (0.35–0.50)
#    obs_abs_thr      : min absolute green value to count pixel (60–120)
#    obs_cut_frac     : fraction of image height used as detection zone (0.2–0.45)
#
#  OBSTACLE AVOIDANCE  (_avoid_obstacles)
#    obs_reflex_thr   : score above which avoidance triggers (0.01–0.05)
#    obs_passthru_thr : max |diff| to consider symmetric → pass through (0.05–0.15)
#    obs_turn_mag     : steering drive magnitude when avoiding (2.0–5.0)
#    avoid_hold_steps : steps to keep turning after obstacle clears (30–100)
#
#  TILT
#    max_pitch_deg    : pitch angle (°) above which pitch correction activates
#    max_roll_deg     : roll angle (°) above which roll correction activates
#    K_PITCH / K_ROLL : gains passed to tilt_to_control_signal
# ═══════════════════════════════════════════════════════════════════════════════


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.retina = Retina()

        # ── Locomotion ────────────────────────────────────────────────────────
        self.speed_gain       = 2.8
        self.attractive_gain  = 1000

        # ── Alignment ────────────────────────────────────────────────────────
        self.align_bias_thr   = 0.1    # |bias| below this → aligned
        self.align_drive      = 2.0    # turning drive while aligning
        self._aligned         = False
        self._align_fade   = 0    # steps de transition restants
        self._align_fade_steps = 200  # durée du fade

        # ── Tilt compensation ─────────────────────────────────────────────────
        self.K_PITCH          = 0.05
        self.K_ROLL           = 0.06
        self.max_pitch_boost  = 0.5
        self.max_roll_boost   = 0.35
        self.max_pitch_deg    = 10
        self.max_roll_deg     = 8
        self.tilt_gain        = 1.0

    


#    self.max_roll_deg    = 8

        # ── Vision / obstacle detection ───────────────────────────────────────
        self.obs_green_thr    = 0.40   # g/total threshold
        self.obs_abs_thr      = 80.0   # absolute green value threshold
        self.obs_cut_frac     = 0.33   # top fraction of image to inspect

        # ── Obstacle avoidance ────────────────────────────────────────────────
        self.obs_reflex_thr   = 0.018  # detection score to trigger avoidance
        self.obs_passthru_thr = 0.08   # symmetric → pass through
        self.obs_turn_mag     = 3.0    # steering magnitude
        self.avoid_hold_steps = 60     # hold turn this many steps after clear

        # ── Internal state ────────────────────────────────────────────────────
        self._counter          = 0
        self._vision_interval  = 80    # steps between vision updates
        self._obs_l            = 0.0
        self._obs_r            = 0.0
        self._avoid_hold       = 0
        self._last_turn        = 0.0
        self._red_l            = 0.0
        self._red_r            = 0.0
        self._current_crop_row = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  VISION
    # ══════════════════════════════════════════════════════════════════════════

    def _get_vision(self, sim, pitch_deg: float = 0.0):
        """
        Returns (obs_left, obs_right, red_left, red_right).

        obs_* : fraction of forward-facing pixels in the upper zone that are
                bright green (= grass blade).  Range [0, 1].
        red_* : fraction of pixels that are strongly red (= dragonfly face).
        """
        images = sim.get_raw_vision(sim.fly.name)
        H, W   = images[0].shape[:2]

        # Shift detection zone based on pitch
        # pitch_deg > 0 → nose up (climbing) → horizon lower → look lower
        pitch_px  = int(np.clip(pitch_deg * 1.2, -H // 4, H // 4))
        cut_row   = int(np.clip(H * self.obs_cut_frac + pitch_px, H // 8, H // 2))
        self._current_crop_row = cut_row

        def grass_score(img, col_start, col_end):
            strip = img[:cut_row, col_start:col_end]
            r = strip[:, :, 0].astype(float)
            g = strip[:, :, 1].astype(float)
            b = strip[:, :, 2].astype(float)
            total = r + g + b + 1e-6
            mask = (
                (g / total > self.obs_green_thr) &
                (g > self.obs_abs_thr)           &
                (g > b * 1.2)                    &   # not sky-blue
                (total > 30)                         # not black border
            )
            return float(mask.mean())

        def red_score(img):
            r = img[:, :, 0].astype(float)
            g = img[:, :, 1].astype(float)
            b = img[:, :, 2].astype(float)
            return float(((r > 150) & (r > 2 * g) & (r > 2 * b)).mean())

        # Left eye  → forward zone = right half of image
        # Right eye → forward zone = left  half of image
        obs_l = grass_score(images[0], W // 2, W)
        obs_r = grass_score(images[1], 0,      W // 2)
        red_l = red_score(images[0])
        red_r = red_score(images[1])

        return obs_l, obs_r, red_l, red_r

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE AVOIDANCE LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _avoid_obstacles(self, obs_l: float, obs_r: float):
        """
        Returns (turn, should_avoid).

        turn > 0 → steer left  (obstacle on right)
        turn < 0 → steer right (obstacle on left)
        """
        diff = obs_l - obs_r

        if obs_l > self.obs_reflex_thr or obs_r > self.obs_reflex_thr:
            # Symmetric → can pass through
            if abs(diff) < self.obs_passthru_thr:
                print(f"  [VISION] PASS THROUGH  L={obs_l:.3f} R={obs_r:.3f}  |diff|={abs(diff):.3f} < {self.obs_passthru_thr}")
                return 0.0, False

            # Asymmetric → turn away from heavier side
            turn = -np.sign(diff) * self.obs_turn_mag
            print(f"  [VISION] AVOID  L={obs_l:.3f} R={obs_r:.3f}  diff={diff:+.3f}  turn={turn:+.1f}")
            return turn, True

        return 0.0, False

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAGONFLY
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_dragonfly(self):
        thr = 1e-3
        l, r = self._red_l, self._red_r
        if l < thr and r < thr:
            return False, 0
        if l > thr and r > thr:
            return True, 0   # front
        return True, (-1 if l > thr else +1)

    def _dragonfly_drives(self, side: int) -> np.ndarray:
        if side == -1:
            return np.array([2.0, 3.5])   # left  → turn right
        if side == +1:
            return np.array([3.5, 2.0])   # right → turn left
        return np.array([4.0, 4.0])        # front → run

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN STEP
    # ══════════════════════════════════════════════════════════════════════════

    def step(self, sim: MiniprojectSimulation):
        self._counter += 1
        fly_name = sim.fly.name

        # ── Sensors ──────────────────────────────────────────────────────────
        olfaction = sim.get_olfaction(fly_name)
        quat      = sim.get_body_rotations(fly_name)[0]

        odor_steer, bias = odor_attraction.odor_intensity_to_control_signal(
            olfaction, -self.attractive_gain
        )
        roll_comp, pitch_comp, pitch_deg, roll_deg = OLD_movement_correction.tilt_to_control_signal(
            quat, self.K_PITCH, self.K_ROLL, self.max_pitch_boost, self.max_roll_boost
        )

        # ── Vision (throttled) ───────────────────────────────────────────────
        if self._counter % self._vision_interval == 0:
            self._obs_l, self._obs_r, self._red_l, self._red_r = \
                self._get_vision(sim, pitch_deg)
            print(f"[VISION] obs_L={self._obs_l:.4f}  obs_R={self._obs_r:.4f}  "
                  f"red_L={self._red_l:.4f}  red_R={self._red_r:.4f}  "
                  f"pitch={pitch_deg:.1f}°  cut_row={self._current_crop_row}")

        # ── Dragonfly (priority 1) ────────────────────────────────────────────
        danger, df_side = self._detect_dragonfly()
        if danger:
            print(f"[DRAGONFLY] side={df_side}")
            drives = self._dragonfly_drives(df_side)
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion

        # ── Obstacle avoidance (priority 2) ──────────────────────────────────
        turn, is_reflex = self._avoid_obstacles(self._obs_l, self._obs_r)

        if is_reflex:
            self._avoid_hold     = self.avoid_hold_steps
            self._last_turn      = turn
        elif self._avoid_hold > 0:
            self._avoid_hold -= 1
            turn = self._last_turn
            is_reflex = True
            print(f"  [AVOID HOLD] {self._avoid_hold} steps left  turn={turn:+.1f}")

        if is_reflex:
            left_drive  = np.clip(1.0 - turn, 0.0, 4.0)
            right_drive = np.clip(1.0 + turn, 0.0, 4.0)

            # Still apply tilt corrections on top
            if abs(roll_deg) > self.max_roll_deg:
                left_drive  += roll_comp[0]
                right_drive += roll_comp[1]
            if abs(pitch_deg) > self.max_pitch_deg:
                left_drive  += pitch_comp[0] * self.tilt_gain
                right_drive += pitch_comp[1] * self.tilt_gain

            joint_angles, adhesion = self.turning_controller.step(
                np.array([left_drive, right_drive])
            )
            return joint_angles, adhesion

        # ── Initial alignment (priority 3) ───────────────────────────────────
        if not self._aligned:
            if abs(bias) < self.align_bias_thr:
                self._aligned = True
                self._align_fade = self._align_fade_steps
                print("[ALIGN] ✅ Aligned — starting fade")
            else:
                drives = (np.array([self.align_drive, 0.0]) if bias > 0
                        else np.array([0.0, self.align_drive]))
                if abs(roll_deg) > self.max_roll_deg:
                    print("! ROLL CORRECTION !")
                    drives += np.array([roll_comp[0], roll_comp[1]])
                if abs(pitch_deg) > self.max_pitch_deg:
                    drives += np.array([pitch_comp[0], pitch_comp[1]]) * self.tilt_gain
                drives = np.clip(drives, 0.0, 4.0)
                print(f"[ALIGN] bias={bias:+.3f}  roll={roll_deg:.1f}°  pitch={pitch_deg:.1f}°")
                joint_angles, adhesion = self.turning_controller.step(drives)
                return joint_angles, adhesion

        # Fade-out après alignement
        if self._align_fade > 0:
            t = self._align_fade / self._align_fade_steps
            
            # ✅ Utiliser le bias actuel, pas celui du moment de l'alignement
            align_drives = (np.array([self.align_drive, 0.0]) if bias > 0
                            else np.array([0.0, self.align_drive]))
            normal_drives = np.clip(odor_steer * self.speed_gain, 0.0, self.speed_gain)
            drives = (1 - t) * normal_drives + t * align_drives
            self._align_fade -= 1
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion
        # ── Normal odor tracking (priority 4) ────────────────────────────────
        intended = odor_steer.copy()

        if abs(roll_deg) > self.max_roll_deg:
            print("JSIADJOSDFoskdoksd")
            intended += roll_comp
        if abs(pitch_deg) > self.max_pitch_deg:
            intended += pitch_comp * self.tilt_gain

        drives = np.clip(intended * self.speed_gain, 0.0, self.speed_gain)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
