import numpy as np
from scipy.spatial.transform import Rotation
from miniproject.simulation import MiniprojectSimulation
from .odor_attraction import odor_intensity_to_control_signal


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND DETECTION — variance-based (no baseline needed)
# ───────────────────────────────────────────────────────────────────────────────
#  Wind makes the antennas fluctuate rapidly → high std over a sliding window.
#  No wind → antennas stable → low std.
#  This works regardless of the resting position (no baseline calibration).
#
#  TUNING GUIDE
# ───────────────────────────────────────────────────────────────────────────────
#  wind_var_thr       0.002–0.015  std above = wind detected
#  wind_detect_steps  20–60        consecutive steps to confirm wind active
#  ant_window         30–80        sliding window size for std computation
#
#  obs_turn_mag       2.0–5.0      turn magnitude without wind
#  obs_turn_mag_wind  1.0–3.0      turn magnitude with wind (gentler = stable)
#  avoid_hold_steps   30–100
#  avoid_hold_wind    10–40
#
#  K_PITCH/K_ROLL     0.03–0.10
#  max_pitch_boost    0.3–0.8
#  max_roll_boost     0.2–0.6
#  max_pitch_deg      5–20
#  max_roll_deg       3–15
# ═══════════════════════════════════════════════════════════════════════════════


def _tilt_correction(quat, k_pitch, k_roll, max_pitch_boost, max_roll_boost):
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch_deg, roll_deg, _ = rot.as_euler('xyz', degrees=True)

    pitch_scalar = max_pitch_boost * float(np.tanh(max(0.0, pitch_deg) * k_pitch))
    pitch_corr   = np.array([pitch_scalar, pitch_scalar])

    roll_scalar  = max_roll_boost * float(np.tanh(roll_deg * k_roll))
    roll_corr    = np.array([-roll_scalar, roll_scalar])

    return roll_corr, pitch_corr, pitch_deg, roll_deg


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Locomotion ────────────────────────────────────────────────────────
        self.speed_gain      = 1.0
        self.attractive_gain = 1000.0

        # ── Alignment ─────────────────────────────────────────────────────────
        self.align_bias_thr   = 0.10
        self.align_drive      = 1.0
        self.align_fade_steps = 20

        # ── Tilt correction ───────────────────────────────────────────────────
        self.K_PITCH         = 0.05
        self.K_ROLL          = 0.06
        self.max_pitch_boost = 0.50
        self.max_roll_boost  = 0.35
        self.max_pitch_deg   = 4.0
        self.max_roll_deg    = 6.0

        # ── Obstacle detection ────────────────────────────────────────────────
        self.obs_green_thr   = 0.40
        self.obs_abs_thr     = 120.0
        self.obs_cut_frac    = 0.33

        # ── Obstacle avoidance ────────────────────────────────────────────────
        self.obs_reflex_thr    = 0.014
        self.obs_passthru_thr  = 0.0
        self.obs_turn_mag      = 2.5    # without wind
        self.obs_turn_mag_wind = 1.0    # with wind (gentler → more stable)
        self.avoid_hold_steps  = 30
        self.avoid_hold_wind   = 10

        # ── Wind detection (variance-based) ───────────────────────────────────
        self.wind_var_thr      = 0.0004  # std above this = wind  (tune 0.002–0.015)
        self.wind_detect_steps = 30     # consecutive steps to confirm wind active
        self.ant_window        = 50     # sliding window size for std

        self._ant_history  = []         # sliding window of antenna values
        self._wind_count   = 0
        self._wind_active  = False

        # wind_tie / wind_strong still used inside _obstacle_turn
        self.wind_tie_thr    = 0.01
        self.wind_strong_thr = 0.05
        self._antenna_baseline = self._measure_baseline(sim)

        # ── Internal state ─────────────────────────────────────────────────────
        self._aligned      = False
        self._align_fade   = 0
        self._counter      = 0
        self._vision_every = 80
        self._obs_l        = 0.0
        self._obs_r        = 0.0
        self._red_l        = 0.0
        self._red_r        = 0.0
        self._avoid_hold   = 0
        self._last_turn    = 0.0
        self._crop_row     = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  WIND SENSING
    # ══════════════════════════════════════════════════════════════════════════

    def _measure_baseline(self, sim) -> float:
        """Keep for _get_wind_lateral direction signal."""
        try:
            ant      = sim.get_antenna_data(sim.fly.name)
            baseline = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            print(f"[WIND] antenna baseline = {baseline:.4f}")
            return baseline
        except Exception:
            return 0.0

    def _get_wind_lateral(self, sim) -> float:
        """
        Returns signed lateral wind signal (direction only, for Case A/B).
        Positive → wind from left, Negative → wind from right.
        """
        try:
            ant     = sim.get_antenna_data(sim.fly.name)
            current = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            return current - self._antenna_baseline
        except Exception:
            return 0.0

    def _update_wind_detection(self, sim):
        """
        Update sliding window std to detect wind presence.
        Wind → high antenna variance. No wind → stable antennas.
        Returns the current std (magnitude proxy).
        """
        try:
            ant = sim.get_antenna_data(sim.fly.name)
            val = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
        except Exception:
            return 0.0

        self._ant_history.append(val)
        if len(self._ant_history) > self.ant_window:
            self._ant_history.pop(0)

        if len(self._ant_history) < 10:
            return 0.0

        wind_var = float(np.std(self._ant_history))

        # Update wind active state
        prev = self._wind_active
        if wind_var > self.wind_var_thr:
            self._wind_count = min(self._wind_count + 1, self.wind_detect_steps)
        else:
            self._wind_count = max(self._wind_count - 1, 0)
        self._wind_active = (self._wind_count >= self.wind_detect_steps)

        if self._wind_active != prev:
            print(f"[WIND] {'ACTIVE ✓' if self._wind_active else 'INACTIVE'}"
                  f"  std={wind_var:.4f}")

        return wind_var

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE + WIND-AWARE TURN
    # ══════════════════════════════════════════════════════════════════════════

    def _obstacle_turn(self, sim):
        """Returns (turn, reflex, hold_steps)."""
        L, R  = self._obs_l, self._obs_r
        diff  = L - R

        if not ((L > self.obs_reflex_thr or R > self.obs_reflex_thr) and self._aligned):
            return 0.0, False, self.avoid_hold_steps

        if abs(diff) < self.obs_passthru_thr:
            print(f"  [OBS] GAP → pass through  L={L:.3f} R={R:.3f}")
            return 0.0, False, self.avoid_hold_steps

        wind   = self._get_wind_lateral(sim)
        strong = abs(wind) > self.wind_strong_thr and self._wind_active

        # Use gentler turn magnitude when wind is active
        base_mag = self.obs_turn_mag_wind if self._wind_active else self.obs_turn_mag
        print(f"MAGNITUDE : {base_mag}")
        intensity = max(L, R)
        turn_mag  = np.clip(base_mag * np.tanh(intensity * 10), 0.5, base_mag)
        print(f"MAGNITUDE AVOID={base_mag}")

        # Case B : obstacle centered + wind → choose tailwind direction
        if abs(diff) < self.wind_tie_thr and strong:
            turn      = np.sign(wind) * turn_mag
            direction = "left" if turn > 0 else "right"
            print(f"  [OBS+WIND] centered → chose {direction} (tailwind)  "
                  f"wind={wind:+.4f}  turn={turn:+.2f}")
            return turn, True, self.avoid_hold_steps

        # Case A : obstacle forces direction
        forced_turn = -np.sign(diff) * turn_mag

        if strong:
            wind_opposes = np.sign(forced_turn) != np.sign(wind)
            if wind_opposes:
                hold = self.avoid_hold_wind
                print(f"  [OBS+WIND] forced {'left' if forced_turn > 0 else 'right'} "
                      f"INTO wind → short hold={hold}  wind={wind:+.4f}")
            else:
                hold = self.avoid_hold_steps
                print(f"  [OBS+WIND] forced {'left' if forced_turn > 0 else 'right'} "
                      f"WITH wind → normal hold={hold}  wind={wind:+.4f}")
        else:
            hold = self.avoid_hold_steps
            print(f"  [OBS] AVOID  L={L:.3f} R={R:.3f}  diff={diff:+.3f}  turn={forced_turn:+.2f}")

        return forced_turn, True, hold

    # ══════════════════════════════════════════════════════════════════════════
    #  VISION
    # ══════════════════════════════════════════════════════════════════════════

    def _update_vision(self, sim, pitch_deg: float):
        images = sim.get_raw_vision(sim.fly.name)
        H, W   = images[0].shape[:2]

        pitch_px       = int(np.clip(pitch_deg * 1.2, -H // 4, H // 4))
        cut            = int(np.clip(H * self.obs_cut_frac + pitch_px, H // 8, H // 2))
        self._crop_row = cut

        def grass(img, c0, c1):
            s = img[:cut, c0:c1]
            r = s[:, :, 0].astype(float)
            g = s[:, :, 1].astype(float)
            b = s[:, :, 2].astype(float)
            t = r + g + b + 1e-6
            return float(((g / t > self.obs_green_thr) &
                          (g > self.obs_abs_thr)        &
                          (g > b * 1.2)                 &
                          (t > 30)).mean())

        def red(img):
            r = img[:, :, 0].astype(float)
            g = img[:, :, 1].astype(float)
            b = img[:, :, 2].astype(float)
            return float(((r > 150) & (r > 2*g) & (r > 2*b)).mean())

        self._obs_l = grass(images[0], W // 2, W)
        self._obs_r = grass(images[1], 0,      W // 2)
        self._red_l = red(images[0])
        self._red_r = red(images[1])

        print(f"  [VISION] obs L={self._obs_l:.4f} R={self._obs_r:.4f} | "
              f"red L={self._red_l:.4f} R={self._red_r:.4f} | "
              f"cut={cut}px  pitch={pitch_deg:.1f}°")

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAGONFLY
    # ══════════════════════════════════════════════════════════════════════════

    def _dragonfly(self):
        thr = 1e-3
        L, R = self._red_l, self._red_r
        if L < thr and R < thr: return False, 0
        if L > thr and R > thr: return True, 0
        return True, (-1 if L > thr else +1)

    def _dragonfly_drives(self, side):
        if side == -1: return np.array([3.0, 2.0])
        if side == +1: return np.array([2.0, 3.0])
        return np.array([4.0, 4.0])

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN STEP
    # ══════════════════════════════════════════════════════════════════════════

    def step(self, sim: MiniprojectSimulation):
        self._counter += 1

        # ── Sensors ───────────────────────────────────────────────────────────
        olfaction = sim.get_olfaction(sim.fly.name)
        quat      = sim.get_body_rotations(sim.fly.name)[0]

        odor_drives, bias = odor_intensity_to_control_signal(olfaction, -self.attractive_gain)
        roll_corr, pitch_corr, pitch_deg, roll_deg = _tilt_correction(
            quat, self.K_PITCH, self.K_ROLL,
            self.max_pitch_boost, self.max_roll_boost
        )

        # ── Vision (throttled) ────────────────────────────────────────────────
        if self._counter % self._vision_every == 0:
            self._update_vision(sim, pitch_deg)

        # ── Wind detection (variance-based, every step) ───────────────────────
        wind_std = self._update_wind_detection(sim)
        wind_lat = self._get_wind_lateral(sim)
        print(f"  [WIND] std={wind_std:.4f}  lateral={wind_lat:+.4f}"
              f"  active={self._wind_active}  count={self._wind_count}")

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 1 : Dragonfly
        # ─────────────────────────────────────────────────────────────────────
        danger, df_side = self._dragonfly()
        if danger:
            print(f"[DRAGONFLY] side={df_side}")
            joint_angles, adhesion = self.turning_controller.step(
                self._dragonfly_drives(df_side)
            )
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 2 : Obstacle avoidance (wind-aware)
        # ─────────────────────────────────────────────────────────────────────
        turn, reflex, dyn_hold = self._obstacle_turn(sim)

        if reflex:
            self._avoid_hold = dyn_hold
            self._last_turn  = turn
        elif self._avoid_hold > 0:
            self._avoid_hold -= 1
            turn   = self._last_turn
            reflex = True
            print(f"  [OBS HOLD] {self._avoid_hold} steps left  turn={turn:+.1f}")

        if reflex:
            ld = np.clip(1.0 - turn, 0.5, 4.0)
            rd = np.clip(1.0 + turn, 0.5, 4.0)
            if abs(roll_deg)  > self.max_roll_deg:
                ld += roll_corr[0]; rd += roll_corr[1]
            if abs(pitch_deg) > self.max_pitch_deg:
                ld += pitch_corr[0]; rd += pitch_corr[1]
            joint_angles, adhesion = self.turning_controller.step(np.array([ld, rd]))
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 3 : Alignment
        # ─────────────────────────────────────────────────────────────────────
        if not self._aligned:
            if abs(bias) < self.align_bias_thr:
                self._aligned    = True
                self._align_fade = self.align_fade_steps
                print(f"[ALIGN] The fly is aligned  bias={bias:+.3f}")
            else:
                drives = (np.array([self.align_drive, 0.0]) if bias > 0
                          else np.array([0.0, self.align_drive]))
                if abs(roll_deg)  > self.max_roll_deg:  drives += roll_corr
                if abs(pitch_deg) > self.max_pitch_deg: drives += pitch_corr
                drives = np.clip(drives, 0.0, 4.0)
                print(f"[ALIGN] bias={bias:+.3f}  roll={roll_deg:.1f}°  pitch={pitch_deg:.1f}°")
                joint_angles, adhesion = self.turning_controller.step(drives)
                return joint_angles, adhesion

        if self._align_fade > 0:
            t        = self._align_fade / self.align_fade_steps
            a_drives = (np.array([self.align_drive, 0.0]) if bias > 0
                        else np.array([0.0, self.align_drive]))
            n_drives = np.clip(odor_drives * self.speed_gain, 0.0, self.speed_gain)
            drives   = (1 - t) * n_drives + t * a_drives
            self._align_fade -= 1
            print(f"[FADE] t={t:.2f}  drives={np.round(drives, 2)}")
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 4 : Odor tracking + tilt
        # ─────────────────────────────────────────────────────────────────────
        drives = odor_drives.copy()

        if abs(roll_deg) > self.max_roll_deg:
            drives += roll_corr
            print(f"  [TILT] roll={roll_deg:.1f}°  corr={np.round(roll_corr,3)}")
        if abs(pitch_deg) > self.max_pitch_deg:
            drives += pitch_corr
            print(f"  [TILT] pitch={pitch_deg:.1f}°  corr={np.round(pitch_corr,3)}")

        drives = np.clip(drives * self.speed_gain, 0.0, self.speed_gain)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion