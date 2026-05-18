import numpy as np
from scipy.spatial.transform import Rotation
from miniproject.simulation import MiniprojectSimulation
from .odor_attraction import odor_intensity_to_control_signal


# ═══════════════════════════════════════════════════════════════════════════════
#  WIND-AWARE TURN STRATEGY
# ───────────────────────────────────────────────────────────────────────────────
#  When avoiding an obstacle, the fly chooses its turn direction to minimize
#  lateral wind exposure :
#
#  Case A — obstacle forces a direction (one side clearly blocked) :
#    → Keep the forced direction. If wind opposes, reduce hold_steps so
#      the fly finishes the turn faster and returns to stable forward walking.
#
#  Case B — obstacle roughly centered (|diff| < wind_tie_thr) :
#    → No forced direction. Choose the side that puts wind at the back.
#      wind from left  → prefer turning left  (wind becomes tailwind)
#      wind from right → prefer turning right (wind becomes tailwind)
#
#  Wind direction is read from antenna qpos[1] sum vs a baseline measured
#  at startup (no wind). Deviation from baseline = lateral wind signal.
#
#  TUNING GUIDE
# ───────────────────────────────────────────────────────────────────────────────
#  wind_tie_thr        0.02–0.10   |diff| below which turn dir is free to choose
#  wind_strong_thr     0.02–0.08   antenna deviation = "strong wind"
#  avoid_hold_steps    30–100      normal hold duration
#  avoid_hold_wind     10–40       reduced hold when turning into wind
#
#  obs_reflex_thr      0.01–0.05
#  obs_passthru_thr    0.00–0.12
#  obs_turn_mag        2.0–5.0
#
#  K_PITCH/K_ROLL      0.03–0.10
#  max_pitch_boost     0.3–0.8
#  max_roll_boost      0.2–0.6
#  max_pitch_deg       5–20
#  max_roll_deg        3–15
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
        self.align_drive      = 4.0
        self.align_fade_steps = 20

        # ── Tilt correction ───────────────────────────────────────────────────
        self.K_PITCH         = 0.05
        self.K_ROLL          = 0.06
        self.max_pitch_boost = 0.50
        self.max_roll_boost  = 0.35
        self.max_pitch_deg   = 10.0
        self.max_roll_deg    = 8.0

        # ── Obstacle detection ────────────────────────────────────────────────
        self.obs_green_thr   = 0.40
        self.obs_abs_thr     = 120.0
        self.obs_cut_frac    = 0.33

        # ── Obstacle avoidance ────────────────────────────────────────────────
        self.obs_reflex_thr   = 0.014 #0.018 #0.020
        self.obs_passthru_thr = 0.0
        self.obs_turn_mag     = 3.0
        self.avoid_hold_steps = 30 #60   # normal hold duration
        self.avoid_hold_wind  = 10 #20    # reduced hold when turning into wind

        # ── Wind-aware turn ───────────────────────────────────────────────────
        self.wind_tie_thr    = 0.03 #0.05  # |diff| below = obstacle centered = free choice
        self.wind_strong_thr = 0.02 #0.03  # antenna deviation = wind is strong

        # Measure antenna baseline at startup (no wind yet)
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
        """Measure antenna qpos[1] sum at startup (no wind)."""
        try:
            ant      = sim.get_antenna_data(sim.fly.name)
            baseline = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            print(f"[WIND] antenna baseline = {baseline:.4f}")
            return baseline
        except Exception:
            return -0.0023 #-0.069   # observed default without wind

    def _get_wind_lateral(self, sim) -> float:
        """
        Returns signed lateral wind signal from antenna deflection.
          Positive → wind from left  (fly pushed right)
          Negative → wind from right (fly pushed left)
          ~0       → no significant wind
        """
        try:
            ant     = sim.get_antenna_data(sim.fly.name)
            current = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            signal  = current - self._antenna_baseline
            return signal
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE + WIND-AWARE TURN
    # ══════════════════════════════════════════════════════════════════════════

    def _obstacle_turn(self, sim):
        """
        Returns (turn, reflex, hold_steps).

        Sign convention :
          turn > 0 → fly turns LEFT  (right drive faster)
          turn < 0 → fly turns RIGHT (left drive faster)

        Wind-aware logic :
          Case A (|diff| >= wind_tie_thr) : obstacle forces direction
            → keep forced direction
            → if wind opposes turn, shorten hold to reduce exposure time
          Case B (|diff| < wind_tie_thr) : obstacle centered, free choice
            → choose direction that puts wind at back
        """
        L, R  = self._obs_l, self._obs_r
        diff  = L - R   # + = more obstacle on left, - = more on right

        if not ((L > self.obs_reflex_thr or R > self.obs_reflex_thr) and self._aligned):
            return 0.0, False, self.avoid_hold_steps

        if abs(diff) < self.obs_passthru_thr:
            print(f"  [OBS] GAP → pass through  L={L:.3f} R={R:.3f}")
            return 0.0, False, self.avoid_hold_steps

        wind   = self._get_wind_lateral(sim)
        strong = abs(wind) > self.wind_strong_thr

        intensity = max(L, R)
        turn_mag  = np.clip(self.obs_turn_mag * np.tanh(intensity * 10), 0.5, self.obs_turn_mag)

        # ── Case B : obstacle centered → free to choose direction ─────────────
        if abs(diff) < self.wind_tie_thr and strong:
            # wind from left (wind+) → turn left (turn+) → wind becomes tailwind
            # wind from right (wind-) → turn right (turn-) → wind becomes tailwind
            turn      = np.sign(wind) * turn_mag
            hold      = self.avoid_hold_steps
            direction = "left" if turn > 0 else "right"
            print(f"  [OBS+WIND] centered → chose {direction} (tailwind)  "
                  f"wind={wind:+.4f}  turn={turn:+.2f}")
            return turn, True, hold

        # ── Case A : obstacle forces direction ────────────────────────────────
        # diff > 0 → obstacle left → must turn right → turn = -turn_mag
        forced_turn = -np.sign(diff) * turn_mag

        if strong:
            # Check if wind opposes the forced turn
            # turn > 0 (left) opposed by wind from right (wind < 0)
            # turn < 0 (right) opposed by wind from left (wind > 0)
            wind_opposes = np.sign(forced_turn) != np.sign(wind)
            if wind_opposes:
                hold = self.avoid_hold_wind   # shorter hold → less exposure
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
        if side == -1: return np.array([2.0, 3.5])
        if side == +1: return np.array([3.5, 2.0])
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
        

        # Dans step(), ajouter temporairement :
        wind = self._get_wind_lateral(sim)
        if abs(wind) > 0.001:
            print(f"  [WIND SIGNAL] {wind:+.5f}  (baseline={self._antenna_baseline:.4f})")
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
                print(f"[ALIGN] ✅ aligned  bias={bias:+.3f}")
            else:
                drives = (np.array([self.align_drive, 0.0]) if bias > 0
                          else np.array([0.0, self.align_drive]))
                if abs(roll_deg)  > self.max_roll_deg:  drives += roll_corr
                if abs(pitch_deg) > self.max_pitch_deg: drives += pitch_corr
            ###=================
                """ wind = self._get_wind_lateral(sim)
                if abs(wind) > self.wind_strong_thr:
                    # Booster le côté qui résiste au vent latéral
                    wind_boost = np.clip(abs(wind) * 5.0, 0.0, 0.5)
                    drives[0] += wind_boost if wind < 0 else 0.0  # vent de droite → boost gauche
                    drives[1] += wind_boost if wind > 0 else 0.0  # vent de gauche → boost droite
                    print(f"  [ALIGN WIND] wind={wind:+.4f}  boost={wind_boost:.3f}")

                drives = np.clip(drives, 0.0, 4.0)
                joint_angles, adhesion = self.turning_controller.step(drives)
                return joint_angles, adhesion """
            ###==================
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
    