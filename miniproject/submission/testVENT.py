import numpy as np
from scipy.spatial.transform import Rotation
from miniproject.simulation import MiniprojectSimulation
from .odor_attraction import odor_intensity_to_control_signal


# ═══════════════════════════════════════════════════════════════════════════════
#  STRATEGY OVERVIEW
# ───────────────────────────────────────────────────────────────────────────────
#
#  1. ALIGNMENT (once at start)
#  2. OBSTACLE AVOIDANCE (priority 2)
#  3. WIND PAUSE (priority 1.5 — between obstacle and alignment)
#     If the fly is turning AND detects strong wind via antennae →
#     stop completely (drives = [0.3, 0.3]) until wind calms down.
#     This prevents tipping during turns under crosswind.
#  4. ODOR + TILT (default)
#
#  TUNING
# ───────────────────────────────────────────────────────────────────────────────
#  wind_pause_thr    0.01–0.10   antenna signal above which wind is "strong"
#  wind_pause_drive  0.1–0.5     drive while waiting (low = more stable)
#  wind_calm_steps   50–200      steps of calm wind before resuming turn
#
#  obs_reflex_thr    0.01–0.05
#  obs_passthru_thr  0.00–0.12
#  obs_turn_mag      2.0–5.0
#  avoid_hold_steps  30–100
#
#  K_PITCH / K_ROLL  0.03–0.10
#  max_pitch_boost   0.3–0.8
#  max_roll_boost    0.2–0.6
#  max_pitch_deg     5–20
#  max_roll_deg      3–15
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
        self.speed_gain      = 2.0
        self.attractive_gain = 1000.0

        # ── Alignment ─────────────────────────────────────────────────────────
        self.align_bias_thr   = 0.10
        self.align_drive      = 2.0
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
        self.obs_reflex_thr   = 0.020
        self.obs_passthru_thr = 0.0
        self.obs_turn_mag     = 3.0
        self.avoid_hold_steps = 60

        # ── Wind pause ────────────────────────────────────────────────────────
        # If turning AND wind strong → pause until wind calms
        self.wind_pause_thr   = 0.09   # antenna |L-R| above = strong wind
        self.wind_pause_drive = 0.0 #0.3    # drive while paused (low = stable)
        self.wind_calm_steps  = 200  #80   # steps of calm before resuming
        self._antenna_baseline = self._measure_antenna_baseline(sim)

        # ── Internal state ─────────────────────────────────────────────────────
        self._aligned         = False
        self._align_fade      = 0
        self._counter         = 0
        self._vision_every    = 80
        self._obs_l           = 0.0
        self._obs_r           = 0.0
        self._red_l           = 0.0
        self._red_r           = 0.0
        self._avoid_hold      = 0
        self._last_turn       = 0.0
        self._crop_row        = 0
        self._wind_pause      = False   # currently paused for wind?
        self._wind_calm_count = 0       # steps of calm wind seen so far

    # ══════════════════════════════════════════════════════════════════════════
    #  WIND DETECTION (antenna-based)
    # ══════════════════════════════════════════════════════════════════════════


    def _measure_antenna_baseline(self, sim) -> float:
        """Mesure la somme qpos[1] au repos pour calibrer."""
        try:
            ant = sim.get_antenna_data(sim.fly.name)
            baseline = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            print(f"  [WIND] baseline antenne = {baseline:.4f}")
            return baseline
        except Exception:
            return -0.069   # valeur par défaut observée sans vent

    def _get_wind_strength(self, sim) -> float:
        """
        Retourne la déviation latérale des antennes par rapport au repos.
        Positif = vent de gauche, négatif = vent de droite.
        Magnitude = force du vent latéral.
        """
        try:
            ant    = sim.get_antenna_data(sim.fly.name)
            current = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            signal  = current - self._antenna_baseline   # déviation vs repos
            print(f"  [WIND] sum={current:.4f}  baseline={self._antenna_baseline:.4f}  signal={signal:+.4f}")
            return abs(signal)   # magnitude seulement pour la pause
        except Exception:
            return 0.0

    def _check_wind_pause(self, sim, is_turning: bool) -> bool:
        """
        Returns True if the fly should pause due to strong wind during a turn.

        Logic:
          - If turning AND wind strong → enter pause, reset calm counter
          - While paused AND wind still strong → stay paused
          - While paused AND wind calm → increment calm counter
          - When calm counter reaches wind_calm_steps → exit pause
        """
        wind_strength = self._get_wind_strength(sim)
        print(f"wind_strength : {wind_strength}")
        wind_strong   = wind_strength > self.wind_pause_thr

        if not self._wind_pause:
            # Not paused — check if we should start
            if is_turning and wind_strong:
                self._wind_pause      = True
                self._wind_calm_count = 0
                #print(f"  [WIND PAUSE] START  wind={wind_strength:.4f} > {self.wind_pause_thr}")
            return self._wind_pause

        # Already paused — check if wind has calmed
        if wind_strong:
            self._wind_calm_count = 0   # reset calm counter on gust
            print(f"  [WIND PAUSE] waiting  wind={wind_strength:.4f}  calm={self._wind_calm_count}/{self.wind_calm_steps}")
        else:
            self._wind_calm_count += 1
            print(f"  [WIND PAUSE] calming  wind={wind_strength:.4f}  calm={self._wind_calm_count}/{self.wind_calm_steps}")
            if self._wind_calm_count >= self.wind_calm_steps:
                self._wind_pause = False
                print(f"  [WIND PAUSE] RESUME")
                self._aligned=False #we restart the alignement

        return self._wind_pause

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

        #print(f"  [VISION] obs L={self._obs_l:.4f} R={self._obs_r:.4f} | "
              #f"red L={self._red_l:.4f} R={self._red_r:.4f} | "
              #f"cut={cut}px  pitch={pitch_deg:.1f}°")

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE
    # ══════════════════════════════════════════════════════════════════════════

    def _obstacle_turn(self):
        L, R  = self._obs_l, self._obs_r
        diff  = L - R

        if (L > self.obs_reflex_thr or R > self.obs_reflex_thr) and self._aligned:
            if abs(diff) < self.obs_passthru_thr:
                #print(f"  [OBS] GAP → pass through  L={L:.3f} R={R:.3f}")
                return 0.0, False
            intensity = max(L, R)
            turn_mag  = np.clip(self.obs_turn_mag * np.tanh(intensity * 10), 0.5, self.obs_turn_mag)
            turn      = -np.sign(diff) * turn_mag
            #print(f"  [OBS] AVOID  L={L:.3f} R={R:.3f}  diff={diff:+.3f}  turn={turn:+.1f}")
            return turn, True

        return 0.0, False

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

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 1 : Dragonfly
        # ─────────────────────────────────────────────────────────────────────
        danger, df_side = self._dragonfly()
        if danger:
            #print(f"[DRAGONFLY] side={df_side}")
            joint_angles, adhesion = self.turning_controller.step(
                self._dragonfly_drives(df_side)
            )
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 2 : Obstacle avoidance
        # ─────────────────────────────────────────────────────────────────────
        turn, reflex = self._obstacle_turn()

        if reflex:
            self._avoid_hold = self.avoid_hold_steps
            self._last_turn  = turn
        elif self._avoid_hold > 0:
            self._avoid_hold -= 1
            turn   = self._last_turn
            reflex = True
            #print(f"  [OBS HOLD] {self._avoid_hold} steps left  turn={turn:+.1f}")

        # Is the fly currently turning?
        is_turning = reflex or (self._align_fade > 0) or (abs(bias) > 0.3)

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 2.5 : Wind pause (only during turns)
        # ─────────────────────────────────────────────────────────────────────
        should_pause = self._check_wind_pause(sim, is_turning)
        if should_pause:
            # Stand still with low symmetric drive — stable low CG
            drives = np.array([self.wind_pause_drive, self.wind_pause_drive])
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion

        if reflex:
            ld = np.clip(1.0 - turn, 0.0, 4.0)
            rd = np.clip(1.0 + turn, 0.0, 4.0)
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
                #print(f"[ALIGN] ✅ aligned  bias={bias:+.3f}")
            else:
                drives = (np.array([self.align_drive, 0.0]) if bias > 0
                          else np.array([0.0, self.align_drive]))
                if abs(roll_deg)  > self.max_roll_deg:  drives += roll_corr
                if abs(pitch_deg) > self.max_pitch_deg: drives += pitch_corr
                drives = np.clip(drives, 0.0, 4.0)
                #print(f"[ALIGN] bias={bias:+.3f}  roll={roll_deg:.1f}°  pitch={pitch_deg:.1f}°")
                joint_angles, adhesion = self.turning_controller.step(drives)
                return joint_angles, adhesion

        if self._align_fade > 0:
            t        = self._align_fade / self.align_fade_steps
            a_drives = (np.array([self.align_drive, 0.0]) if bias > 0
                        else np.array([0.0, self.align_drive]))
            n_drives = np.clip(odor_drives * self.speed_gain, 0.0, self.speed_gain)
            drives   = (1 - t) * n_drives + t * a_drives
            self._align_fade -= 1
            #print(f"[FADE] t={t:.2f}  drives={np.round(drives, 2)}")
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 4 : Odor tracking + tilt
        # ─────────────────────────────────────────────────────────────────────
        drives = odor_drives.copy()

        if abs(roll_deg) > self.max_roll_deg:
            drives += roll_corr
            #print(f"  [TILT] roll={roll_deg:.1f}°  corr={np.round(roll_corr,3)}")
        if abs(pitch_deg) > self.max_pitch_deg:
            drives += pitch_corr
            #print(f"  [TILT] pitch={pitch_deg:.1f}°  corr={np.round(pitch_corr,3)}")

        drives = np.clip(drives * self.speed_gain, 0.0, self.speed_gain)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion