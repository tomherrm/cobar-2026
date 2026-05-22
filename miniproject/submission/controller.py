#===========================================================================
# BIOENG-456 Controlling Behavior in Animals and Robots - Miniproject
# Tom Herrmann  / Alexandros Dellios  / Flavio Caroli 
#============================================================================

import numpy as np
from scipy.spatial.transform import Rotation
from miniproject.simulation import MiniprojectSimulation


def _tilt_correction(quat, k_pitch, k_roll, max_pitch_boost, max_roll_boost):
    """ Returns the roll and pitch correction that the fly needs to stay stable. 
        Also returns the pitch and roll measured (in degrees). """

    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch_deg, roll_deg, _ = rot.as_euler('xyz', degrees=True)

    pitch_scalar = max_pitch_boost * float(np.tanh(max(0.0, pitch_deg) * k_pitch))
    pitch_corr   = np.array([pitch_scalar, pitch_scalar])

    roll_scalar  = max_roll_boost * float(np.tanh(roll_deg * k_roll))
    roll_corr    = np.array([-roll_scalar, roll_scalar])

    return roll_corr, pitch_corr, pitch_deg, roll_deg



def odor_intensity_to_control_signal(
    odor_intensities,
    attractive_gain
):
    """(Adapted from the exercises)
    Convert odor sensor readings to a turning control signal.

    Parameters
    ----------
    odor_intensities : np.ndarray
        Odor intensities from the four sensors, shape ``(4, n_odor_dims)``.
    attractive_gain : float
        Gain applied to the attractive odor dimension.
    
    Returns
    -------
    np.ndarray
        Control signal of shape ``(2,)`` for left and right descending drive.
    """

    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    
    attractive_bias = (
        attractive_gain
        * (attractive_intensities[0] - attractive_intensities[1])
        / attractive_intensities.mean()
        if attractive_intensities.mean() != 0
        else 0
    )
    
    effective_bias_norm = np.tanh(attractive_bias**2) * np.sign(attractive_bias)
    assert np.sign(effective_bias_norm) == np.sign(attractive_bias)

    control_signal = np.ones(2)
    side_to_modulate = int(effective_bias_norm > 0) 
    modulation_amount = np.abs(effective_bias_norm) * 0.99
    control_signal[side_to_modulate] = 1-modulation_amount # This was modified from the exercises in order to achieve a more sharp turns. 
    return control_signal, effective_bias_norm

class Controller:
    """ This class represents our controller handling the fly path decision making. """

    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)

        # ── Locomotion ────────────────────────────────────────────────────────
        self.speed_gain      = 1.0
        self.attractive_gain = 1000.0 # Attractive gain towards the odor source.

        # ── Alignment ─────────────────────────────────────────────────────────
        self.align_bias_thr   = 0.30 # Threshold after which we can say that the fly is aligned with the goal. 
        self.align_drive      = 1.0 # Kept low in order to withstand the wind when alignement.
        self.align_fade_steps = 20 # To end smoothly the alignement procedure.

        # ── Tilt correction ───────────────────────────────────────────────────
        self.K_PITCH         = 0.05
        self.K_ROLL          = 0.06
        self.max_pitch_boost = 0.50
        self.max_roll_boost  = 0.35
        self.max_pitch_deg   = 4.0 # Maximum pitch angle allowed before applying correction.
        self.max_roll_deg    = 6.0 # Maximum roll angle allowed before applying correction.

        # ── Obstacle detection ────────────────────────────────────────────────
        self.obs_green_thr   = 0.40 
        self.obs_abs_thr     = 120.0
        self.obs_cut_frac    = 0.33

        # ── Obstacle avoidance ────────────────────────────────────────────────
        self.obs_reflex_thr   = 0.014 # Threshold above which obstacle detection happens.
        self.obs_turn_mag     = 1.926 #tuned
        self.avoid_hold_steps = 30 # Number of steps during which we keep turning during avoidance.
        self.avoid_hold_wind  = 10 # Same as before but reduced in case of defavorable wind direction.
        self.hill_roll_thr    = 6.7 # Roll above which hill correction activates.
        self.pitch_avoid_threshold = 4.0 

        # ── Wind-aware turn ───────────────────────────────────────────────────
        self.wind_tie_thr    = 0.01 
        self.wind_strong_thr = 0.05 # Threshold above which wind is considered "strong".
        self._antenna_baseline = self._measure_baseline(sim)

        # ── Internal states ─────────────────────────────────────────────────────
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


    # ══════════════════════════════════════════════════════════════════════════
    #  WIND SENSING
    # ══════════════════════════════════════════════════════════════════════════

    def _measure_baseline(self, sim) -> float:
        try:
            ant      = sim.get_antenna_data(sim.fly.name)
            baseline = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            return baseline
        except Exception:
            return -0.0023

    def _get_wind_lateral(self, sim) -> float:
        try:
            ant     = sim.get_antenna_data(sim.fly.name)
            current = float(ant['l']['qpos'][1]) + float(ant['r']['qpos'][1])
            return current - self._antenna_baseline
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE AVOIDANCE + WIND/HILL-AWARE TURN
    # ══════════════════════════════════════════════════════════════════════════

    def _obstacle_turn(self, sim, roll_deg: float = 0.0, pitch_deg: float =0.0):
        """
        Returns (turn, reflex, hold_steps).

        turn > 0 → fly turns LEFT 
        turn < 0 → fly turns RIGHT 

        Hill correction :
          If roll is significant, override forced direction toward uphill side.
          roll > 0 → right side down → LEFT is safer
          roll < 0 → left side down  → RIGHT is safer
        """
        L, R  = self._obs_l, self._obs_r
        diff  = L - R

        if not ((L > self.obs_reflex_thr or R > self.obs_reflex_thr) and self._aligned):
            return 0.0, False, self.avoid_hold_steps

    
        wind   = self._get_wind_lateral(sim)
        strong = abs(wind) > self.wind_strong_thr 

        intensity = max(L, R)
        turn_mag  = np.clip(self.obs_turn_mag * np.tanh(intensity * 10), 0.5, self.obs_turn_mag)

        # Case A : obstacle centered -> choose wind-favorable direction
        if abs(diff) < self.wind_tie_thr and strong:
            turn      = np.sign(wind) * turn_mag
            return turn, True, self.avoid_hold_steps

        # Case B : obstacle forces direction
        forced_turn = -np.sign(diff) * turn_mag

        if strong:
            wind_opposes = np.sign(forced_turn) != np.sign(wind)
            if wind_opposes:
                hold = self.avoid_hold_wind
            else:
                hold = self.avoid_hold_steps
                
        else:
            hold = self.avoid_hold_steps
           

        # Case C : Hill correction -> if tilted, prefer uphill direction
        if abs(roll_deg) > self.hill_roll_thr and abs(pitch_deg)>self.pitch_avoid_threshold:
            safe_sign = -np.sign(roll_deg)   
            if np.sign(forced_turn) != safe_sign:
                forced_turn = safe_sign * turn_mag

        return forced_turn, True, hold

    # ══════════════════════════════════════════════════════════════════════════
    #  VISION
    # ══════════════════════════════════════════════════════════════════════════

    def _update_vision(self, sim, pitch_deg: float):
        images = sim.get_raw_vision(sim.fly.name)
        H, W   = images[0].shape[:2]

        pitch_px       = int(np.clip(pitch_deg * 1.2, -H // 4, H // 4))
        cut            = int(np.clip(H * self.obs_cut_frac + pitch_px, H // 8, H // 2))

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


    # ══════════════════════════════════════════════════════════════════════════
    #  DRAGONFLY
    # ══════════════════════════════════════════════════════════════════════════

    def _dragonfly(self):
        """ Returns if a dragonfly is detected and on which side it is.  """
        thr = 1e-4 # This was experimentaly tuned.
        L, R = self._red_l, self._red_r
        if L < thr and R < thr: return False, 0
        if L > thr and R > thr: return True, 0
        return True, (-1 if L > thr else +1)

    def _dragonfly_drives(self, side):
        """ Returns the according drive to escape the dragonfly. """
        if side == -1: return np.array([3.3, 2.0])
        if side == +1: return np.array([2.0, 3.3])
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
        
        if self._counter % self._vision_every == 0: # We get the vision data once every "self._vision_every" steps.
            self._update_vision(sim, pitch_deg)

        #======================================================================
        #   HIERARCHICAL DECISION TREE
        #======================================================================
        
        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 1 : Dragonfly
        # ─────────────────────────────────────────────────────────────────────
        danger, df_side = self._dragonfly()

        if danger:
            joint_angles, adhesion = self.turning_controller.step(
                self._dragonfly_drives(df_side)
            )
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 2 : Obstacle avoidance (wind + hill aware)
        # ─────────────────────────────────────────────────────────────────────
        turn, reflex, dyn_hold = self._obstacle_turn(sim, roll_deg, pitch_deg)

        if reflex:
            self._avoid_hold = dyn_hold
            self._last_turn  = turn
        elif self._avoid_hold > 0:
            self._avoid_hold -= 1
            turn   = self._last_turn
            reflex = True

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
                print(f"\n The fly is aligned with the goal.")
            else:
                drives = (np.array([self.align_drive, 0.0]) if bias > 0
                          else np.array([0.0, self.align_drive]))
                
                if abs(roll_deg)  > self.max_roll_deg:  drives += roll_corr
                if abs(pitch_deg) > self.max_pitch_deg: drives += pitch_corr
            
                drives = np.clip(drives, 0.0, 4.0)
                joint_angles, adhesion = self.turning_controller.step(drives)
                return joint_angles, adhesion

        if self._align_fade > 0: # We set a cooldown where the fly slows down to avoid an abrupt change of drives.
            t = self._align_fade / self.align_fade_steps
            a_drives = (np.array([self.align_drive, 0.0]) if bias > 0
                        else np.array([0.0, self.align_drive]))
            n_drives = np.clip(odor_drives * self.speed_gain, 0.0, self.speed_gain)
            drives   = (1 - t) * n_drives + t * a_drives
            self._align_fade -= 1
            joint_angles, adhesion = self.turning_controller.step(drives)
            return joint_angles, adhesion

        # ─────────────────────────────────────────────────────────────────────
        #  PRIORITY 4 : Odor tracking + tilt
        # ─────────────────────────────────────────────────────────────────────
        drives = odor_drives.copy()

        if abs(roll_deg) > self.max_roll_deg:
            drives += roll_corr
        if abs(pitch_deg) > self.max_pitch_deg:
            drives += pitch_corr

        drives = np.clip(drives * self.speed_gain, 0.0, self.speed_gain)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion