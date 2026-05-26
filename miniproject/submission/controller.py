"""Hierarchical sensorimotor controller with active-sensing state machine.

Architecture (single controller for levels 0-4):

    NAVIGATE  ── stuck + central blade(s) ──>  SCAN
       ^                                         │
       │                                         │ central forward clear
       │ commit done (moved >= commit_min_disp                   ▼
       │  or commit timeout or blade-wall hit)              COMMIT (drive forward
       └──────────────────────────────────────  in chosen heading)

Universal safety overrides (apply in every mode):
    severe tilt (pitch>60 or |roll|>65)  ── emergency backup, return to NAVIGATE
    moderate tilt during NAVIGATE        ── blend in pitch/roll compensation
    dragonfly looming (level 4 only)     ── full evasion override

Vision feature: per-column longest consecutive dark run on the forward
ommatidia band (`vertical_run_features`). Discriminates blades from ground
texture much better than mean dark fraction.
"""
import numpy as np
from collections import deque
from enum import Enum

from scipy.spatial.transform import Rotation

from miniproject.simulation import MiniprojectSimulation
from flygym.vision.retina import Retina

from . import odor_attraction
from . import movement_correction


class Mode(Enum):
    NAVIGATE = "navigate"
    SCAN = "scan"
    COMMIT = "commit"
    BACKUP = "backup"


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.retina = Retina()
        self.enable_dragonfly = getattr(sim, "enable_dragonfly", False)

        # === Tilt control (kept from prior controller) ===
        # attractive_gain raised 50 -> 200: stronger odor pull cuts open-terrain
        # traversal time (~50k steps for level-0 seed 1 -> aim for <30k).
        self.attractive_gain = 200
        self.K_PITCH = 10
        self.K_ROLL = 50
        self.max_pitch_boost = 0.5
        self.max_roll_boost = 0.3
        self.max_pitch = 20
        self.max_roll = 40

        # === Vertical-run feature parameters ===
        # Kept only so the ommatidia overlay in preview_debug.py keeps rendering
        # the dark-mask. The vertical-run feature is NOT used for decisions
        # because the yellow/pale ommatidia checkerboard caps consecutive dark
        # runs at ~4 even on a fully blade-covered column. Blade detection
        # runs on raw RGB (see raw_vision_* parameters below).
        self.dark_threshold = 0.15
        self.min_blade_run = 6

        # === Raw RGB blade detector parameters ===
        # green pixel = G > R+green_diff AND G > B+green_diff AND G > green_min
        # Per-column count of greens in the very TOP slice of the eye image is
        # an unambiguous blade-tip signal: ground texture and hill geometry
        # cannot reach the top 20% of the visual field, so any green there is
        # a blade rising above the horizon. This eliminates the false-positive
        # floor that came with top_band=0.5 (where hill ground tilted into the
        # band and triggered constant centering noise).
        self.rgb_top_band = 0.30
        self.rgb_green_count_threshold = 20
        self.rgb_green_diff = 15
        self.rgb_green_min = 60
        # Central forward fraction of each eye image (W=450 typical).
        # 0.22 ≈ central 100 cols ≈ "directly ahead".
        self.rgb_central_fraction = 0.22

        # === NAVIGATE parameters ===
        # navigate_speed raised 1.2 -> 1.8: this was lowered during cluster
        # tuning to make wedging less violent, but the LPLC1 modulator now
        # auto-slows in clusters (clutter_factor drops to 0.3 with blades),
        # so open-terrain speed can be high without harming level 2.
        self.navigate_speed = 1.8
        self.navigate_centering_gain = 1.5
        self.navigate_centering_clip = 0.6  # cap centering so odor stays dominant

        # === Reflex turn (DISABLED — empirically destabilizes the fly) ===
        # Tom's max-magnitude reflex was tested with threshold=5 at level 2
        # and made things worse: 27% of frames had saturated drive, hard
        # turns built up roll rapidly, fly flipped at ~14k steps (vs 80k
        # without reflex). The CPG can't sustain saturated differential
        # drive without losing balance. Threshold set to a value
        # n_central can't reach (>>100) so the block is effectively dead.
        # Keep code path so we can revisit later.
        self.reflex_n_central_threshold = 9999
        self.reflex_drive_magnitude = 3.0
        self.reflex_speed_factor = 1.0
        self.reflex_min_steps = 20
        self._reflex_remaining = 0
        self._reflex_sign = 0

        # === LPLC1 speed modulation (Tanaka & Clark 2022) ===
        # Real flies decelerate when obstacles are visible during walking
        # (LPLC1 visual projection neurons → slowing, not turning). We were
        # crashing into clusters at full speed because nothing slowed us down.
        # speed_multiplier = 1 in open terrain → 0.3 surrounded by blades.
        self.lplc1_full_slow_at_n_blades = 80
        self.lplc1_min_speed_factor = 0.3

        # === Wind compensation (level 3) ===
        # Antennae get pushed by wind (Drosophila Johnston's organ analog).
        # We compute the lateral component of antenna deflection in the fly's
        # body frame: positive = wind pushing antennae rightward (so wind
        # comes from the LEFT), negative = wind from right. Counter-steer
        # additively into the wind to maintain heading toward odor source.
        self.wind_gain = 1.5          # gain on body-frame antenna lateral deflection
        self.wind_correction_clip = 0.5  # cap so wind never overwhelms odor

        # === Campaniform sensilla — wall vs hill discriminator ===
        # Real flies tell wall from hill by force DIRECTION on legs (Tuthill &
        # Wilson 2016). MuJoCo exposes the equivalent: per-contact 3D force
        # vectors via sim.get_external_force. We project total contact force
        # into the fly's body frame; fore-aft component (-x) tells us:
        #   hill climbing  → +x (ground reaction helps fly forward)
        #   wall contact   → -x (wall pushes fly back)
        # Used to gate the severe-pitch BACKUP trigger so hills don't fire it.
        self.wall_force_threshold = 5.0  # body-frame -x force magnitude → wall

        # === SCAN parameters ===
        # Tight CCW yaw-curve. Drive sign alternates between consecutive scans
        # so we don't keep trying the same direction in a tight cluster.
        self.scan_drive_magnitude = 0.4
        self.scan_speed = 0.8
        self.scan_min_steps = 30      # debounce: don't exit before this
        self.scan_max_steps = 300     # timeout fallback
        self.scan_clear_count = 0     # exit when this many central cols (or fewer) carry blades
        # Central forward columns of the rectified band (W=31). 11..19 inclusive
        # = central 9 cols ~= 30% of horizontal field of view. "Dead ahead".
        self.central_cols = slice(11, 20)

        # === COMMIT parameters ===
        self.commit_drive = np.array([1.0, 1.0])
        self.commit_speed = 1.2
        self.commit_max_steps = 600
        self.commit_min_disp = 1.0      # was 2.0; smaller commit steps so the
                                         # fly can't overshoot banana as far
                                         # (matters near the goal where it
                                         # was orbiting at min_dist 6.4)
        self.commit_yaw_gain = 1.0 / 30.0   # deg of error -> drive correction
        self.commit_yaw_correction_max = 0.4
        self.commit_pitch_abort = 35.0  # walked into a wall: bail out (-> BACKUP)
        # COMMIT-odor blend was tested 09/05 night and regressed L3 s1 to
        # FLIP at 12k (was TIMEOUT 100k min_dist 6.4). The orbit-at-6.4
        # behavior comes from SCAN/COMMIT *cycles* approaching from varying
        # angles, not a single locked commit; adding odor blend disrupted
        # those cycles. Kept variable as 0 (no blend) for future revisit.
        self.commit_odor_blend_max = 0.0

        # === BACKUP parameters ===
        # Stateful escape from a wedged contact. Naive backup cannot disengage
        # when physics pins the fly against a blade — we need MAX-magnitude
        # backward drive PLUS yaw, AND we must try both rotational directions
        # within a single backup so neither yaw direction can dead-end us.
        # Layout: phase 1 = pure straight backup at max; phase 2+ = back+yaw
        # alternating direction every backup_yaw_period steps.
        self.backup_duration = 150
        self.backup_phase1_steps = 30    # pure straight backup
        self.backup_yaw_period = 30      # flip yaw direction every N steps in phase 2
        self.backup_speed = 1.0          # full speed
        self.backup_safe_pitch = 25.0
        self.backup_safe_roll = 25.0
        self.backup_min_steps = 30       # must run at least this long even if tilt drops
        # Adhesion release during BACKUP — GRADED by tilt severity.
        # Binary gating fails in the dead zone (pitch 40-55): adhesion stays
        # ON, fly oscillates at that pitch indefinitely, BACKUP exit condition
        # (pitch<25) never triggers, eventually drifts up to 68° and cascades.
        # Smooth ramp instead:
        #   tilt < release_low_tilt  → multiplier 0   (full release: slip)
        #   tilt > release_high_tilt → multiplier 1   (full keep: brake fall)
        #   in between               → linear interp  (graded grip)
        # The dead zone gets PARTIAL grip: enough to slow the fall, not so
        # much that the fly is glued in place.
        self.backup_release_adhesion = True
        self.backup_release_low_tilt = 30.0
        self.backup_release_high_tilt = 50.0

        # === Goal mode (close-in approach) ===
        # When odor signal is strong, the fly is close to the banana. Lock
        # control to NAVIGATE so the odor gradient drives the final approach
        # directly — SCAN/COMMIT locks the yaw onto a heading that walks
        # PAST the banana (observed empirically: L3 s1 reached min dist 6.4
        # but COMMIT-locked drive [1.22, 1.18] kept walking past the goal).
        # BACKUP and severe-tilt still fire normally (safety overrides goal).
        self.goal_odor_threshold = 1.0e-6
        self._last_odor_mean = 0.0

        # === Stuck detection (NAVIGATE -> SCAN trigger) ===
        # Trigger requires BOTH displacement-stuck AND visible blades ahead.
        # Originally was disp-only because the ommatidia feature was broken
        # (always returned n_central=0). With RGB now working, the AND-check
        # is restored — without it, the fly spuriously enters SCAN in open
        # terrain near the banana, COMMIT then walks 2u in an arbitrary
        # direction past the goal. Empirical 09/05: L3 s1 orbited at min
        # dist 6.4 because of these spurious open-terrain SCAN triggers.
        self._pos_history = deque(maxlen=500)
        self.stuck_threshold = 0.5
        self.stuck_min_central_blades = 2

        # === Severe tilt thresholds (universal safety) ===
        # severe_pitch is GATED by the wall-force discriminator (cue 1) so
        # hills don't false-trigger BACKUP. The earlier severe_pitch_hard=60
        # ungated fallback was tested and removed: on a steep hill (no wall),
        # firing BACKUP releases adhesion and the fly slides downhill and
        # flips faster than without intervention. So we accept that pure
        # backward-tipping on hills isn't recovered — that mode is rare.
        self.severe_pitch = 30.0
        self.severe_roll = 50.0

        # === Dragonfly detection (level 4) ===
        # Adopted from Tom's `origin/tom` branch: the dragonfly is RED in this
        # simulation, not dark. The previous `detect_looming` (dark-pixel
        # growth) was structurally wrong. Red color discriminator: a pixel
        # passes when R > 150 AND R > 2G AND R > 2B. Per-eye fraction above
        # threshold = "this eye sees a dragonfly."
        self.dragon_red_threshold = 1e-3
        # Legacy params kept only because debug_info references them:
        self.prev_dragon_areas = np.zeros(2)
        self.dragon_pixel_threshold = 80
        self.dragon_growth_threshold = 0.05
        self.dragon_proximity_threshold = 0.30

        # === State ===
        self.mode = Mode.NAVIGATE
        self._mode_step = 0
        self._mode_start_pos = None
        self._scan_best_yaw = None
        self._scan_best_score = float("inf")
        self._scan_dir_sign = 1.0     # alternates each SCAN entry
        self._backup_dir_sign = 1.0   # alternates each BACKUP entry (yaw direction)
        self._commit_target_yaw = None

        # === Death log (kept for diagnostics) ===
        self._history = deque(maxlen=15000)
        self.debug_info = {}

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _wrap_deg(x):
        """Wrap an angle to [-180, 180]."""
        return (float(x) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _yaw_from_quat(quat):
        """Z-axis rotation in degrees, same convention as `tilt_to_control_signal`."""
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        _, _, yaw = rot.as_euler("xyz", degrees=True)
        return float(yaw)

    # ----------------------------------------------------------- perception

    def _perceive(self, raw_vision):
        """Blade-detection features from raw vision — the perception SEAM.

        The base controller uses RGB green-segmentation in the top image band.
        The CNNController subclass overrides this method with a trained
        segmentation CNN; because it returns the same feature contract on the
        same scale, the state machine downstream is identical either way.

        Returns a dict:
          col_blade  (2, W)  per-column "blade present" for L/R eye
          top_count  (2, W)  per-column top-band green count
          n_central  int     central-FOV columns carrying a blade
          L_blades   float   blade-column count, left eye
          R_blades   float   blade-column count, right eye
        """
        col_blade, top_count = movement_correction.raw_vision_features(
            raw_vision,
            top_band=self.rgb_top_band,
            green_count_threshold=self.rgb_green_count_threshold,
            green_diff=self.rgb_green_diff,
            green_min=self.rgb_green_min,
        )
        W = col_blade.shape[-1]
        half_band = max(1, int(W * self.rgb_central_fraction / 2))
        c_lo = W // 2 - half_band
        c_hi = W // 2 + half_band
        n_central = int(col_blade[:, c_lo:c_hi].any(axis=0).sum())
        return {
            "col_blade": col_blade,
            "top_count": top_count,
            "n_central": n_central,
            "L_blades": float(col_blade[0].sum()),
            "R_blades": float(col_blade[1].sum()),
        }

    def _navigate_obstacle_avoidance(self, perc):
        """Extra NAVIGATE steering from object-level perception — a SEAM.

        The base controller has only blade-presence counts, so this is a
        no-op. CNNController overrides it: when the segmentation CNN reports a
        close blade dead ahead, it steers toward the clearest gap and slows
        down — active avoidance *before* contact, rather than the base
        controller's gentle blade-asymmetry centering that still walks the
        fly into the wedge.

        Returns (avoid_drive (2,), speed_factor): avoid_drive is ADDED to the
        NAVIGATE intended drive, speed_factor is MULTIPLIED into the speed.
        """
        return np.zeros(2), 1.0

    # ------------------------------------------------------ mode transitions

    def _enter_navigate(self, pos):
        self.mode = Mode.NAVIGATE
        self._mode_step = 0
        self._mode_start_pos = np.asarray(pos, dtype=float).copy()
        self._scan_best_yaw = None
        self._scan_best_score = float("inf")
        self._commit_target_yaw = None
        # Don't immediately re-trigger from pre-commit positions.
        self._pos_history.clear()

    def _enter_scan(self, pos, yaw):
        self.mode = Mode.SCAN
        self._mode_step = 0
        self._mode_start_pos = np.asarray(pos, dtype=float).copy()
        self._scan_best_yaw = yaw
        self._scan_best_score = float("inf")
        # Alternate scan direction each entry so consecutive failed scans
        # don't keep curving the same way through the cluster.
        self._scan_dir_sign *= -1.0

    def _enter_commit(self, pos, target_yaw):
        self.mode = Mode.COMMIT
        self._mode_step = 0
        self._mode_start_pos = np.asarray(pos, dtype=float).copy()
        self._commit_target_yaw = float(target_yaw)

    def _enter_backup(self, pos):
        self.mode = Mode.BACKUP
        self._mode_step = 0
        self._mode_start_pos = np.asarray(pos, dtype=float).copy()
        # Alternate yaw direction each entry so consecutive backups try
        # different rotational escapes from the wedge.
        self._backup_dir_sign *= -1.0

    # --------------------------------------------------------- main control

    def drive_logic(self, olfaction, quat, omm, raw_vision, pos,
                    contact_forces_world=None, antenna_data=None):
        # === 1. Sensory features (always computed) ===
        odor_steer = odor_attraction.odor_intensity_to_control_signal(
            olfaction, -self.attractive_gain
        )
        # Compute odor magnitude (matches odor_attraction's internal averaging)
        # to drive goal_mode (SCAN/COMMIT suppression near banana).
        odor_lr = np.average(
            olfaction[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
        )
        odor_mean = float(odor_lr.mean())
        self._last_odor_mean = odor_mean
        goal_mode = odor_mean > self.goal_odor_threshold
        roll_comp, pitch_comp, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, self.K_PITCH, self.K_ROLL, self.max_pitch_boost, self.max_roll_boost
        )
        yaw = self._yaw_from_quat(quat)

        # --- Wind sensor: antenna passive-force asymmetry ---
        # Adopted from Tom's `origin/tom` branch. Wind deflects the antennae;
        # the joint stiffness produces a passive restoring force whose
        # magnitude correlates with how far each antenna has deflected.
        # Difference between left and right magnitudes tells us which
        # antenna is more bent → wind direction. Cleaner than my previous
        # quaternion formulation, which assumes neutral resting pose is
        # known (it's not; the joint quaternion ≈ identity at neutral, so
        # walking jitter dominated the signal).
        # Sign convention: wind_lateral < 0 means right antenna more bent
        # → wind from right → fly should counter-steer right.
        if antenna_data is not None:
            try:
                lf = np.asarray(antenna_data["l"]["qfrc_passive"])
                rf = np.asarray(antenna_data["r"]["qfrc_passive"])
                wind_lateral = float(np.linalg.norm(lf) - np.linalg.norm(rf))
            except Exception:
                wind_lateral = 0.0
        else:
            wind_lateral = 0.0

        # --- Campaniform sensilla: total external force in fly's body frame ---
        # World-frame leg contact forces -> body-frame projection. Body-frame
        # +x is "forward" (head direction); negative x = something pushing the
        # fly's nose backward (wall). Hill climbing produces +x ground reaction.
        if contact_forces_world is not None and contact_forces_world.size > 0:
            rot_body_to_world = Rotation.from_quat(
                [quat[1], quat[2], quat[3], quat[0]]
            )
            forces_body = rot_body_to_world.inv().apply(contact_forces_world)
            total_body_force = forces_body.sum(axis=0)  # (3,)
            wall_force_signed = float(total_body_force[0])  # body-frame fore-aft
            # Wall contact = significant *negative* fore-aft force on body
            is_wall_contact = wall_force_signed < -self.wall_force_threshold
        else:
            wall_force_signed = 0.0
            is_wall_contact = False

        # --- Perception (the seam): blade features from vision. Base class
        # uses RGB green-segmentation; CNNController overrides _perceive with
        # the segmentation CNN. Same contract, same downstream state machine.
        perc = self._perceive(raw_vision)
        col_blade_rgb = perc["col_blade"]
        top_count = perc["top_count"]
        n_central = perc["n_central"]
        L_blades = perc["L_blades"]
        R_blades = perc["R_blades"]

        # --- Ommatidia path (kept ONLY for the debug overlay) ---
        rect = movement_correction._crop_hex_to_rect(omm, self.retina.ommatidia_id_map)
        run_lengths, col_blade_omm = movement_correction.vertical_run_features(
            rect,
            dark_threshold=self.dark_threshold,
            min_blade_run=self.min_blade_run,
        )
        dark_mask = rect < self.dark_threshold
        left_dark = float(dark_mask[0].mean())
        right_dark = float(dark_mask[1].mean())
        asym_dark = left_dark - right_dark
        col_dark = dark_mask.mean(axis=1)
        col_signal = float(col_dark.max())

        # Track 2D position for stuck detection.
        pos_xy = np.asarray(pos, dtype=float)
        self._pos_history.append(pos_xy)

        # === 2. State transitions ===
        # Severe tilt → BACKUP. Pitch is gated by wall contact so steep hills
        # don't false-trigger; lateral instability (roll) always triggers.
        severe_tilt = (
            (abs(pitch) > self.severe_pitch and is_wall_contact)
            or abs(roll) > self.severe_roll
        )
        if severe_tilt and self.mode is not Mode.BACKUP:
            self._enter_backup(pos_xy)
        elif self.mode is Mode.NAVIGATE:
            if len(self._pos_history) == self._pos_history.maxlen:
                disp = float(np.linalg.norm(self._pos_history[-1] - self._pos_history[0]))
                # SCAN trigger is displacement-only by design, even though it
                # fires spuriously in open terrain. Tested 10/05: gating with
                # `n_central >= 2` regressed L3 s1 to FLIP 21k min_dist 19.9
                # (was TIMEOUT 100k min_dist 6.4). The "spurious" SCAN/COMMIT
                # cycles in open terrain were doing useful work — small
                # asymmetries the gentle NAVIGATE centering misses get
                # corrected by SCAN's deliberate yaw curve. Don't gate.
                if disp < self.stuck_threshold:
                    self._enter_scan(pos_xy, yaw)
        elif self.mode is Mode.SCAN:
            # Track best heading we've seen so far.
            if n_central < self._scan_best_score:
                self._scan_best_score = n_central
                self._scan_best_yaw = yaw
            # Found clear forward -> commit immediately (after debounce).
            if n_central <= self.scan_clear_count and self._mode_step >= self.scan_min_steps:
                self._enter_commit(pos_xy, yaw)
            # Timeout -> commit to best heading found so far.
            elif self._mode_step >= self.scan_max_steps:
                target = self._scan_best_yaw if self._scan_best_yaw is not None else yaw
                self._enter_commit(pos_xy, target)
        elif self.mode is Mode.COMMIT:
            disp = float(np.linalg.norm(pos_xy - self._mode_start_pos))
            # Hit a wall during commit -> BACKUP (not NAVIGATE) so we get a
            # stateful disengagement instead of immediately re-entering normal
            # navigation while still pinned.
            if abs(pitch) > self.commit_pitch_abort:
                self._enter_backup(pos_xy)
            elif disp >= self.commit_min_disp or self._mode_step >= self.commit_max_steps:
                self._enter_navigate(pos_xy)
        elif self.mode is Mode.BACKUP:
            # Exit when timer expires, OR earlier if we've clearly recovered.
            # We exit to SCAN (not NAVIGATE) because every BACKUP is evidence
            # the fly is in a problem spot — going straight back to NAVIGATE
            # just walks into the same blade. SCAN actively looks for a clear
            # heading and COMMIT walks the fly out. This breaks the
            # NAVIGATE↔BACKUP loop that traps the fly with full pos_history
            # never able to fill before the next BACKUP fires (so the
            # NAVIGATE-side stuck trigger could never run).
            recovered = (
                abs(pitch) < self.backup_safe_pitch
                and abs(roll) < self.backup_safe_roll
                and self._mode_step >= self.backup_min_steps
            )
            if recovered or self._mode_step >= self.backup_duration:
                self._enter_scan(pos_xy, yaw)

        # === 3. Mode-specific intended movement ===
        if self.mode is Mode.NAVIGATE:
            # --- Reflex turn (Tom's idea): if many central cols are blade,
            # do a hard max-magnitude turn away from the denser side. This
            # short-circuits the gentle centering when a real cluster is
            # ahead, preventing the fly from entering the pitch-45° wedge.
            # L_blades / R_blades come from _perceive() (the seam).
            n_blades_total = max(int(L_blades), int(R_blades))

            reflex_active = False
            if n_central >= self.reflex_n_central_threshold:
                # Trigger / refresh reflex with sign aimed away from denser side
                self._reflex_sign = +1 if L_blades > R_blades else -1
                self._reflex_remaining = max(self._reflex_remaining, self.reflex_min_steps)
            if self._reflex_remaining > 0:
                reflex_active = True
                # Sign +1 → turn right ([+a, -a]), sign -1 → turn left
                intended = np.array([
                    self._reflex_sign * self.reflex_drive_magnitude,
                    -self._reflex_sign * self.reflex_drive_magnitude,
                ])
                speed = self.navigate_speed * self.reflex_speed_factor
                grass_centering = np.zeros(2)
                self._reflex_remaining -= 1

            if not reflex_active:
                # Standard centering + wind + LPLC1 path
                total = L_blades + R_blades
                if total > 0:
                    asym_run = (L_blades - R_blades) / total
                else:
                    asym_run = 0.0
                centering = np.array([asym_run, -asym_run]) * self.navigate_centering_gain
                centering = np.clip(
                    centering,
                    -self.navigate_centering_clip,
                    self.navigate_centering_clip,
                )
                wind_correction = np.array([-wind_lateral, wind_lateral]) * self.wind_gain
                wind_correction = np.clip(
                    wind_correction,
                    -self.wind_correction_clip,
                    self.wind_correction_clip,
                )
                intended = odor_steer + centering + wind_correction
                # LPLC1 speed modulation: slow proportionally to blade clutter so
                # the fly enters clusters at low speed instead of crashing in.
                clutter_factor = 1.0 - (1.0 - self.lplc1_min_speed_factor) * min(
                    1.0, n_blades_total / self.lplc1_full_slow_at_n_blades
                )
                speed = self.navigate_speed * clutter_factor
                grass_centering = centering
                # Object-level avoidance: steer toward the clearest gap when a
                # close blade is dead ahead. No-op in the base controller;
                # CNNController overrides _navigate_obstacle_avoidance.
                avoid_drive, avoid_speed = self._navigate_obstacle_avoidance(perc)
                intended = intended + avoid_drive
                speed = speed * avoid_speed
            else:
                clutter_factor = 1.0  # for debug_info compatibility
        elif self.mode is Mode.SCAN:
            scan_drive = (
                np.array([self.scan_drive_magnitude, -self.scan_drive_magnitude])
                * self._scan_dir_sign
            )
            intended = scan_drive
            speed = self.scan_speed
            grass_centering = np.zeros(2)
        elif self.mode is Mode.COMMIT:
            yaw_err = self._wrap_deg(self._commit_target_yaw - yaw)
            yaw_corr = float(np.clip(
                yaw_err * self.commit_yaw_gain,
                -self.commit_yaw_correction_max,
                self.commit_yaw_correction_max,
            ))
            intended = self.commit_drive + np.array([-yaw_corr, +yaw_corr])
            speed = self.commit_speed
            grass_centering = np.zeros(2)
        else:  # Mode.BACKUP
            # Phase 1: pure straight backup at max magnitude — physical
            # disengagement push.
            # Phase 2+: back + yaw at max magnitude. Sign flips every
            # `backup_yaw_period` steps WITHIN this BACKUP, so a single
            # backup tries both rotational directions before giving up.
            # Across separate BACKUP entries, the initial sign is also
            # alternated via `_backup_dir_sign`.
            if self._mode_step < self.backup_phase1_steps:
                intended = np.array([-1.0, -1.0])
            else:
                periods_in = (self._mode_step - self.backup_phase1_steps) // self.backup_yaw_period
                sign = self._backup_dir_sign * (1.0 if periods_in % 2 == 0 else -1.0)
                if sign > 0:
                    intended = np.array([-1.0, -0.4])  # back + strong right yaw
                else:
                    intended = np.array([-0.4, -1.0])  # back + strong left yaw
            speed = self.backup_speed
            grass_centering = np.zeros(2)

        # === 4. Moderate-tilt blending (NAVIGATE only; severe tilt is BACKUP) ===
        if self.mode is Mode.NAVIGATE:
            if pitch > self.max_pitch:
                intended = 0.6 * intended + 0.4 * pitch_comp
            if abs(roll) > self.max_roll:
                intended = 0.5 * intended + 0.5 * roll_comp
                speed = min(speed, self.navigate_speed * 0.6)

        # Dragonfly override (level 4 only) — RED color detection (Tom's idea).
        # When dragonfly is visible, override every other behaviour with a
        # hard escape. Sides:
        #   "left"  → dragonfly to the left → turn RIGHT (drives [3.0, 0.5])
        #   "right" → dragonfly to the right → turn LEFT  (drives [0.5, 3.0])
        #   "front" → both eyes red → run forward at max ([3.0, 3.0])
        dragon_loom = False
        dragon_side = "none"
        if self.enable_dragonfly:
            dragon_loom, dragon_side = movement_correction.detect_dragonfly_red(
                raw_vision, threshold_fraction=self.dragon_red_threshold
            )
            if dragon_loom:
                if dragon_side == "left":
                    intended = np.array([3.0, 0.5])
                elif dragon_side == "right":
                    intended = np.array([0.5, 3.0])
                else:  # "front"
                    intended = np.array([3.0, 3.0])
                speed = self.navigate_speed
                if self.mode is not Mode.NAVIGATE:
                    self._enter_navigate(pos_xy)

        # === 5. Final clip ===
        max_drive = max(self.navigate_speed, self.commit_speed)
        final_drive = intended * speed
        clipped = np.clip(final_drive, -max_drive, max_drive)

        # === 6. Debug + history ===
        # Map state-machine mode to a legacy `escape_phase` so the existing
        # preview_debug.py overlay keeps rendering meaningfully.
        legacy_phase = {
            Mode.NAVIGATE: "none",
            Mode.SCAN: "scan",
            Mode.COMMIT: "commit",
            Mode.BACKUP: "backup",
        }[self.mode]
        legacy_remaining = 0
        if self.mode is Mode.SCAN:
            legacy_remaining = max(0, self.scan_max_steps - self._mode_step)
        elif self.mode is Mode.COMMIT:
            legacy_remaining = max(0, self.commit_max_steps - self._mode_step)
        elif self.mode is Mode.BACKUP:
            legacy_remaining = max(0, self.backup_duration - self._mode_step)

        self.debug_info = {
            # Vision (used by overlay)
            "rect": rect,
            "left_dark": left_dark,
            "right_dark": right_dark,
            "asym": asym_dark,
            "edge_asym": 0.0,
            "hrc_asym": 0.0,
            "combined_asym": asym_dark,
            "hrc": np.zeros((2, 16, 30), dtype=np.float32),
            "min_dark": min(left_dark, right_dark),
            "bull_eye": min(left_dark, right_dark) > 0.20,
            "col_signal": col_signal,
            # Posture / centering
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "roll_attenuation": 1.0,
            "grass_centering": grass_centering,
            # State machine
            "mode": self.mode.value,
            "mode_step": self._mode_step,
            "n_central_blades": n_central,
            "run_max_L": int(run_lengths[0].max()),
            "run_max_R": int(run_lengths[1].max()),
            "blades_L": int(col_blade_rgb[0].sum()),
            "blades_R": int(col_blade_rgb[1].sum()),
            "top_count_max_L": int(top_count[0].max()),
            "top_count_max_R": int(top_count[1].max()),
            "scan_best_score": (
                int(self._scan_best_score)
                if self._scan_best_score != float("inf")
                else None
            ),
            "scan_best_yaw": (
                None if self._scan_best_yaw is None else float(self._scan_best_yaw)
            ),
            "commit_target_yaw": (
                None if self._commit_target_yaw is None else float(self._commit_target_yaw)
            ),
            # New: campaniform/LPLC1/Johnston signals
            "wall_force": wall_force_signed,
            "is_wall_contact": is_wall_contact,
            "clutter_factor": (
                clutter_factor if self.mode is Mode.NAVIGATE else 1.0
            ),
            "wind_lateral": wind_lateral,
            "odor_mean": odor_mean,
            "goal_mode": goal_mode,
            # Legacy keys (preview_debug.py + old analysis scripts)
            "escape_phase": legacy_phase,
            "escape_remaining": int(legacy_remaining),
            "gap_mode": self.mode is Mode.SCAN,
            "dragon_loom": dragon_loom,
        }

        self._history.append({
            "mode": self.mode.value,
            "mode_step": self._mode_step,
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "left_dark": left_dark,
            "right_dark": right_dark,
            "asym": float(asym_dark),
            "col_signal": float(col_signal),
            "n_central_blades": n_central,
            "run_max_L": int(run_lengths[0].max()),
            "run_max_R": int(run_lengths[1].max()),
            "blades_L": int(col_blade_rgb[0].sum()),
            "blades_R": int(col_blade_rgb[1].sum()),
            "top_count_max_L": int(top_count[0].max()),
            "top_count_max_R": int(top_count[1].max()),
            "centering_L": float(grass_centering[0]),
            "drive_L": float(clipped[0]),
            "drive_R": float(clipped[1]),
            "commit_target_yaw": (
                None if self._commit_target_yaw is None else float(self._commit_target_yaw)
            ),
            "dragon_loom": bool(dragon_loom),
            "escape_phase": legacy_phase,
            "escape_remaining": int(legacy_remaining),
            "blade_contact": bool(col_signal > 0.30),
            "gap_mode": self.mode is Mode.SCAN,
            "wall_force": wall_force_signed,
            "is_wall_contact": bool(is_wall_contact),
            "clutter_factor": float(
                clutter_factor if self.mode is Mode.NAVIGATE else 1.0
            ),
            "wind_lateral": float(wind_lateral),
            "odor_mean": float(odor_mean),
            "goal_mode": bool(goal_mode),
        })

        self._mode_step += 1
        return clipped

    # ------------------------------------------------------------------ step

    def step(self, sim: MiniprojectSimulation):
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations(sim.fly.name)[0]
        omm = sim.get_ommatidia_readouts(sim.fly.name)
        raw_vision = sim.get_raw_vision(sim.fly.name)
        pos = np.array(sim.get_body_positions(sim.fly.name)[0][:2])
        # Per-contact 3D external forces (campaniform-sensilla equivalent).
        contact_forces_world = sim.get_external_force(
            sim.fly.name, subtract_adhesion_force=True
        )
        # Antenna deflections (Johnston's organ analog) — wind sensor.
        # Returns {'l': {'qpos':[w,x,y,z], 'qvel':..., 'qacc':..., ...}, 'r': ...}
        try:
            antenna_data = sim.get_antenna_data(sim.fly.name)
        except Exception:
            antenna_data = None

        if self._mode_start_pos is None:
            self._mode_start_pos = pos.copy()

        drives = self.drive_logic(
            olfaction, quat, omm, raw_vision, pos,
            contact_forces_world, antenna_data,
        )
        joint_angles, adhesion = self.turning_controller.step(drives)
        # Graded adhesion release during BACKUP:
        #   low tilt → full release (slip off blade)
        #   mid tilt → partial release (some slip, some grip — break the dead-
        #              zone equilibrium where binary gate kept fly stuck)
        #   high tilt → full keep (brake the fall)
        if self.backup_release_adhesion and self.mode is Mode.BACKUP:
            cur_pitch = self.debug_info.get("pitch", 0.0)
            cur_roll = self.debug_info.get("roll", 0.0)
            tilt = max(abs(cur_pitch), abs(cur_roll))
            lo = self.backup_release_low_tilt
            hi = self.backup_release_high_tilt
            if tilt <= lo:
                keep_fraction = 0.0
            elif tilt >= hi:
                keep_fraction = 1.0
            else:
                keep_fraction = (tilt - lo) / (hi - lo)
            adhesion = adhesion * keep_fraction
        return joint_angles, adhesion
