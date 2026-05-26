"""ReactiveController — v4 of the heuristic+CNN attempts.

v1/v2/v3 all bolted an avoidance term onto the heuristic's NAVIGATE/SCAN/COMMIT
state machine and plateaued at ≈ heuristic — because COMMIT drives blind and
SCAN turns in place, so the avoidance kept getting overridden or fought with
that machine. v4 *removes* the state machine and replaces it with a single
**continuous reactive steering law**: every step, the CNN segmentation mask is
combined into one drive vector (odor pull + continuous obstacle repulsion from
"near" sectors + clutter-driven speed modulation), plus an **early anti-wedge
reflex**. BACKUP is kept as the only stateful mode, for escape if a wedge does
form. SCAN and COMMIT are gone — perception and action stay in a closed
reactive loop, the way biology and reactive robotics both do it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from submission import odor_attraction, movement_correction
from submission.controller import Mode

from cnn_perception.cnn_controller import CNNController


class ReactiveController(CNNController):
    """v4: continuous reactive controller. Replaces NAVIGATE/SCAN/COMMIT with
    one reactive law that runs every step. Keeps BACKUP.
    """

    def __init__(self, sim, cnn_path: str | None = None, cnn_res: int = 128):
        super().__init__(sim, cnn_path=cnn_path, cnn_res=cnn_res)

        # Reactive steering — every step, no triggering.
        self.reactive_speed = 1.5              # base forward speed
        self.obstacle_gain = 4.0               # obstacle-repulsion turn gain
        self.obstacle_turn_clip = 0.5          # bounded perturbation (not override)
        self.clutter_speed_alpha = 0.6         # how aggressively clutter slows speed
        self.clutter_full_near = 0.025         # near-score ahead at which speed bottoms out
        self.reactive_min_speed_factor = 0.3   # never go below this fraction

        # Early anti-wedge reflex — hard back-off at pitch climbing past this
        # WITH wall contact, before the ~45° dead-zone equilibrium can form.
        # Gated by wall-force so hills don't false-trigger.
        # Bidirectional: abs(pitch) so face-plant (negative pitch from L3 wind
        # or downward slope) also triggers; back-drive [-1,-1] works in both
        # directions (pulls center-of-mass away from contact point either way).
        self.anti_wedge_pitch = 25.0
        self.anti_wedge_drive = np.array([-1.0, -1.0], dtype=np.float32)

        # === Safety overrides tuned from L0-L4 diagnostics (15/05/2026) ===
        # severe_pitch_hard: ungated last-resort pitch trigger. The wall-force
        # gate fails on (a) walking onto a steep blade base where contact is
        # under-belly not head-on, (b) dragonfly interactions, (c) some hill
        # geometries. Without this, pitch climbs to 90° in NAVIGATE with no
        # BACKUP firing (seen reliably on L4 s1 at step 12996 and L3 s1 at
        # 48339). Adhesion stays fully retained at tilt>=50° (graded ramp
        # high-end), so this fallback doesn't slip-flip on hills.
        self.severe_pitch_hard = 50.0
        # Lower severe_roll 50 -> 40: L2 s67 flipped because BACKUP fired only
        # when roll already exceeded 50°, but BACKUP's back-drive can't right
        # a fly past ~roll 60° (it's already on its side). Earlier trigger
        # gives the back-drive time to right the fly before it tumbles.
        self.severe_roll = 40.0

    def drive_logic(self, olfaction, quat, omm, raw_vision, pos,
                    contact_forces_world=None, antenna_data=None):
        # === 1. Sensory features ===
        perc = self._perceive(raw_vision)
        odor_steer = odor_attraction.odor_intensity_to_control_signal(
            olfaction, -self.attractive_gain
        )
        odor_lr = np.average(
            olfaction[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
        )
        odor_mean = float(odor_lr.mean())
        roll_comp, pitch_comp, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, self.K_PITCH, self.K_ROLL,
            self.max_pitch_boost, self.max_roll_boost,
        )

        # Wall force (campaniform-sensilla proxy) — gates anti-wedge and the
        # severe-tilt -> BACKUP trigger so hills don't false-fire.
        if contact_forces_world is not None and contact_forces_world.size > 0:
            rot_b = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            forces_body = rot_b.inv().apply(contact_forces_world)
            wall_force = float(forces_body.sum(axis=0)[0])
            is_wall_contact = wall_force < -self.wall_force_threshold
        else:
            wall_force = 0.0
            is_wall_contact = False

        # Wind from antenna passive-force asymmetry.
        wind_lateral = 0.0
        if antenna_data is not None:
            try:
                lf = np.asarray(antenna_data["l"]["qfrc_passive"])
                rf = np.asarray(antenna_data["r"]["qfrc_passive"])
                wind_lateral = float(np.linalg.norm(lf) - np.linalg.norm(rf))
            except Exception:
                wind_lateral = 0.0

        pos_xy = np.asarray(pos, dtype=float)
        self._pos_history.append(pos_xy)

        # === 2. Mode transitions — REACTIVE (== NAVIGATE) <-> BACKUP only ===
        # Three triggers, any of which fires BACKUP:
        #   (a) early: moderate pitch (>30°) WITH wall contact -- responsive
        #       to obvious wall hits even at low absolute angle
        #   (b) hard fallback: pitch above 50° regardless of wall contact --
        #       catches the failure modes seen in L3/L4 diagnostics where
        #       the wall-force gate never fires but pitch keeps climbing
        #   (c) roll: any roll past 40° (lowered from 50°) -- BACKUP cannot
        #       recover a roll past 60°, so we need a hard earlier trigger
        severe_tilt = (
            (abs(pitch) > self.severe_pitch and is_wall_contact)
            or abs(pitch) > self.severe_pitch_hard
            or abs(roll) > self.severe_roll
        )
        if severe_tilt and self.mode is not Mode.BACKUP:
            self._enter_backup(pos_xy)
        elif self.mode is Mode.BACKUP:
            recovered = (
                abs(pitch) < self.backup_safe_pitch
                and abs(roll) < self.backup_safe_roll
                and self._mode_step >= self.backup_min_steps
            )
            if recovered or self._mode_step >= self.backup_duration:
                self._enter_navigate(pos_xy)
        # We NEVER transition to SCAN or COMMIT — that's the whole point.

        # === 3. Mode-specific drive ===
        clutter_factor = 1.0
        if self.mode is Mode.BACKUP:
            # Stateful escape (verbatim from parent).
            if self._mode_step < self.backup_phase1_steps:
                intended = np.array([-1.0, -1.0])
            else:
                periods_in = ((self._mode_step - self.backup_phase1_steps)
                              // self.backup_yaw_period)
                sign = (self._backup_dir_sign
                        * (1.0 if periods_in % 2 == 0 else -1.0))
                intended = (np.array([-1.0, -0.4]) if sign > 0
                            else np.array([-0.4, -1.0]))
            speed = self.backup_speed
        else:
            # ---- REACTIVE mode (the v4 law) ----
            seg = perc["seg_feats"]
            density = np.asarray(seg["sector_density"], dtype=np.float32)
            proximity = np.asarray(seg["sector_proximity"], dtype=np.float32)
            n = len(density)
            center = (n - 1) / 2.0

            # near-obstacle score per sector — discriminates near vs distant
            # (distant horizon grass has proximity ~0.5 -> ~0; a near blade
            #  has high density AND high proximity -> high score).
            near = density * np.clip((proximity - 0.55) / 0.45, 0.0, 1.0)

            # Obstacle repulsion turn: sum_s near[s] * (center - s) / center.
            # A near obstacle on the LEFT (s < center) contributes positive
            # turn -> [+turn, -turn] turns RIGHT, away from the obstacle.
            sectors = np.arange(n)
            raw_turn = float(np.sum(near * (center - sectors) / center))
            obstacle_turn = float(np.clip(
                raw_turn * self.obstacle_gain,
                -self.obstacle_turn_clip, self.obstacle_turn_clip,
            ))

            # Clutter ahead -> slow down (LPLC1-style).
            central = [int(np.floor(center)), int(np.ceil(center))]
            clutter_ahead = float(near[central].mean())
            clutter_factor = max(
                self.reactive_min_speed_factor,
                1.0 - self.clutter_speed_alpha * min(
                    1.0, clutter_ahead / self.clutter_full_near
                ),
            )

            # Wind correction (matters on L3+; harmless on L2).
            wind_correction = np.array(
                [-wind_lateral, wind_lateral], dtype=np.float32
            ) * self.wind_gain
            wind_correction = np.clip(
                wind_correction,
                -self.wind_correction_clip, self.wind_correction_clip,
            )

            # Combined reactive drive: goal pull + obstacle repulsion + wind.
            intended = (odor_steer
                        + wind_correction
                        + np.array([obstacle_turn, -obstacle_turn],
                                   dtype=np.float32))
            speed = self.reactive_speed * clutter_factor

            # ---- Early anti-wedge reflex (bidirectional) ----
            # Pitch climbing into the danger zone WITH wall contact = heading
            # into a wedge. Hard back-drive to bleed forward momentum BEFORE
            # the ~45° dead-zone equilibrium can form.
            # Bidirectional after L3 diagnostic: negative pitch (face-plant)
            # also needs the same back-drive — pulls the head up and away
            # from the ground/obstacle the fly is tipping into.
            if abs(pitch) > self.anti_wedge_pitch and is_wall_contact:
                intended = self.anti_wedge_drive.copy()
                speed = self.backup_speed

            # Moderate tilt blending (kept from parent).
            if pitch > self.max_pitch:
                intended = 0.6 * intended + 0.4 * pitch_comp
            if abs(roll) > self.max_roll:
                intended = 0.5 * intended + 0.5 * roll_comp
                speed = min(speed, self.reactive_speed * 0.6)

        # === 4. Dragonfly override (L4 only) ===
        dragon_loom = False
        if self.enable_dragonfly:
            dragon_loom, dragon_side = movement_correction.detect_dragonfly_red(
                raw_vision, threshold_fraction=self.dragon_red_threshold
            )
            if dragon_loom:
                if dragon_side == "left":
                    intended = np.array([3.0, 0.5])
                elif dragon_side == "right":
                    intended = np.array([0.5, 3.0])
                else:
                    intended = np.array([3.0, 3.0])
                speed = self.reactive_speed
                if self.mode is not Mode.NAVIGATE:
                    self._enter_navigate(pos_xy)

        # === 5. Final clip ===
        max_drive = max(self.reactive_speed, self.backup_speed)
        clipped = np.clip(intended * speed, -max_drive, max_drive)

        # === 6. Minimal debug (parent's step() reads pitch/roll for adhesion) ===
        self.debug_info = {
            "mode": self.mode.value,
            "mode_step": self._mode_step,
            "pitch": float(pitch),
            "roll": float(roll),
            "wall_force": wall_force,
            "is_wall_contact": bool(is_wall_contact),
            "clutter_factor": float(clutter_factor),
            "odor_mean": float(odor_mean),
            "wind_lateral": float(wind_lateral),
            "blades_L": float(perc["L_blades"]),
            "blades_R": float(perc["R_blades"]),
            "n_central_blades": int(perc["n_central"]),
            "dragon_loom": dragon_loom,
        }
        self._history.append({
            "mode": self.mode.value,
            "mode_step": self._mode_step,
            "pitch": float(pitch),
            "roll": float(roll),
            "drive_L": float(clipped[0]),
            "drive_R": float(clipped[1]),
            "n_central_blades": int(perc["n_central"]),
            "blades_L": float(perc["L_blades"]),
            "blades_R": float(perc["R_blades"]),
            "clutter_factor": float(clutter_factor),
            "wall_force": float(wall_force),
            "is_wall_contact": bool(is_wall_contact),
            "dragon_loom": bool(dragon_loom),
        })

        self._mode_step += 1
        return clipped

    # --------------------------------------------------------------- step
    def step(self, sim):
        """Override parent step to add pin-escape adhesion release.

        The parent (Controller.step) applies a graded adhesion release in
        BACKUP that goes 0 -> 1 as tilt rises from 30° to 50°. The original
        intent (brake fall at high tilt) backfires when the fly is pinned by
        an external force: at tilt 50°+ adhesion is fully retained, gluing
        the legs to whatever is holding the fly up, and BACKUP's back-drive
        can't escape. Observed reliably on L4 s1 (fly pinned, pitch climbed
        50°->90° during BACKUP, mode_counts: 12886 navigate, 108 backup).

        We add a second ramp on top of the parent: above tilt 60°, scale
        adhesion back DOWN from 1.0 to 0.0 over a 25° window. This lets the
        fly detach when it's clearly pinned, while leaving the parent's
        low-tilt slip-out and mid-tilt full-keep behaviors intact.
        """
        joint_angles, adhesion = super().step(sim)
        if self.mode is Mode.BACKUP:
            cur_pitch = self.debug_info.get("pitch", 0.0)
            cur_roll = self.debug_info.get("roll", 0.0)
            tilt = max(abs(cur_pitch), abs(cur_roll))
            if tilt > 60.0:
                # 1.0 at 60° -> 0.0 at 85°; clamp below 60° handled above
                release_factor = max(0.0, 1.0 - (tilt - 60.0) / 25.0)
                adhesion = adhesion * release_factor
        return joint_angles, adhesion
