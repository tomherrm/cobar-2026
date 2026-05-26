"""Gymnasium environment wrapper around MiniprojectSimulation for RL training.

Observation: Dict {vision: (6,64,64) uint8 — both eyes downsampled & stacked
channels-first; scalars: (8,) float32 — odor / tilt / contact / wind / last
action}. The CNN branch learns its own object-level obstacle representation
from raw vision; the scalars carry the signals a CNN cannot recover from
pixels (olfaction is a separate sense and the goal signal, proprioceptive
tilt, body-frame contact force, wind).
Action: continuous [drive_L, drive_R] in [-1.8, 1.8], matches TurningController.
Reward: delta_dist_to_banana per step + terminal bonuses for success/flip.
Termination: success (within 3 units of banana), flip (|roll|>85 or |pitch|>90),
or step budget exhausted.
"""
from __future__ import annotations

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from miniproject.simulation import MiniprojectSimulation
from flygym.compose import ActuatorType
from flygym.vision.retina import Retina

from submission import movement_correction


# ----------------------------- Observation builder -----------------------------

# Observation layout. The CNN branch reads `vision`; the MLP branch reads
# `scalars`. Change these constants if you change the layout below.
VISION_SHAPE = (6, 64, 64)   # both eyes, downsampled, stacked channels-first
SCALAR_DIM = 8               # odor_diff, odor_mean, pitch, roll, wall, wind, last_L, last_R


def _downsample_vision(raw_vision, size: int = 64) -> np.ndarray:
    """(2, H, W, 3) uint8 raw vision -> (6, size, size) uint8.

    Each eye is area-averaged down to size x size (INTER_AREA is the
    anti-aliased downsample — critical so thin grass blades don't alias
    away), transposed to channels-first, then both eyes are stacked into
    6 channels so a single CNN sees the whole visual field.
    """
    raw_vision = np.asarray(raw_vision)
    eyes = []
    for eye in raw_vision:  # eye: (H, W, 3) uint8
        small = cv2.resize(eye, (size, size), interpolation=cv2.INTER_AREA)
        eyes.append(np.transpose(small, (2, 0, 1)))  # (3, size, size)
    vision = np.concatenate(eyes, axis=0)            # (6, size, size)
    return np.ascontiguousarray(vision, dtype=np.uint8)


def _build_observation(
    olfaction: np.ndarray,
    quat: np.ndarray,
    pitch: float,
    roll: float,
    raw_vision,
    contact_forces_world: np.ndarray,
    antenna_data: dict | None,
    last_action: np.ndarray,
) -> dict:
    """Construct the Dict observation: downsampled vision + 8 scalars.

    The CNN learns its own obstacle representation from `vision`. `scalars`
    carries the signals a CNN cannot recover from pixels alone — olfaction
    (a separate sense, and the goal signal), proprioceptive tilt, body-frame
    contact force, wind, and the last action.
    """
    # Odor: weighted left/right (front-biased) + scalar mean
    odor_lr = np.average(
        olfaction[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    odor_mean = float(odor_lr.mean())
    odor_diff = float(odor_lr[0] - odor_lr[1])

    # Tilt (proprioception)
    pitch_n = pitch / 90.0   # [-1, 1] approx
    roll_n = roll / 90.0

    # Body-frame fore-aft contact force (wall vs hill)
    wall_force = 0.0
    if contact_forces_world is not None and contact_forces_world.size > 0:
        rot_b = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        forces_body = rot_b.inv().apply(contact_forces_world)
        wall_force = float(forces_body.sum(axis=0)[0])

    # Wind via antenna passive-force asymmetry
    wind_lat = 0.0
    if antenna_data is not None:
        try:
            lf = np.asarray(antenna_data["l"]["qfrc_passive"])
            rf = np.asarray(antenna_data["r"]["qfrc_passive"])
            wind_lat = float(np.linalg.norm(lf) - np.linalg.norm(rf))
        except (KeyError, ValueError, TypeError):
            wind_lat = 0.0

    scalars = np.array([
        np.tanh(odor_diff * 1e6),       # left-right odor asymmetry, squashed
        np.tanh(odor_mean * 1e6),       # mean odor magnitude, squashed
        np.clip(pitch_n, -1.5, 1.5),    # pitch / 90
        np.clip(roll_n, -1.5, 1.5),     # roll / 90
        np.tanh(wall_force / 10.0),     # body-frame fore-aft contact force
        np.tanh(wind_lat * 1000.0),     # antenna passive-force asymmetry
        float(last_action[0]),          # last drive_L
        float(last_action[1]),          # last drive_R
    ], dtype=np.float32)
    assert scalars.shape == (SCALAR_DIM,), \
        f"scalars shape {scalars.shape} != ({SCALAR_DIM},)"

    vision = _downsample_vision(raw_vision, size=VISION_SHAPE[1])
    assert vision.shape == VISION_SHAPE, \
        f"vision shape {vision.shape} != {VISION_SHAPE}"

    return {"vision": vision, "scalars": scalars}


# ------------------------------- Gym Env class --------------------------------


class CobarEnv(gym.Env):
    """Single-fly cobar-2026 environment for PPO.

    Parameters
    ----------
    level : int, default 2
        Cobar level (0..4). 2 = grass clusters, 3 = +wind, 4 = +dragonfly.
    seed_pool : list[int] | None
        Pool of seeds to draw from at reset. None = random per reset.
    max_steps : int, default 20000
        Episode step budget for training. Eval typically uses 100000.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        level: int = 2,
        seed_pool: list[int] | None = None,
        max_steps: int = 30000,
        success_radius: float = 3.0,
        flip_pitch: float = 90.0,
        flip_roll: float = 85.0,
        # Path B: action smoothing + reshape reward.
        # action_smooth_alpha: applied = (1-alpha)*prev + alpha*action.
        # alpha=1.0 = no smoothing (raw PPO output). alpha=0.3 = strong
        # smoothing — good for CPG which expects coherent descending commands.
        action_smooth_alpha: float = 0.3,
        # Reward shape. The terminal flip penalty is invisible to the value
        # function over long episodes (gamma horizon << episode length), so
        # "don't flip" is carried instead by a DENSE per-step tilt penalty:
        # free below tilt_safe_*, quadratic ramp beyond it. flip_penalty is
        # kept small (the dense signal does the work; a large terminal spike
        # was the PPO variance bomb in session 13).
        flip_penalty: float = 50.0,
        timeout_penalty: float = 10.0,
        success_bonus: float = 1000.0,
        time_cost: float = 0.0001,
        delta_scale: float = 1.0,
        tilt_cost: float = 1e-4,
        tilt_safe_pitch: float = 25.0,
        tilt_safe_roll: float = 25.0,
    ):
        super().__init__()
        self.level = level
        self.seed_pool = seed_pool
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.flip_pitch = flip_pitch
        self.flip_roll = flip_roll
        self.action_smooth_alpha = float(action_smooth_alpha)
        self.flip_penalty = float(flip_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.success_bonus = float(success_bonus)
        self.time_cost = float(time_cost)
        self.delta_scale = float(delta_scale)
        self.tilt_cost = float(tilt_cost)
        self.tilt_safe_pitch = float(tilt_safe_pitch)
        self.tilt_safe_roll = float(tilt_safe_roll)

        # Action: continuous [drive_L, drive_R] in [-1.8, 1.8].
        # Range matches the heuristic's clip (navigate_speed=1.8) so BC
        # demonstrations fit the action space exactly.
        self.action_space = spaces.Box(
            low=-1.8, high=1.8, shape=(2,), dtype=np.float32
        )
        # Observation: Dict {vision: CNN input, scalars: MLP input}.
        # SB3 MultiInputPolicy routes each sub-space to its own extractor.
        self.observation_space = spaces.Dict({
            "vision": spaces.Box(
                low=0, high=255, shape=VISION_SHAPE, dtype=np.uint8
            ),
            "scalars": spaces.Box(
                low=-np.inf, high=np.inf, shape=(SCALAR_DIM,), dtype=np.float32
            ),
        })

        self._rng = np.random.default_rng()
        self._sim = None
        self._turning = None
        self._retina = None
        self._step_idx = 0
        self._prev_dist = None
        self._last_action = np.zeros(2, dtype=np.float32)
        self._smoothed_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a seed for the sim
        if self.seed_pool is not None:
            sim_seed = int(self._rng.choice(self.seed_pool))
        else:
            sim_seed = int(self._rng.integers(0, 1_000_000))

        self._sim = MiniprojectSimulation(level=self.level, seed=sim_seed)
        from flygym.examples.locomotion import TurningController
        self._turning = TurningController(self._sim.timestep)
        self._retina = Retina()
        self._step_idx = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self._smoothed_action = np.zeros(2, dtype=np.float32)

        banana_xy = np.asarray(self._sim.world.banana_xy, dtype=np.float32)
        self._banana_xy = banana_xy

        obs = self._build_obs()
        # Initial distance for delta-reward shaping
        fly_xy = np.asarray(
            self._sim.get_body_positions(self._sim.fly.name)[0][:2]
        )
        self._prev_dist = float(np.linalg.norm(fly_xy - banana_xy))

        return obs, {"sim_seed": sim_seed, "banana_xy": banana_xy}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Action smoothing: low-pass filter PPO's noisy outputs so the CPG
        # receives coherent commands. alpha=1 disables smoothing.
        a = self.action_smooth_alpha
        self._smoothed_action = (1.0 - a) * self._smoothed_action + a * action
        applied = self._smoothed_action
        self._last_action = applied

        # Apply via TurningController (same as the heuristic controller)
        joint_angles, adhesion = self._turning.step(applied)
        self._sim.set_actuator_inputs(
            self._sim.fly.name, ActuatorType.POSITION, joint_angles
        )
        self._sim.set_actuator_inputs(
            self._sim.fly.name, ActuatorType.ADHESION, adhesion
        )
        self._sim.step()
        self._step_idx += 1

        obs = self._build_obs()

        # Reward shaping: dense distance progress + a DENSE per-step tilt
        # penalty (see __init__), with terminal bonuses on top.
        fly_xy = np.asarray(
            self._sim.get_body_positions(self._sim.fly.name)[0][:2]
        )
        dist = float(np.linalg.norm(fly_xy - self._banana_xy))
        delta = self._prev_dist - dist  # positive when getting closer
        self._prev_dist = dist
        reward = float(self.delta_scale * delta - self.time_cost)

        # Pitch/roll every step (not just at episode end) so the dense tilt
        # penalty is immediately creditable to the action that caused it.
        quat = self._sim.get_body_rotations(self._sim.fly.name)[0]
        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )
        over_pitch = max(0.0, abs(pitch) - self.tilt_safe_pitch)
        over_roll = max(0.0, abs(roll) - self.tilt_safe_roll)
        reward -= self.tilt_cost * (over_pitch ** 2 + over_roll ** 2)

        terminated = False
        truncated = False
        info = {"dist": dist}

        if dist <= self.success_radius:
            terminated = True
            reward += self.success_bonus
            info["outcome"] = "success"
        elif abs(pitch) > self.flip_pitch or abs(roll) > self.flip_roll:
            terminated = True
            reward -= self.flip_penalty
            info["outcome"] = "flip"
        elif self._step_idx >= self.max_steps:
            truncated = True
            reward -= self.timeout_penalty
            info["outcome"] = "timeout"

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------- internal helpers

    def _build_obs(self) -> dict:
        sim = self._sim
        fly_name = sim.fly.name
        olfaction = sim.get_olfaction(fly_name)
        quat = sim.get_body_rotations(fly_name)[0]
        raw_vision = sim.get_raw_vision(fly_name)
        contact_forces_world = sim.get_external_force(
            fly_name, subtract_adhesion_force=True
        )
        try:
            antenna_data = sim.get_antenna_data(fly_name)
        except Exception:
            antenna_data = None

        _, _, pitch, roll = movement_correction.tilt_to_control_signal(
            quat, 10, 50, 0.5, 0.3
        )

        return _build_observation(
            olfaction, quat, pitch, roll, raw_vision,
            contact_forces_world, antenna_data, self._last_action,
        )

    def close(self):
        # MuJoCo sim cleanup is done via garbage collection
        self._sim = None
