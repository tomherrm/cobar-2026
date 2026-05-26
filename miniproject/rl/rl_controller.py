"""Wrap a trained PPO policy as a cobar `Controller` so we can evaluate it
with the project's standard run_headless.py / preview_controller.py scripts.

Usage:
    from rl.rl_controller import RLController
    # or, to drop into the submission/ directory expected by preview_controller:
    # `import RLController as Controller`

By default the policy is loaded from `miniproject/rl/checkpoints/policy.zip`.
Override via the `policy_path` argument or COBAR_RL_POLICY env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from miniproject.simulation import MiniprojectSimulation
from flygym.examples.locomotion import TurningController
from flygym.compose import ActuatorType
from stable_baselines3 import PPO

from rl.cobar_env import _build_observation
from submission import movement_correction


_DEFAULT_POLICY = (
    Path(__file__).resolve().parent / "checkpoints" / "policy.zip"
)


class RLController:
    """Drop-in replacement for the heuristic Controller that uses a PPO
    policy. Same step() signature so run_headless.py can use it as-is.
    """

    def __init__(self, sim: MiniprojectSimulation, policy_path: str | None = None,
                 smooth_alpha: float = 0.7):
        self.turning_controller = TurningController(sim.timestep)
        if policy_path is None:
            policy_path = os.environ.get(
                "COBAR_RL_POLICY", str(_DEFAULT_POLICY)
            )
        if not Path(policy_path).exists():
            raise FileNotFoundError(
                f"Trained policy not found at {policy_path}. "
                f"Train one first with miniproject/rl/train.py"
            )
        self.policy = PPO.load(policy_path, device="auto")
        # Action smoothing must match CobarEnv.step(): the policy was trained
        # on smoothed actions, so deployment has to smooth identically.
        self.smooth_alpha = float(smooth_alpha)
        self._last_action = np.zeros(2, dtype=np.float32)
        self._smoothed = np.zeros(2, dtype=np.float32)
        # Keep _history minimally compatible so existing analysis scripts work
        from collections import deque
        self._history = deque(maxlen=15000)
        self.debug_info = {}

    def step(self, sim: MiniprojectSimulation):
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

        obs = _build_observation(
            olfaction, quat, pitch, roll, raw_vision,
            contact_forces_world, antenna_data, self._last_action,
        )
        action, _state = self.policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.8, 1.8).astype(np.float32)
        # Low-pass filter, identical to CobarEnv.step(): the policy was
        # trained on smoothed actions, so deployment must smooth the same
        # way, and the last_action fed back into the obs is the smoothed one.
        a = self.smooth_alpha
        self._smoothed = (1.0 - a) * self._smoothed + a * action
        self._last_action = self._smoothed

        joint_angles, adhesion = self.turning_controller.step(self._smoothed)

        # Minimal log entry for compatibility with existing analysis scripts
        self._history.append({
            "drive_L": float(self._smoothed[0]),
            "drive_R": float(self._smoothed[1]),
            "pitch": float(pitch),
            "roll": float(roll),
            "mode": "rl",
        })
        self.debug_info = {
            "pitch": float(pitch),
            "roll": float(roll),
            "drive_L": float(self._smoothed[0]),
            "drive_R": float(self._smoothed[1]),
            "mode": "rl",
        }
        return joint_angles, adhesion
