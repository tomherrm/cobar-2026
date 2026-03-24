from typing import Callable, Iterable

import numpy as np
from flygym import Simulation, assets_dir
from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, BaseWorld, Fly
from flygym.compose.pose import KinematicPose
from flygym.compose.world import FlatGroundWorld

LEG_NAMES: list[str] = [f"{side}{pos}" for side in "lr" for pos in "fmh"]


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) in radians to the interval [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def create_fly(
    enable_vision: bool = False,
    enable_olfaction: bool = False,
    adhesion_gain: float = 50,
    position_gain: float = 50,
    adhesion_segments: Iterable[str] | None = tuple(
        f"{leg}_tarsus5" for leg in LEG_NAMES
    ),
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL,
    joint_preset: JointPreset = JointPreset.LEGS_ONLY,
    dof_preset: ActuatedDOFPreset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY,
    actuator_type: ActuatorType = ActuatorType.POSITION,
    neutral_pose_path=assets_dir / "model/pose/neutral.yaml",
    **kwargs,
):
    fly = Fly(**kwargs)
    skeleton = Skeleton(axis_order=axis_order, joint_preset=joint_preset)
    neutral_pose = KinematicPose(path=neutral_pose_path)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = fly.skeleton.get_actuated_dofs_from_preset(dof_preset)
    fly.add_actuators(
        actuated_dofs,
        actuator_type=actuator_type,
        kp=position_gain,
        neutral_input=neutral_pose,
    )

    if enable_olfaction:
        fly.add_odor_sensors()

    if enable_vision:
        fly.add_vision()

    if adhesion_segments is not None:
        adhesion_segments = [
            seg if isinstance(seg, BodySegment) else BodySegment(seg)
            for seg in adhesion_segments
        ]
        fly.add_adhesion_actuators(segments=adhesion_segments, gain=adhesion_gain)

    fly.add_force_sensors()
    fly.colorize()
    return fly


def show_video(sim: Simulation, title: str | None = None) -> None:
    """Display the first renderer's video inline in the notebook."""
    sim.renderer.show_in_notebook(title=title)


class History:
    """Container for trial time series recorded during simulation."""

    def __init__(self, n_steps: int, sim: Simulation, fly_name: str):
        self.pos = np.zeros((n_steps, 2), dtype=np.float32)
        self.heading = np.zeros(n_steps, dtype=np.float32)
        self.rel_leg_tip_pos = np.zeros((n_steps, 6), dtype=np.float32)
        self.adhesion_force = np.zeros((n_steps, 6), dtype=np.float32)

        body = sim.mj_data.body(f"{fly_name}/")
        self.curr_quat = body.xquat
        self.curr_pos = body.xpos[:2]
        legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
        self.tarsus_ids = np.array(
            [sim.mj_data.body(f"{fly_name}/{leg}_tarsus5").id for leg in legs]
        )
        self.fly_name = fly_name
        self.sim = sim
        self.xpos = sim.mj_data.xpos[:, :2]

    def get_heading_vec(self) -> tuple[float, float]:
        """Return the heading unit vector projected to the ground plane."""
        w, x, y, z = self.curr_quat
        return 1.0 - 2.0 * (y * y + z * z), 2.0 * (w * z + x * y)

    def get_rel_leg_tip_pos(self, heading_vec: tuple[float, float]) -> np.ndarray:
        """Project each tarsus tip position onto the heading axis."""
        leg_tip_pos = self.xpos[self.tarsus_ids]
        return (leg_tip_pos - self.curr_pos) @ heading_vec

    def step(self, i: int) -> None:
        """Record state variables for one simulation step."""
        heading_vec = self.get_heading_vec()
        self.rel_leg_tip_pos[i] = self.get_rel_leg_tip_pos(heading_vec)
        self.pos[i] = self.curr_pos
        self.heading[i] = np.arctan2(heading_vec[1], heading_vec[0])
        self.adhesion_force[i] = self.sim.get_adhesion_force_magnitudes(self.fly_name)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return recorded history arrays in a single dictionary."""
        return {
            "pos": self.pos,
            "heading": self.heading,
            "rel_leg_tip_pos": self.rel_leg_tip_pos,
            "adhesion_force": self.adhesion_force,
        }


def get_variables(
    data: dict[str, np.ndarray],
    win_len: int,
    adhesion_force_thr: float,
) -> dict[str, np.ndarray]:
    """
    Extract variables used for path integration from trial data.
    The difference between ``load_trial_data`` and ``extract_variables`` is
    that the former loads the raw data from the trial (i.e., physics
    simulation). The latter extracts variables from these raw data subject
    to additional parameters such as time scale. For each trial, we only
    call ``load_trial_data`` once, but we may call ``extract_variables``
    multiple times with different parameters.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Dictionary containing trial data.
    win_len : int
        Length of the time window for path integration.
    adhesion_force_thr : float
        Threshold for adhesion forces. These are used to determine whether
        a leg is in contact with the ground.
    """
    if win_len <= 0:
        raise ValueError("win_len must be a positive integer.")

    if data["pos"].shape[0] <= win_len:
        raise ValueError("Data length must be greater than win_len.")

    def win_diff(a: np.ndarray) -> np.ndarray:
        return a[win_len:] - a[:-win_len]

    pos = data["pos"]
    heading = data["heading"]
    stride_diff_unmasked = np.diff(data["rel_leg_tip_pos"], axis=0, prepend=0)
    adhesion_force = data["adhesion_force"]

    # Mechanosensory signal ==========
    # Calculate cumulative stride (Σstride) for each side
    mask = adhesion_force >= adhesion_force_thr
    stride_cum = np.cumsum(mask * stride_diff_unmasked, axis=0)

    # Calculate difference in Σstride over proprioceptive time window (ΔΣstride)
    stride_win_diff = win_diff(stride_cum)

    # Calculate sum and difference in ΔΣstride over two sides
    stride_win_diff_l = stride_win_diff[:, :3]
    stride_win_diff_r = stride_win_diff[:, 3:6]
    stride_win_diff_lr_sum = stride_win_diff_l + stride_win_diff_r
    stride_win_diff_lr_diff = stride_win_diff_l - stride_win_diff_r

    # Change in locomotion state (heading & displacement) ==========
    # Calculate change in fly orientation over proprioceptive time window (Δheading)
    heading_win_diff = win_diff(heading)
    heading_win_diff = wrap_to_pi(heading_win_diff)

    # Same for displacement projected in the direction of fly's heading
    pos_diff = np.diff(pos, axis=0, prepend=0)
    heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    fwd_disp_mag = np.einsum("ij,ij->i", pos_diff, heading_vec)
    fwd_disp_cum = np.cumsum(fwd_disp_mag)
    fwd_disp_win_diff = win_diff(fwd_disp_cum)

    return {
        "stride_win_diff_lr_sum": stride_win_diff_lr_sum,
        "stride_win_diff_lr_diff": stride_win_diff_lr_diff,
        "heading_win_diff": heading_win_diff,
        "fwd_disp_win_diff": fwd_disp_win_diff,
    }

def path_integrate(
    data: dict[str, np.ndarray],
    heading_model: Callable[[np.ndarray], np.ndarray],
    fwd_disp_model: Callable[[np.ndarray], np.ndarray],
    win_len: int,
    adhesion_force_thr: float,
) -> dict[str, np.ndarray]:
    """
    Perform path integration on trial data.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Dictionary containing trial data.
    heading_model : Callable
        Model for predicting change in heading.
    fwd_disp_model : Callable
        Model for predicting change in forward displacement.
    win_len : int
        Length of the time window for path integration.
    adhesion_force_thr : float
        Threshold for adhesion forces. These are used to determine whether
        a leg is in contact with the ground.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the following keys:
        * "heading_pred": Predicted heading.
        * "heading_actual": Actual heading.
        * "pos_pred": Predicted position.
        * "pos_actual": Actual position.
        * "heading_diff_pred": Predicted change in heading.
        * "heading_diff_actual": Actual change in heading.
        * "disp_diff_pred": Predicted change in displacement.
        * "disp_diff_actual": Actual change in displacement.
    """
    variables = get_variables(data, win_len, adhesion_force_thr)

    # Integrate heading
    heading_win_diff_pred = np.asarray(
        heading_model(variables["stride_win_diff_lr_diff"]), dtype=np.float64
    )
    heading_pred = np.cumsum(heading_win_diff_pred / win_len)
    # Start from the real heading at the moment when path integration actually starts.
    real_heading_start = data["heading"][win_len]
    heading_pred += real_heading_start

    # Integrate displacement
    fwd_disp_win_diff_pred = np.asarray(
        fwd_disp_model(variables["stride_win_diff_lr_sum"]), dtype=np.float64
    )
    heading_vec_pred = np.stack([np.cos(heading_pred), np.sin(heading_pred)], axis=-1)
    disp_pred = fwd_disp_win_diff_pred[..., None] * heading_vec_pred / win_len
    pos_pred = np.cumsum(disp_pred, axis=0) + data["pos"][win_len]

    # Pad with NaN where prediction not available
    padding = np.full(win_len, np.nan)
    heading_pred = np.concatenate([padding, heading_pred])
    pos_pred = np.concatenate([np.full((win_len, 2), np.nan), pos_pred], axis=0)
    heading_win_diff_pred = np.concatenate([padding, heading_win_diff_pred])
    heading_win_diff_actual = np.concatenate([padding, variables["heading_win_diff"]])
    fwd_disp_win_diff_pred = np.concatenate([padding, fwd_disp_win_diff_pred])
    fwd_disp_win_diff_actual = np.concatenate([padding, variables["fwd_disp_win_diff"]])

    return {
        "heading_pred": heading_pred,
        "heading_actual": data["heading"],
        "pos_pred": pos_pred,
        "pos_actual": data["pos"],
        "heading_win_diff_pred": heading_win_diff_pred,
        "heading_win_diff_actual": heading_win_diff_actual,
        "fwd_disp_win_diff_pred": fwd_disp_win_diff_pred,
        "fwd_disp_win_diff_actual": fwd_disp_win_diff_actual,
    }

class BaseBallArena(FlatGroundWorld):
    def __init__(
        self,
        half_size: float = 50,
        seed: int = 0,
        cues_radius: float = 1.0,
        cues_height: float = 300,
        cues_radius_interval: tuple[int, int] = (50, 60),
        n_cues: int = 20,
    ):
        super().__init__(half_size=half_size)
        self.rng = np.random.default_rng(seed=seed)
        self.cues = []
        self.set_baseball_floor()
        self.add_visual_cues(cues_radius, cues_height, cues_radius_interval, n_cues)

    def add_visual_cues(
        self, cues_radius, cues_height, cues_radius_interval, n_cues, seed=None
    ):
        rng = self.rng if seed is None else np.random.default_rng(seed=seed)
        for i in range(n_cues):
            r = rng.uniform(*cues_radius_interval)
            theta = rng.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            rgba = (*rng.uniform(0, 1, 3), 1)
            self.add_cue(x, y, cues_radius, cues_height, rgba)

    def add_cue(self, x, y, r, h, rgba=(0.25, 0.25, 0.25, 1)):
        # Note that here we are not attaching the cue to a specific
        i = len(self.cues)
        cue = self.mjcf_root.worldbody.add(
            "geom",
            type="cylinder",
            name=f"cue_{i}",
            pos=(x, y, 0),
            size=(r, h / 2),
            rgba=rgba,
            contype=0,  # This can be used to set contacts see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-contype
            # Contacts are taking up a lot of computation time, so disabling them for objects that we do not need to interact with is good practice
        )
        self.cues.append(cue)

    def set_baseball_floor(self):
        baseball_texture = self.mjcf_root.asset.add(
            "texture",
            type="2d",
            file="assets/ground_texture.png",
        )
        grid = self.mjcf_root.asset.add(
            "material",
            name="baseball_floor",
            texture=baseball_texture,
            texrepeat=(1, 1),
        )
        self.ground_contact_geoms[0].material = grid
