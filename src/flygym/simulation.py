from operator import call
from functools import cached_property, cache
from collections import defaultdict
from typing import Any

import mujoco
import dm_control.mjcf as mjcf
import numpy as np
from jaxtyping import Float
from time import perf_counter_ns
from collections.abc import Sequence

from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.rendering import Renderer, CameraSpec
from flygym.utils.profiling import print_perf_report


class Simulation:
    def __init__(self, world: BaseWorld) -> None:
        if len(world.fly_lookup) == 0:
            raise ValueError("The world must contain at least one fly.")
        self.renderer: Renderer | None = None
        self.world = world
        self.mj_model, self.mj_data = world.compile()

        # Map internal IDs in the compiled MuJoCo model. This allows users to read from
        # or write to body/joint/actuator in orders defined by Fly objects.
        self._map_internal_bodyids()
        self._map_internal_qposqveladrs()
        self._map_internal_actuator_ids()
        self._map_internal_odor_sensor_ids()
        self._map_internal_eye_camera_ids()
        self._map_internal_hidden_segment_ids()
        self._map_internal_contact_body_segment_ids()
        self._map_internal_adhesion_geom_ids()

        # For performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0
        self._last_vision_render_time = None

        # Reset everything (physics, renderers, and profiling stats)
        self.reset()

    def reset(self) -> None:
        # Reset physics
        neutral_keyframe_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "neutral"
        )
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, neutral_keyframe_id)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Reset renderers
        if self.renderer is not None:
            self.renderer.reset(self.mj_data)

        # Stuff for performance profiling
        self._curr_step = 0
        self._frames_rendered = 0
        self._total_physics_time_ns = 0
        self._total_render_time_ns = 0
        self._last_vision_render_time = None

    def step(self) -> None:
        physics_start_ns = perf_counter_ns()
        mujoco.mj_step(self.mj_model, self.mj_data)
        physics_finish_ns = perf_counter_ns()
        self._total_physics_time_ns += physics_finish_ns - physics_start_ns
        self._curr_step += 1

        if self._last_vision_render_time is not None:
            if self.time - self._last_vision_render_time >= 1 / 500:
                self.get_raw_vision.cache_clear()
                self.get_ommatidia_readouts.cache_clear()

    def set_renderer(
        self,
        camera: CameraSpec | Sequence[CameraSpec],
        *,
        camera_res: tuple[int, int] = (240, 320),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        **kwargs: Any,
    ) -> Renderer:
        self.renderer = Renderer(
            self.mj_model,
            camera,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            **kwargs,
        )
        return self.renderer

    def render_as_needed(self) -> bool:
        render_start_ns = perf_counter_ns()
        render_done = self.renderer.render_as_needed(self.mj_data)
        render_finish_ns = perf_counter_ns()
        self._total_render_time_ns += render_finish_ns - render_start_ns
        if render_done:
            self._frames_rendered += 1
        return render_done

    def get_joint_angles(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        internal_ids = self._intern_qposadrs_by_fly[fly_name]
        return self.mj_data.qpos[internal_ids]

    def get_joint_velocities(self, fly_name: str) -> Float[np.ndarray, "n_jointdofs"]:
        internal_ids = self._intern_qveladrs_by_fly[fly_name]
        return self.mj_data.qvel[internal_ids]

    def get_body_positions(self, fly_name: str) -> Float[np.ndarray, "n_bodies 3"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xpos[internal_ids, :]

    def get_body_rotations(self, fly_name: str) -> Float[np.ndarray, "n_bodies 4"]:
        internal_ids = self._internal_bodyids_by_fly[fly_name]
        return self.mj_data.xquat[internal_ids, :]

    def get_olfaction(
        self, fly_name: str, **kwargs
    ) -> Float[np.ndarray, "n_sensors n_odor_dimensions"]:
        if callable(getattr(self.world, "get_olfaction", None)):
            internal_ids = self._intern_odor_sensorids_by_fly[fly_name]
            indices = self.mj_model.sensor_adr[internal_ids][:, None] + np.arange(3)
            sensor_positions = self.mj_data.sensordata[indices]
            return getattr(self.world, "get_olfaction")(sensor_positions, **kwargs)
        else:
            raise NotImplementedError("The current world does not support olfaction.")

    def get_adhesion_force_magnitudes(
        self, fly_name: str
    ) -> Float[np.ndarray, "n_contact_sensors"]:
        adhesion_actuator_ids = self._intern_actuatorids_by_type_by_fly[
            ActuatorType.ADHESION
        ][fly_name]
        return self.mj_data.actuator_force[adhesion_actuator_ids]

    def get_external_force(
        self,
        fly_name: str,
        subtract_adhesion_force: bool,
    ) -> Float[np.ndarray, "n_contact_sensors 3"]:
        body_ids = self._internal_contact_body_segment_ids_by_fly[fly_name]
        contact_forces = self.mj_data.cfrc_ext[body_ids, 3:]

        if subtract_adhesion_force:
            adhesion_force_magnitudes = self.get_adhesion_force_magnitudes(fly_name)
            adhesion_geom_ids = self._intern_adhesion_geom_ids_by_fly[fly_name]

            for i, geom_id in enumerate(adhesion_geom_ids):
                mask = self.mj_data.contact.geom2 == geom_id
                if mask.any():
                    normal = self.mj_data.contact.frame[mask, :3].mean(0)
                    contact_forces[i] -= normal * adhesion_force_magnitudes[i]

        return contact_forces

    def get_antenna_data(self, fly_name: str) -> dict[str, dict[str, np.ndarray]]:
        return_dict = {}
        for side in ["l", "r"]:
            try:
                data = self.mj_data.joint(f"{fly_name}/{side}_funiculus_ball_joint")
            except KeyError:
                raise NotImplementedError(
                    "The fly does not have the expected antenna joints. "
                    "Make sure to add ball joints to the funiculus segments in the fly construction "
                    "by calling fly.add_antenna_joints()"
                )
            return_dict[side] = {
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "qacc": data.qacc.copy(),
                "qfrc_passive": data.qfrc_passive.copy(),
            }
        return return_dict

    @cached_property
    def eye_renderer(self):
        from flygym import assets_dir
        import yaml

        with open(assets_dir / "model/vision.yaml", "r") as f:
            vision_config = yaml.safe_load(f)

        renderer = mujoco.Renderer(
            self.mj_model,
            height=vision_config["raw_img_height_px"],
            width=vision_config["raw_img_width_px"],
        )
        return renderer

    @cache
    def get_raw_vision(
        self, fly_name: str
    ) -> list[Float[np.ndarray, "height width 3"]]:
        self._last_vision_render_time = self.time
        internal_hidden_segment_ids = self._intern_hidden_segment_ids_by_fly[fly_name]
        alpha = self.mj_model.geom_rgba[internal_hidden_segment_ids, 3].copy()
        # Hide hidden segments by setting alpha to 0
        self.mj_model.geom_rgba[internal_hidden_segment_ids, 3] = 0
        internal_eye_camera_ids = self._intern_eye_camera_ids_by_fly[fly_name]
        frames = []
        retina = self.world.fly_lookup[fly_name].retina

        for cam_id in internal_eye_camera_ids:
            self.eye_renderer.update_scene(self.mj_data, cam_id)
            raw_frame = self.eye_renderer.render()
            fish_img = retina.correct_fisheye(raw_frame)
            frames.append(fish_img)

        # Restore original alpha values
        self.mj_model.geom_rgba[internal_hidden_segment_ids, 3] = alpha
        return frames

    @cache
    def get_ommatidia_readouts(
        self, fly_name: str
    ) -> Float[np.ndarray, "n_cameras n_ommatidia 2"]:
        raw_vision = self.get_raw_vision(fly_name)
        retina = self.world.fly_lookup[fly_name].retina
        ommatidia_readouts = np.array(
            [retina.raw_image_to_hex_pxls(image) for image in raw_vision],
            dtype=np.float32,
        )
        return ommatidia_readouts

    def set_actuator_inputs(
        self,
        fly_name: str,
        actuator_type: ActuatorType,
        inputs: Float[np.ndarray, "n_actuators"],
    ):
        internal_ids = self._intern_actuatorids_by_type_by_fly[actuator_type][fly_name]
        if len(inputs) != len(internal_ids):
            raise ValueError(
                f"Expected {len(internal_ids)} inputs for actuator type "
                f"'{actuator_type.name}', but got {len(inputs)}"
            )
        self.mj_data.ctrl[internal_ids] = inputs

    def _map_internal_bodyids(self) -> None:
        internal_bodyids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for bodyseg, mjcf_body_element in fly.bodyseg_to_mjcfbody.items():
                internal_body_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    mjcf_body_element.full_identifier,
                )
                internal_bodyids_by_fly[fly_name].append(internal_body_id)

        self._internal_bodyids_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_bodyids_by_fly.items()
        }

    def _map_internal_qposqveladrs(self) -> None:
        internal_qposadrs_by_fly = defaultdict(list)
        internal_qveladrs_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for jointdof, mjcf_joint_element in fly.jointdof_to_mjcfjoint.items():
                internal_joint_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    mjcf_joint_element.full_identifier,
                )
                qposadr = self.mj_model.jnt_qposadr[internal_joint_id]
                qveladr = self.mj_model.jnt_dofadr[internal_joint_id]
                internal_qposadrs_by_fly[fly_name].append(qposadr)
                internal_qveladrs_by_fly[fly_name].append(qveladr)

        self._intern_qposadrs_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_qposadrs_by_fly.items()
        }
        self._intern_qveladrs_by_fly = {
            k: np.array(v, dtype=np.int32) for k, v in internal_qveladrs_by_fly.items()
        }

    def _map_internal_actuator_ids(self) -> None:
        internal_actuatorids_by_fly_by_type = defaultdict(lambda: defaultdict(list))

        for fly_name, fly in self.world.fly_lookup.items():
            for actuator_ty, actuators in fly.jointdof_to_mjcfactuator_by_type.items():
                for jointdof, actuator_element in actuators.items():
                    internal_actuator_id = mujoco.mj_name2id(
                        self.mj_model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        actuator_element.full_identifier,
                    )
                    internal_actuatorids_by_fly_by_type[actuator_ty][fly_name].append(
                        internal_actuator_id
                    )

            for actuator_element in fly.bodyseg_to_mjcfactuator.values():
                internal_actuator_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_element.full_identifier,
                )
                internal_actuatorids_by_fly_by_type[ActuatorType.ADHESION][
                    fly_name
                ].append(internal_actuator_id)

        self._intern_actuatorids_by_type_by_fly = {
            actuator_ty: {
                fly_name: np.array(ids, dtype=np.int32)
                for fly_name, ids in ids_by_fly.items()
            }
            for actuator_ty, ids_by_fly in internal_actuatorids_by_fly_by_type.items()
        }

    def _map_internal_odor_sensor_ids(self) -> None:
        internal_odor_sensorids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for odor_sensor_element in fly.odorsensorname_to_mjcfsensor.values():
                internal_odor_sensor_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_SENSOR,
                    odor_sensor_element.full_identifier,
                )
                internal_odor_sensorids_by_fly[fly_name].append(internal_odor_sensor_id)

        self._intern_odor_sensorids_by_fly = {
            k: np.array(v, dtype=np.int32)
            for k, v in internal_odor_sensorids_by_fly.items()
        }

    def _map_internal_eye_camera_ids(self):
        internal_eye_camera_ids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for eye_camera_element in fly.eyecameraname_to_mjcfcamera.values():
                internal_eye_camera_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    eye_camera_element.full_identifier,
                )
                internal_eye_camera_ids_by_fly[fly_name].append(internal_eye_camera_id)

        self._intern_eye_camera_ids_by_fly = {
            k: np.array(v, dtype=np.int32)
            for k, v in internal_eye_camera_ids_by_fly.items()
        }

    def _map_internal_hidden_segment_ids(self):
        internal_hidden_segment_ids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for hidden_segment_element in fly.hiddenbodyseg_to_mjcfgeom.values():
                internal_hidden_segment_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    hidden_segment_element.full_identifier,
                )
                internal_hidden_segment_ids_by_fly[fly_name].append(
                    internal_hidden_segment_id
                )

        self._intern_hidden_segment_ids_by_fly = {
            k: np.array(v, dtype=np.int32)
            for k, v in internal_hidden_segment_ids_by_fly.items()
        }

    def _map_internal_contact_body_segment_ids(self):
        internal_contact_body_segment_ids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for body_segment in fly.contactbodyseg_to_mjcfbody.values():
                internal_body_segment_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_segment.full_identifier,
                )
                internal_contact_body_segment_ids_by_fly[fly_name].append(
                    internal_body_segment_id
                )

        self._internal_contact_body_segment_ids_by_fly = {
            k: np.array(v, dtype=np.int32)
            for k, v in internal_contact_body_segment_ids_by_fly.items()
        }

    def _map_internal_adhesion_geom_ids(self):
        internal_adhesion_geom_ids_by_fly = defaultdict(list)

        for fly_name, fly in self.world.fly_lookup.items():
            for geom in fly.adhesionbodyseg_to_mjcfgeom.values():
                internal_geom_id = mujoco.mj_name2id(
                    self.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom.full_identifier,
                )
                internal_adhesion_geom_ids_by_fly[fly_name].append(internal_geom_id)

        self._intern_adhesion_geom_ids_by_fly = {
            k: np.array(v, dtype=np.int32)
            for k, v in internal_adhesion_geom_ids_by_fly.items()
        }

    @property
    def time(self) -> float:
        return self.mj_data.time

    @property
    def timestep(self) -> float:
        return self.mj_model.opt.timestep

    def print_performance_report(self) -> None:
        print_perf_report(
            n_steps=self._curr_step,
            n_frames_rendered=self._frames_rendered,
            total_physics_time_ns=self._total_physics_time_ns,
            total_render_time_ns=self._total_render_time_ns,
            timestep=self.mj_model.opt.timestep,
        )
