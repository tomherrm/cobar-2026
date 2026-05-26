"""Ground-truth segmentation from the fly's eye cameras.

This is *privileged* information — it reads MuJoCo geom IDs, which the fly does
not have access to. It is used ONLY to build training labels for the
segmentation CNN (`gen_seg_labels.py`). At deployment the controller uses the
CNN's prediction from raw vision alone; this module is never called.

Class labels: 0 = background, 1 = grass blade (obstacle), 2 = banana (goal).
"""
from __future__ import annotations

import re

import numpy as np
import mujoco
import yaml

from flygym import assets_dir

# Grass blades are created by GrassMixin.add_grass_blade with a body+geom both
# named uuid4().hex (32 hex chars). The banana is the fixed 'peel'/'flesh' geoms.
HEX32 = re.compile(r"^[0-9a-f]{32}$")

BG, GRASS, BANANA = 0, 1, 2


def classify_geoms(mj_model):
    """Return (grass_geom_ids, banana_geom_ids) for one compiled model.

    Must be called per-sim: geom IDs — and the grass uuids — differ between
    seeds (each seed places a different number of blades at fresh uuids).
    """
    grass, banana = [], []
    for gid in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if HEX32.match(name):
            grass.append(gid)
        elif name in ("peel", "flesh"):
            banana.append(gid)
    return np.array(sorted(grass)), np.array(sorted(banana))


class SegRenderer:
    """Renders ground-truth class masks from the fly's eye cameras, fisheye-
    corrected to align pixel-for-pixel with ``sim.get_raw_vision()``.

    One instance per sim — a MuJoCo Renderer binds to a specific compiled
    model. Call ``close()`` when done with the sim.
    """

    def __init__(self, sim, fly_name: str | None = None):
        self.sim = sim
        self.fly_name = fly_name or sim.fly.name
        self.m = sim.mj_model
        self.eye_cam_ids = sim._intern_eye_camera_ids_by_fly[self.fly_name]
        self.hidden_ids = sim._intern_hidden_segment_ids_by_fly[self.fly_name]
        self.retina = sim.world.fly_lookup[self.fly_name].retina
        self.grass_ids, self.banana_ids = classify_geoms(self.m)

        with open(assets_dir / "model/vision.yaml") as f:
            vc = yaml.safe_load(f)
        self.H = vc["raw_img_height_px"]
        self.W = vc["raw_img_width_px"]
        self.renderer = mujoco.Renderer(self.m, height=self.H, width=self.W)
        self.renderer.enable_segmentation_rendering()

    def render(self) -> list[np.ndarray]:
        """Class masks for both eyes — a list of (H, W) uint8 arrays,
        fisheye-corrected to match ``get_raw_vision()``.
        """
        # Hide the fly's own self-segments exactly as get_raw_vision() does,
        # so the mask aligns with the RGB the controller actually sees.
        alpha = self.m.geom_rgba[self.hidden_ids, 3].copy()
        self.m.geom_rgba[self.hidden_ids, 3] = 0
        masks = []
        for cam_id in self.eye_cam_ids:
            self.renderer.update_scene(self.sim.mj_data, cam_id)
            seg = self.renderer.render()             # (H, W, 2) int32: [id, type]
            objid = seg[:, :, 0]
            mask = np.zeros((self.H, self.W), dtype=np.uint8)
            mask[np.isin(objid, self.grass_ids)] = GRASS
            mask[np.isin(objid, self.banana_ids)] = BANANA
            # Fisheye correction is a nearest-neighbour pixel copy, so the
            # integer class labels survive intact (no interpolation).
            mask_fe = self.retina.correct_fisheye(
                np.repeat(mask[:, :, None], 3, axis=2)
            )[:, :, 0]
            masks.append(np.ascontiguousarray(mask_fe))
        self.m.geom_rgba[self.hidden_ids, 3] = alpha
        return masks

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
