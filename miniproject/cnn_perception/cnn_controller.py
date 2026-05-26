"""CNNController — the heuristic state machine with its perception replaced by
a trained segmentation CNN.

It subclasses the heuristic `Controller` and overrides `_perceive()`: instead
of RGB green-segmentation in the top image band, it runs the `SegCNN` on the
raw vision and derives the *same* feature contract (`col_blade`, `top_count`,
`n_central`, `L_blades`, `R_blades`) on the *same* scale — so the proven
NAVIGATE/SCAN/COMMIT/BACKUP state machine is unchanged — but the grass evidence
now comes from an accurate object-level segmentation rather than a colour
heuristic that cannot tell a blade from green terrain.

The rich object-level features from `seg_features.mask_to_features` (per-sector
density, proximity, gap bearing) are also attached to the perception dict as
`seg_feats`, for debugging and for a proximity-aware v2.

Usage (drop-in for the heuristic Controller):
    from cnn_perception.cnn_controller import CNNController
    controller = CNNController(sim)            # loads seg_cnn.pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

THIS_DIR = Path(__file__).resolve().parent
MP_ROOT = THIS_DIR.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from submission.controller import Controller
from cnn_perception.seg_model import load_seg_cnn
from cnn_perception.seg_features import mask_to_features, GRASS

_DEFAULT_CNN = THIS_DIR / "seg_cnn.pt"


class CNNController(Controller):
    """Heuristic controller with CNN-segmentation perception."""

    def __init__(self, sim, cnn_path: str | None = None, cnn_res: int = 128):
        super().__init__(sim)
        if cnn_path is None:
            cnn_path = os.environ.get("COBAR_SEG_CNN", str(_DEFAULT_CNN))
        if not Path(cnn_path).exists():
            raise FileNotFoundError(
                f"Segmentation CNN not found at {cnn_path}. "
                f"Train one with miniproject/cnn_perception/train_seg_cnn.py"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seg_cnn = load_seg_cnn(cnn_path, device=self.device)
        self.cnn_res = cnn_res
        # A column "carries a blade" when it has more than this many grass
        # pixels (full column height of the CNN mask). The CNN mask is clean
        # (real blade geoms, not RGB-green noise), so this fires reliably on
        # genuine blades. From the _calibrate.py threshold sweep.
        self.cnn_blade_count_threshold = 12
        # The CNN sees grass far more reliably than the RGB heuristic, so the
        # blade-count features land on a larger scale. Retune the two
        # scale-dependent state-machine params accordingly; everything else in
        # the state machine is scale-invariant (centering asymmetry) or has a
        # timeout fallback (SCAN -> COMMIT after scan_max_steps).
        self.lplc1_full_slow_at_n_blades = 40   # was 80 (RGB 450-wide scale)
        self.scan_clear_count = 3               # was 0 (RGB n_central ~always 0)
        # v3: goal-aware continuous gap-following in NAVIGATE.
        # The hard part is distinguishing "a blade is BLOCKING my path" from
        # "grass is visible on the horizon" — density alone can't (the central
        # sectors always carry ~5% distant horizon grass), nor can proximity
        # alone (high baseline). The discriminator: a NEAR obstacle has grass
        # density AND that grass extends LOW in the image (high proximity).
        # near = density * clip((proximity - 0.55) / 0.45, 0, 1) — distant
        # horizon grass (proximity ~0.5) scores ~0; a near blade scores high.
        # Avoidance is then a BOUNDED small turn (a perturbation on the odor
        # pull, not an override) toward the minimal-deviation gap.
        self.avoid_near_threshold = 0.004          # central near-obstacle score to engage
        self.avoid_center_bias = 0.010             # preference for minimal deviation
        self.avoid_gain = 0.6                      # steering strength
        self.avoid_turn_clip = 0.5                 # avoid_drive turn magnitude cap
        self.avoid_full_near = 0.025               # central near-score at which urgency = 1
        self.avoid_slow = 0.4                      # max slowdown fraction at full urgency

    @torch.no_grad()
    def _segment(self, raw_vision) -> list[np.ndarray]:
        """Run the CNN on both eyes -> list of 2 class masks at cnn_res."""
        masks = []
        for eye in raw_vision:
            small = cv2.resize(eye, (self.cnn_res, self.cnn_res),
                               interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(small).permute(2, 0, 1).float()[None] / 255.0
            logits = self.seg_cnn(x.to(self.device))
            masks.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
        return masks

    def _perceive(self, raw_vision):
        """Override: CNN segmentation in place of RGB green-segmentation.

        Returns the same contract as Controller._perceive (so the state
        machine is unchanged), plus `seg_feats` with the rich object-level
        features for debugging / future use.

        Unlike the RGB heuristic, grass is counted over the FULL column, not
        a top band: the RGB top-band restriction existed only to reject green
        terrain, which the CNN already does — and the CNN puts grass around
        the horizon (mid-image), which the top band would miss entirely.
        """
        arr = np.asarray(raw_vision)
        masks_small = self._segment(arr)            # 2 x (cnn_res, cnn_res) uint8
        is_grass = np.stack([(m == GRASS) for m in masks_small])  # (2, res, res)

        # per-column grass-pixel count over the full column height
        top_count = is_grass.sum(axis=1)                            # (2, res)
        col_blade = top_count > self.cnn_blade_count_threshold       # (2, res)

        W = col_blade.shape[1]
        half_band = max(1, int(W * self.rgb_central_fraction / 2))
        c_lo = W // 2 - half_band
        c_hi = W // 2 + half_band
        n_central = int(col_blade[:, c_lo:c_hi].any(axis=0).sum())

        # Rich object-level features (per-sector density, proximity, gap
        # bearing). Not consumed by the base state machine, but attached for
        # debugging and a proximity-aware v2.
        seg_feats = mask_to_features(masks_small[0], masks_small[1])

        return {
            "col_blade": col_blade,
            "top_count": top_count,
            "n_central": n_central,
            "L_blades": float(col_blade[0].sum()),
            "R_blades": float(col_blade[1].sum()),
            "seg_feats": seg_feats,
        }

    def _navigate_obstacle_avoidance(self, perc):
        """v3 override: goal-aware continuous gap-follower.

        v1 swapped the perception source but consumed only blade-presence
        counts -> behaved like the heuristic. v2 steered toward the absolute
        clearest sector -> goal-blind, made no progress. v3 makes the MINIMAL
        deviation that clears the path ahead: it steers toward the clearest
        sector weighted toward straight-ahead, and only when the central
        sectors are actually blocked — so a clear path lets the odor pull
        drive straight at the goal, and an obstacle gets routed around with
        the smallest heading change that works.
        """
        seg = perc.get("seg_feats")
        if seg is None:
            return np.zeros(2, dtype=np.float32), 1.0

        density = np.asarray(seg["sector_density"], dtype=np.float32)
        proximity = np.asarray(seg["sector_proximity"], dtype=np.float32)
        n = len(density)
        center = (n - 1) / 2.0
        central = [int(np.floor(center)), int(np.ceil(center))]

        # per-sector "near obstacle" score: grass density that extends LOW in
        # the image. Distant horizon grass (proximity ~0.5) scores ~0; a near
        # blade (high density, proximity -> 1) scores high.
        near = density * np.clip((proximity - 0.55) / 0.45, 0.0, 1.0)
        center_near = float(near[central].mean())
        if center_near < self.avoid_near_threshold:
            return np.zeros(2, dtype=np.float32), 1.0       # path ahead clear

        # minimal-deviation gap: lowest-near sector, biased toward center
        sectors = np.arange(n)
        desirability = -near - self.avoid_center_bias * np.abs(sectors - center)
        target = int(np.argmax(desirability))
        bearing = (target - center) / center                # -1 left .. +1 right
        # bounded small turn — a perturbation on the odor pull, not an
        # override. [+turn, -turn] turns right (heuristic centering convention).
        turn = float(np.clip(bearing * self.avoid_gain,
                             -self.avoid_turn_clip, self.avoid_turn_clip))
        avoid_drive = np.array([turn, -turn], dtype=np.float32)
        urgency = min(1.0, center_near / self.avoid_full_near)
        speed_factor = 1.0 - self.avoid_slow * urgency
        return avoid_drive, speed_factor
