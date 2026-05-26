"""Turn a pair of eye segmentation masks into object-level features for the
state-machine controller.

This is the bridge that makes the heuristic+CNN idea work: the old RGB pipeline
gave the controller only flat pixel summaries ("how many green pixels"); these
features give it *spatial structure* — where obstacles are, how close, where
the gap is — which is exactly the object-level representation the heuristic
lacked. The same function works on ground-truth masks (for testing) or the
CNN's predicted masks (at deployment).

Mask convention: (H, W) uint8, 0 = background, 1 = grass blade, 2 = banana.
Eye order: index 0 = left eye, index 1 = right eye.
"""
from __future__ import annotations

import numpy as np

GRASS, BANANA = 1, 2


def _sector_stats(mask: np.ndarray, n_sectors: int):
    """Per-sector grass density and proximity for one eye mask.

    density   : fraction of pixels in the sector that are grass [0, 1].
    proximity : how close the nearest blade in the sector is [0, 1] — the
                lowest grass row in the sector, normalised; near blades reach
                further down the image. 0 if the sector has no grass.
    """
    H, W = mask.shape
    edges = np.linspace(0, W, n_sectors + 1, dtype=int)
    density = np.zeros(n_sectors, dtype=np.float32)
    proximity = np.zeros(n_sectors, dtype=np.float32)
    rows = np.arange(H)[:, None]
    for s in range(n_sectors):
        c0, c1 = edges[s], edges[s + 1]
        sub = mask[:, c0:c1]
        is_grass = sub == GRASS
        density[s] = is_grass.mean()
        if is_grass.any():
            # lowest (largest-row-index) grass pixel, normalised
            proximity[s] = float((rows * is_grass).max()) / max(H - 1, 1)
    return density, proximity


def mask_to_features(mask_l: np.ndarray, mask_r: np.ndarray,
                     n_sectors_per_eye: int = 4) -> dict:
    """Object-level features from the two eye masks.

    Returns a dict (all scalars / small arrays) with:
      grass_frac_l/r       overall grass coverage per eye
      grass_central        grass coverage in the central sectors (forward)
      grass_left/right     coverage on the left vs right half of the field
      obstacle_asymmetry   grass_right - grass_left  (>0: obstacle is to the right)
      nearest_proximity    proximity of the closest blade anywhere [0, 1]
      central_proximity    proximity of the closest blade straight ahead [0, 1]
      sector_density       (2*n_sectors_per_eye,) density, left eye then right eye
      sector_proximity     (2*n_sectors_per_eye,) proximity, same layout
      clearest_sector      index of the lowest-cost (most open) sector
      clearest_bearing     clearest_sector mapped to [-1, +1] (-1 left .. +1 right)
      banana_visible       True if any banana pixels are seen
      banana_bearing       banana centroid bearing [-1, +1], 0 if not visible
    """
    k = n_sectors_per_eye
    dl, pl = _sector_stats(mask_l, k)
    dr, pr = _sector_stats(mask_r, k)

    sector_density = np.concatenate([dl, dr])      # left eye .. right eye
    sector_proximity = np.concatenate([pl, pr])
    n_tot = 2 * k

    # central sectors = the two innermost (right side of L eye, left of R eye)
    central = [k - 1, k]
    grass_central = float(sector_density[central].mean())
    central_proximity = float(sector_proximity[central].max())

    grass_left = float(sector_density[:k].mean())
    grass_right = float(sector_density[k:].mean())

    # steering target: the sector with the lowest "threat" (density + proximity)
    cost = sector_density + sector_proximity
    clearest_sector = int(np.argmin(cost))
    clearest_bearing = (clearest_sector / (n_tot - 1)) * 2.0 - 1.0

    # banana bearing from its centroid column across both eyes
    banana_visible = False
    banana_bearing = 0.0
    cols_l = np.where((mask_l == BANANA).any(axis=0))[0]
    cols_r = np.where((mask_r == BANANA).any(axis=0))[0]
    if len(cols_l) or len(cols_r):
        banana_visible = True
        W = mask_l.shape[1]
        # left eye spans bearing [-1, 0], right eye spans [0, +1]
        bearings = []
        if len(cols_l):
            bearings.append((cols_l.mean() / max(W - 1, 1)) - 1.0)
        if len(cols_r):
            bearings.append(cols_r.mean() / max(W - 1, 1))
        banana_bearing = float(np.mean(bearings))

    return {
        "grass_frac_l": float((mask_l == GRASS).mean()),
        "grass_frac_r": float((mask_r == GRASS).mean()),
        "grass_central": grass_central,
        "grass_left": grass_left,
        "grass_right": grass_right,
        "obstacle_asymmetry": grass_right - grass_left,
        "nearest_proximity": float(sector_proximity.max()),
        "central_proximity": central_proximity,
        "sector_density": sector_density,
        "sector_proximity": sector_proximity,
        "clearest_sector": clearest_sector,
        "clearest_bearing": float(clearest_bearing),
        "banana_visible": banana_visible,
        "banana_bearing": banana_bearing,
    }
