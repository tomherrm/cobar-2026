import numpy as np
from scipy.spatial.transform import Rotation
from flygym.vision.retina import Retina


def tilt_to_control_signal(quat, k_pitch=0.02, k_roll=0.01, max_pitch_boost=0.2, max_roll_boost=0.1):
    """
    Converts the fly's body tilt into a CPG control signal to maintain stability on hills.

    Parameters:
    - quat (np.ndarray): Quaternion [w, x, y, z] of the fly's thorax orientation.
    - k_pitch (float): Proportional gain for uphill climbing compensation.
    - k_roll (float): Proportional gain for lateral balance compensation.
    - max_pitch_boost, max_roll_boost: unused — tanh saturates naturally.

    Returns:
    - roll_boost, pitch_boost (np.ndarray shape (2,)): left/right CPG gain corrections.
    - pitch, roll (float): Euler angles in degrees (used by controller for thresholds).
    """
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch, roll, yaw = rot.as_euler('xyz', degrees=True)

    pitch_boost_raw = max(0, pitch) * k_pitch
    pitch_boost = np.tanh(np.array([pitch_boost_raw, pitch_boost_raw]))

    roll_boost_raw = roll * k_roll
    if roll_boost_raw > 0:
        roll_boost = np.tanh(np.array([-roll_boost_raw, roll_boost_raw]))
    else:
        roll_boost = np.tanh(np.array([roll_boost_raw, -roll_boost_raw]))

    return roll_boost, pitch_boost, pitch, roll


def detect_looming(omm, retina_tool, prev_areas,
                   region='top_third', dark=False,
                   pixel_threshold=100, growth_threshold=0.01,
                   proximity_threshold=1.1, gain=1.0):
    """
    Detects looming stimuli by tracking the growth rate of object area in each eye.

    Inspired by looming-sensitive visual projection neurons (VPNs) in Drosophila.
    An approaching object subtends a larger visual angle each frame, so its area
    in the retinal image grows. The growth RATE is used as a proxy for angular
    expansion, which is the key stimulus for two distinct circuits:

    - Obstacle avoidance during walking (level 2): mediated by LPLC1, which
      encodes objects on near-collision trajectories and drives slowing/avoidance.
      (Tanaka & Clark, Curr. Biol. 2022)
    - Predator escape (level 4): mediated by LC4 (angular velocity) and LPLC2
      (angular size via radial motion opponency), converging on the Giant Fiber
      to trigger escape. (Klapoetke et al., Nature 2017; Ache et al., Curr. Biol. 2019)

    Note: tracking pixel-area growth is an engineering approximation of angular
    expansion. The biological computation separates angular size and velocity
    as distinct signals; here we use their product as a single proxy.

    Parameters:
    - omm: ommatidia readouts, shape (2, 721, 2).
    - retina_tool (Retina): converts hex pixels to 2D image.
    - prev_areas (np.ndarray shape (2,)): eye areas from the previous step.
    - region (str): 'top_third' (grass blades appear at top) or 'full' (predator).
    - dark (bool): False = track bright objects (grass), True = track dark (dragonfly).
    - pixel_threshold (int): intensity boundary (0-255) separating object from background.
    - growth_threshold (float): minimum area growth (fraction of eye) to trigger.
    - proximity_threshold (float): absolute area fraction that triggers avoidance even
      when growth ≈ 0 (fly stuck against obstacle, no longer approaching). Default 1.1
      disables proximity check — set to a value in (0,1) to enable.
    - gain (float): steering amplitude.

    Returns:
    - steer (np.ndarray shape (2,)): left/right drive correction.
    - curr_areas (np.ndarray shape (2,)): updated areas for next step.
    - triggered (bool): whether looming was detected this step.
    """
    left_img = retina_tool.hex_pxls_to_human_readable(omm[0].max(-1), color_8bit=True)
    right_img = retina_tool.hex_pxls_to_human_readable(omm[1].max(-1), color_8bit=True)

    if region == 'top_third':
        h = left_img.shape[0] // 3
        left_img = left_img[:h, :]
        right_img = right_img[:h, :]

    if dark:
        left_area = (left_img < pixel_threshold).mean()
        right_area = (right_img < pixel_threshold).mean()
    else:
        left_area = (left_img > pixel_threshold).mean()
        right_area = (right_img > pixel_threshold).mean()

    curr_areas = np.array([left_area, right_area])
    growth = curr_areas - prev_areas

    steer = np.zeros(2)
    triggered = False

    # Trigger on approach (growth) OR proximity (already very close, growth ≈ 0 when stuck)
    if growth.max() > growth_threshold or curr_areas.max() > proximity_threshold:
        triggered = True
        # Magnitude scales with how much of the eye the object occupies — smooth bias, not binary jerk
        if curr_areas[0] >= curr_areas[1]:
            steer = np.array([1.0, -1.0]) * gain * curr_areas.max()
        else:
            steer = np.array([-1.0, 1.0]) * gain * curr_areas.max()

    return steer, curr_areas, triggered


def _crop_hex_to_rect(omm, ommatidia_id_map):
    """Extract forward-facing rectangular band (≈16×31) from hex ommatidia layout.

    Adapted from week4/solutions_vision.ipynb. Returns shape (2, n_rows, n_cols)
    with values in [0, 1] (max over both channels per ommatidium).
    The central rectangular patch corresponds to the fly's forward visual field.
    """
    rows = [np.unique(row) for row in ommatidia_id_map]
    max_width = max(len(row) for row in rows)
    rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
    cols = [np.unique(col) for col in rows.T]
    min_height = min(len(col) for col in cols)
    cols = [col[:min_height] for col in cols]
    rows = np.array(cols).T
    return omm[..., rows, :].max(-1)


def _longest_run_per_col(mask):
    """Length of the longest consecutive True run along the row axis, per column.

    Uses a one-pass cumulative reset: each row's value is (prev + 1) if the cell is
    True, else 0. Final per-column max gives the longest run. Vectorized over batch
    and column dimensions; small Python loop only over rows (H ~ 16).

    Parameters
    ----------
    mask : np.ndarray, bool, shape (..., H, W)

    Returns
    -------
    np.ndarray, int32, shape (..., W)
    """
    H = mask.shape[-2]
    runs = np.zeros(mask.shape, dtype=np.int32)
    runs[..., 0, :] = mask[..., 0, :].astype(np.int32)
    for i in range(1, H):
        prev = runs[..., i - 1, :]
        cur = mask[..., i, :]
        runs[..., i, :] = np.where(cur, prev + 1, 0)
    return runs.max(axis=-2)


def vertical_run_features(rect, dark_threshold=0.15, min_blade_run=6):
    """Per-column blade evidence from the forward visual band.

    A grass blade is a *tall, thin vertical structure*: in the rectified ommatidia
    patch it shows as a column with many *consecutive* dark rows. Ground texture is
    also dark in places, but its dark pixels are scattered (no long runs). The
    longest consecutive vertical run of dark ommatidia per column therefore
    discriminates blades from ground much better than the average dark fraction:

        Open-ground column:  longest run ~ 1-3
        Blade column:        longest run ~ 8-14

    Replacing the dark-fraction feature with this run-length feature is the cheapest
    SNR fix available from the existing ommatidia stream — no new sensor needed.

    Parameters
    ----------
    rect : np.ndarray, shape (2, H, W) — forward band from `_crop_hex_to_rect`
        max-over-channels ommatidia values in [0, 1]. With current geometry H~=16, W~=31.
    dark_threshold : float — ommatidium counted as dark when max-channel < this.
    min_blade_run : int — minimum vertical run length to call a column blade-like.

    Returns
    -------
    run_lengths : np.ndarray, shape (2, W), int32
        Longest consecutive dark run per column per eye.
    col_blade : np.ndarray, shape (2, W), bool
        True where `run_lengths >= min_blade_run`.
    """
    dark = rect < dark_threshold
    run_lengths = _longest_run_per_col(dark)
    col_blade = run_lengths >= min_blade_run
    return run_lengths, col_blade


def _red_ratio(img):
    """Fraction of red-saturated pixels in an RGB image (uint8 or float)."""
    arr = np.asarray(img)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    return float(((r > 150) & (r > 2.0 * g) & (r > 2.0 * b)).mean())


def detect_dragonfly_red(raw_vision, threshold_fraction=1e-3):
    """Detect a red object (dragonfly) in either fly eye via raw RGB.

    Adopted from Tom's `origin/tom` branch: in this simulation, the dragonfly
    is rendered with high red-channel intensity. A pixel is "red" when
    R > 150 AND R > 2·G AND R > 2·B. Fraction of red pixels per eye gives a
    cheap directional detector — much cleaner than dark-pixel looming, which
    cannot distinguish a red dragonfly from any other dark object.

    Parameters
    ----------
    raw_vision : list[ndarray] or array-like
        Output of ``sim.get_raw_vision(fly_name)``: per-eye RGB images.
    threshold_fraction : float
        Per-eye red-pixel fraction above which a dragonfly is declared.

    Returns
    -------
    tuple
        (detected: bool, side: str). ``side`` is one of
        ``"front"`` (both eyes red), ``"left"`` (more red in left eye),
        ``"right"`` (more red in right eye), or ``"none"`` when not detected.
    """
    arr = np.asarray(raw_vision)
    L = _red_ratio(arr[0])
    R = _red_ratio(arr[1])
    if L < threshold_fraction and R < threshold_fraction:
        return False, "none"
    if L > threshold_fraction and R > threshold_fraction:
        return True, "front"
    if L >= R:
        return True, "left"
    return True, "right"


def raw_vision_features(raw_vision, top_band=0.5,
                         green_count_threshold=80,
                         green_diff=15, green_min=60):
    """Detect grass blades in raw RGB images via green pixels above the horizon.

    Grass blades are the only saturated-green objects that *rise above the
    horizon*. The ground texture is also greenish/brownish but stays in the
    bottom half of the eye image. Counting green pixels in the top half of
    each column gives a clean per-column blade indicator that completely
    sidesteps the yellow/pale ommatidia channel pattern that breaks any
    feature built on the hex retina.

    Color rule: a pixel is "green" when G > R + green_diff AND G > B +
    green_diff AND G > green_min. Sky (high B), ground (R and G similar,
    low B) and shadow all fail this rule — only the saturated chlorophyll
    green of the blades passes.

    Parameters
    ----------
    raw_vision : np.ndarray, shape (2, H, W, 3) uint8
        Output of ``sim.get_raw_vision(fly_name)``, one image per eye.
    top_band : float in (0, 1]
        Fraction of image height (from top) treated as "above horizon".
        0.5 = top half. Ground stays out of this band on flat or gentle hills.
    green_count_threshold : int
        Per-column green-pixel count above which the column is blade-like.
        Tune from data — open-terrain median is ~30-50, blade columns >80.
    green_diff, green_min : int
        Tolerances of the green-pixel rule.

    Returns
    -------
    col_blade : np.ndarray, shape (2, W) bool
        True where the column carries blade evidence (above threshold).
    top_count : np.ndarray, shape (2, W) int
        Green-pixel count per column in the top band (the raw signal).
    """
    arr = np.asarray(raw_vision)
    H = arr.shape[1]
    R = arr[..., 0].astype(np.int16)
    G = arr[..., 1].astype(np.int16)
    B = arr[..., 2].astype(np.int16)
    green = (G > R + green_diff) & (G > B + green_diff) & (G > green_min)
    band = max(1, int(H * top_band))
    top_count = green[:, :band, :].sum(axis=1)
    col_blade = top_count > green_count_threshold
    return col_blade, top_count


def detect_blade_edge(omm, retina_tool, prev_areas,
                      growth_threshold=0.01,
                      proximity_threshold=0.25,
                      pixel_threshold=0.15):
    """
    Detects grass blades via dark ommatidia in the forward visual band.

    Works on raw ommatidia values (normalized [0, 1]). An ommatidium is 'dark'
    when max(Ch0, Ch1) < pixel_threshold. This correctly identifies pale-type
    ommatidia looking at green objects:
      - Pale-type on green blade/ground: Ch0=0, Ch1=B=0 → max=0 → dark
      - Pale-type on blue sky: Ch0=0, Ch1=B≈0.78 → max=0.78 → bright
      - Yellow-type anywhere: Ch0=G≈0.78, Ch1=0 → max≈0.78 → bright

    The forward rectangular band (~16×31) is extracted via _crop_hex_to_rect,
    covering roughly the horizontal visual field ahead of the fly:
      - Sky rows (above horizon): ~0% dark baseline
      - Ground rows (below horizon): ~30% dark (pale-type fraction)
      - Open terrain overall: ~15% dark baseline

    A grass blade entering the forward field pushes the dark fraction above
    baseline. Growth detection catches approach; proximity catches the stuck case.

    Inspired by LPLC1-mediated walking collision avoidance (Tanaka & Clark,
    Curr. Biol. 2022): fly slows down rather than turning abruptly, letting the
    odor gradient redirect it naturally. A small edge-asymmetry bias nudges
    toward whichever eye sees fewer dark pixels (clearer path).

    Parameters
    ----------
    omm : np.ndarray, shape (2, 721, 2) — raw ommatidia readouts, normalized [0, 1]
    retina_tool : Retina — provides ommatidia_id_map for forward band extraction
    prev_areas : np.ndarray, shape (2,) — dark fractions from previous step
    growth_threshold : float — min growth per step to trigger (default 0.01)
    proximity_threshold : float — absolute dark fraction to trigger (default 0.25)
    pixel_threshold : float — ommatidium max value counted as dark (default 0.15)

    Returns
    -------
    slow_factor : float in [0.3, 1.0]
    steer_bias : np.ndarray shape (2,)
    curr_areas : np.ndarray shape (2,)
    triggered : bool
    """
    rect = _crop_hex_to_rect(omm, retina_tool.ommatidia_id_map)  # (2, n_rows, n_cols)

    left_dark = (rect[0] < pixel_threshold).mean()
    right_dark = (rect[1] < pixel_threshold).mean()
    curr_areas = np.array([left_dark, right_dark])
    growth = curr_areas - prev_areas

    triggered = growth.max() > growth_threshold or curr_areas.max() > proximity_threshold

    slow_factor = 1.0
    steer_bias = np.zeros(2)

    if triggered:
        # Scale slowing by how much dark fraction exceeds the ~15% open-terrain baseline
        excess = max(0.0, curr_areas.max() - 0.15)
        slow_factor = max(0.3, 1.0 - excess * 5)

        left_edge = np.abs(np.diff(rect[0], axis=1)).mean()
        right_edge = np.abs(np.diff(rect[1], axis=1)).mean()
        total = left_edge + right_edge + 1e-6
        edge_asym = (left_edge - right_edge) / total
        steer_bias = np.array([edge_asym, -edge_asym]) * 0.15

    return slow_factor, steer_bias, curr_areas, triggered
