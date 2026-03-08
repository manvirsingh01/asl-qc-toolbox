"""
Synthetic brain phantom generator for demos and testing.

Creates a realistic 3-D brain volume (121x145x121 default, matching MNI
1.5 mm space) with anatomical features: cortical grey matter with sulcal
folding patterns that indent the brain surface, deep white matter, lateral
ventricles, interhemispheric fissure, and probabilistic tissue masks.

Designed to produce high-quality QC images that resemble real
ExploreASL outputs at publication quality (300 DPI).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter


@dataclass
class BrainPhantom:
    """Container for synthetic brain arrays."""

    shape: tuple
    brain_mask: np.ndarray       # bool
    gm_mask: np.ndarray          # float  (probabilistic 0-1)
    wm_mask: np.ndarray          # float  (probabilistic 0-1)
    csf_mask: np.ndarray         # float  (probabilistic 0-1)
    ventricles: np.ndarray       # bool
    t1w: np.ndarray              # float  synthetic T1-weighted image


def _sulcal_displacement(shape, rng, n_harmonics=8, base_freq=2.0):
    """Multi-scale noise that modulates the brain surface to create sulci.

    Returns a 3-D field in [0, 1] where high values will push the brain
    surface inward, creating sulcal grooves.
    """
    field = np.zeros(shape, dtype=np.float64)
    for k in range(n_harmonics):
        freq = base_freq * (1.5 ** k)
        amp = 1.0 / (1.0 + 0.5 * k)
        raw = rng.normal(0, 1, shape)
        sigma = max(2.0, shape[0] / (2.0 * freq))
        smoothed = gaussian_filter(raw, sigma=sigma)
        field += amp * smoothed
    # Normalise to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    return field


def generate_brain_phantom(
    shape: tuple = (121, 145, 121),
    rng: np.random.Generator | None = None,
) -> BrainPhantom:
    """Generate a synthetic brain volume with realistic tissue masks.

    Parameters
    ----------
    shape : tuple
        (X, Y, Z) voxel dimensions.  Default (121, 145, 121) matches
        MNI 1.5 mm space for realistic aspect ratios.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    BrainPhantom
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nx, ny, nz = shape

    # Coordinate grids normalised to [-1, 1]
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # ── Outer brain boundary ─────────────────────────────────────
    # Use a realistic brain-shaped envelope (not a pure ellipsoid):
    # - Wider L-R (X), elongated A-P (Y), compact S-I (Z)
    # - Frontal flattening, occipital rounding, temporal bulge
    # - Slight downward tilt of frontal lobe
    r_x = 0.82 + 0.06 * np.tanh(3.0 * Y)                # narrower posteriorly
    r_y = 0.80 + 0.04 * Z                                # elongated anteriorly at top
    r_z = 0.72 - 0.08 * Y.clip(0)                        # flatter top posteriorly

    brain_dist = (X / r_x) ** 2 + (Y / r_y) ** 2 + (Z / r_z) ** 2

    # Carve sulci into the brain surface using multi-scale displacement
    sulcal_field = _sulcal_displacement(shape, rng, n_harmonics=8, base_freq=2.5)
    # Modulate the boundary threshold: sulci push the surface inward
    sulcal_depth = 0.14  # how deep sulci can go
    effective_threshold = 1.0 - sulcal_depth * sulcal_field

    brain_base = brain_dist <= effective_threshold

    # ── Interhemispheric fissure (thin gap, only 1 voxel at true midline) ─
    # Use a very narrow fissure that gets wider near the top (dorsal)
    fissure_width = 0.012 + 0.015 * Z.clip(0)  # wider dorsally
    fissure = np.abs(X) < fissure_width
    brain_mask = brain_base & ~fissure

    # ── Lateral ventricles (horn-shaped cavities) ─────────────────
    vent_left = (
        ((X + 0.11) / 0.06) ** 2
        + ((Y - 0.02) / 0.22) ** 2
        + ((Z + 0.04) / 0.14) ** 2
    ) <= 1.0
    vent_right = (
        ((X - 0.11) / 0.06) ** 2
        + ((Y - 0.02) / 0.22) ** 2
        + ((Z + 0.04) / 0.14) ** 2
    ) <= 1.0
    ventricles = (vent_left | vent_right) & brain_mask
    brain_no_vent = brain_mask & ~ventricles

    # ── Probabilistic tissue masks ───────────────────────────────
    dist_from_surface = distance_transform_edt(brain_no_vent).astype(np.float64)

    # GM sits in the outer "cortical ribbon" of the brain
    # Sulcal field modulates cortical thickness (thinner in sulci)
    cortical_depth = 5.0 + 2.5 * sulcal_field
    gm_prob = np.clip(1.0 - (dist_from_surface - 0.8) / cortical_depth, 0, 1)
    gm_prob[~brain_no_vent] = 0.0

    # WM: high deep inside, steep fall-off toward cortex
    wm_prob = np.clip((dist_from_surface - 4.0) / 5.0, 0, 1)
    wm_prob[~brain_no_vent] = 0.0
    wm_prob[ventricles] = 0.0

    # Smooth for realistic partial volume boundaries
    gm_prob = gaussian_filter(gm_prob, sigma=1.2)
    wm_prob = gaussian_filter(wm_prob, sigma=1.2)

    # Normalise so GM + WM <= 1 inside brain parenchyma
    total = gm_prob + wm_prob + 1e-12
    gm_prob = np.where(brain_no_vent, gm_prob / total, 0.0)
    wm_prob = np.where(brain_no_vent, wm_prob / total, 0.0)

    # CSF
    csf_prob = np.zeros(shape, dtype=np.float64)
    csf_prob[ventricles] = 1.0
    # Sulcal CSF: surface voxels not covered by brain parenchyma
    surface_ring = brain_mask & ~brain_no_vent
    csf_prob[surface_ring] = 0.3

    # ── Synthetic T1-weighted image ──────────────────────────────
    # T1 signal intensities: WM bright (~200), GM mid (~120), CSF dark (~30)
    t1w = np.zeros(shape, dtype=np.float64)
    t1w += wm_prob * 200.0
    t1w += gm_prob * 120.0
    t1w += csf_prob * 30.0
    # Add realistic Rician noise
    noise_level = 6.0
    t1w += rng.normal(0, noise_level, shape)
    t1w = gaussian_filter(t1w, sigma=0.6)
    t1w[~brain_mask] = rng.normal(5, 3, shape)[~brain_mask].clip(0)
    t1w = np.clip(t1w, 0, 255)

    return BrainPhantom(
        shape=shape,
        brain_mask=brain_mask,
        gm_mask=gm_prob,
        wm_mask=wm_prob,
        csf_mask=csf_prob,
        ventricles=ventricles,
        t1w=t1w,
    )


def generate_cbf_map(
    phantom: BrainPhantom,
    gm_cbf: float = 60.0,
    wm_cbf: float = 25.0,
    gm_noise: float = 5.0,
    wm_noise: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic CBF map from the phantom tissue masks.

    Parameters
    ----------
    phantom : BrainPhantom
        Generated brain phantom.
    gm_cbf, wm_cbf : float
        Mean CBF for GM and WM tissue.
    gm_noise, wm_noise : float
        Standard deviation of CBF noise per tissue type.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray
        3-D CBF map (ml/100g/min).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cbf = np.zeros(phantom.shape, dtype=np.float64)

    # Assign CBF using binary mask thresholds to avoid partial-volume
    # variance inflation, then smooth for realistic appearance
    gm_binary = phantom.gm_mask > 0.5
    wm_binary = phantom.wm_mask > 0.5

    noise = rng.normal(0, 1, phantom.shape)
    cbf[gm_binary] = gm_cbf + gm_noise * noise[gm_binary]
    cbf[wm_binary] = wm_cbf + wm_noise * noise[wm_binary]

    # Light smoothing for realistic perfusion appearance
    cbf = gaussian_filter(cbf, sigma=0.4)

    # Zero outside brain
    cbf[~phantom.brain_mask] = 0.0
    cbf[phantom.ventricles] = 0.0

    return cbf


def generate_temporal_sd(
    phantom: BrainPhantom,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic temporal standard deviation map.

    Per ExploreASL QC guidelines, should be smooth and uniform
    without GM/WM contrast for high-quality data.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sd = np.zeros(phantom.shape, dtype=np.float64)
    # Uniform noise inside brain — no structure = good quality
    sd[phantom.brain_mask] = 15.0 + rng.normal(0, 3, phantom.shape)[phantom.brain_mask]
    sd = gaussian_filter(sd, sigma=2.0)
    sd[~phantom.brain_mask] = 0.0
    return sd


def generate_temporal_sd_bad(
    phantom: BrainPhantom,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a BAD temporal SD map showing structured noise.

    Has GM/WM contrast (indicates motion/noise contamination).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sd = np.zeros(phantom.shape, dtype=np.float64)
    # Structured: GM higher than WM = bad (indicates residual motion)
    sd += phantom.gm_mask * (25.0 + rng.normal(0, 8, phantom.shape))
    sd += phantom.wm_mask * (10.0 + rng.normal(0, 4, phantom.shape))
    sd = gaussian_filter(sd, sigma=1.0)
    sd[~phantom.brain_mask] = 0.0
    return sd


def generate_motion_params(
    n_volumes: int = 80,
    max_displacement: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic motion parameters (good — low motion).

    Returns (n_volumes, 6) array: 3 translations (mm) + 3 rotations (deg).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    params = np.zeros((n_volumes, 6))
    for col in range(6):
        # Random walk with drift
        steps = rng.normal(0, max_displacement * 0.05, n_volumes)
        params[:, col] = np.cumsum(steps)
    return params


def generate_motion_params_bad(
    n_volumes: int = 80,
    max_displacement: float = 2.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate BAD motion parameters (high motion with sudden spikes)."""
    if rng is None:
        rng = np.random.default_rng(42)

    params = np.zeros((n_volumes, 6))
    for col in range(6):
        steps = rng.normal(0, max_displacement * 0.1, n_volumes)
        params[:, col] = np.cumsum(steps)
        # Add random spikes
        n_spikes = rng.integers(3, 8)
        spike_idx = rng.choice(n_volumes, n_spikes, replace=False)
        params[spike_idx, col] += rng.normal(0, max_displacement, n_spikes)
    return params


def generate_m0_image(
    phantom: BrainPhantom,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic M0 calibration image.

    Should show smooth, uniform signal inside brain.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    m0 = np.zeros(phantom.shape, dtype=np.float64)
    # Relatively uniform signal with slight bias field
    xs = np.linspace(-1, 1, phantom.shape[0])
    ys = np.linspace(-1, 1, phantom.shape[1])
    zs = np.linspace(-1, 1, phantom.shape[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    bias_field = 1.0 + 0.1 * (X ** 2 + Y ** 2 + Z ** 2)

    m0[phantom.brain_mask] = 1000.0 * bias_field[phantom.brain_mask]
    m0 += rng.normal(0, 20, phantom.shape)
    m0 = gaussian_filter(m0, sigma=1.5)
    m0[~phantom.brain_mask] = rng.normal(50, 15, phantom.shape)[~phantom.brain_mask].clip(0)
    return np.clip(m0, 0, None)
