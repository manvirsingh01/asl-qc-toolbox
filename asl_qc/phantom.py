"""
Anatomically realistic synthetic brain phantom generator.

Creates a 3-D brain volume (121x145x121 default, MNI 1.5 mm space) with:
- Irregular brain boundary with frontal/occipital/temporal lobe shapes
- Deep sulcal grooves carved into the brain surface (cortical folding)
- Lateral ventricles with correct C-shaped morphology
- Subcortical grey matter: caudate, putamen, thalamus
- Corpus callosum as midline WM bridge
- Cerebellum as posterior inferior structure
- Brainstem (pons/medulla) extending inferiorly
- Probabilistic tissue masks with smooth PV transitions
- Synthetic T1-weighted image with realistic tissue contrast

Designed to produce publication-quality QC images at 300 DPI.
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
    subcortical_gm: np.ndarray   # bool  (caudate + putamen + thalamus)


# ── Anatomical parameters (normalised coordinates [-1, 1]) ──────
# Brain envelope radii
_RX_BASE = 0.80   # left-right
_RY_BASE = 0.82   # anterior-posterior
_RZ_BASE = 0.70   # superior-inferior

# Ventricle parameters
_VENT_X_OFF = 0.10    # lateral offset from midline
_VENT_RX = 0.045      # L-R extent
_VENT_RY = 0.20       # A-P extent
_VENT_RZ = 0.12       # S-I extent

# Subcortical nuclei sizes (in normalised coords)
_CAUDATE_RX, _CAUDATE_RY, _CAUDATE_RZ = 0.035, 0.10, 0.09
_PUTAMEN_RX, _PUTAMEN_RY, _PUTAMEN_RZ = 0.05, 0.08, 0.08
_THALAMUS_RX, _THALAMUS_RY, _THALAMUS_RZ = 0.06, 0.06, 0.06

# Cortical ribbon depth (voxels, modulated by sulcal field)
_CORTEX_MIN_DEPTH = 3.5
_CORTEX_DEPTH_RANGE = 2.0

# WM core distance threshold (voxels from surface)
_WM_ONSET = 4.0
_WM_RISE = 5.0


def _sulcal_field(shape, rng, n_harmonics=10, base_freq=2.5):
    """Multi-scale displacement field for cortical folding.

    Returns a 3-D field in [0, 1] — high values indent the brain surface
    to create sulcal grooves. Uses more harmonics and higher frequency
    than before for finer, more realistic folding patterns.
    """
    field = np.zeros(shape, dtype=np.float64)
    for k in range(n_harmonics):
        freq = base_freq * (1.4 ** k)
        amp = 1.0 / (1.0 + 0.4 * k)
        raw = rng.normal(0, 1, shape)
        sigma = max(1.5, shape[0] / (2.0 * freq))
        field += amp * gaussian_filter(raw, sigma=sigma)
    mn, mx = field.min(), field.max()
    return (field - mn) / (mx - mn + 1e-12)


def _ellipsoid(X, Y, Z, cx, cy, cz, rx, ry, rz):
    """Return boolean mask for an ellipsoid centred at (cx, cy, cz)."""
    return (((X - cx) / rx) ** 2 +
            ((Y - cy) / ry) ** 2 +
            ((Z - cz) / rz) ** 2) <= 1.0


def generate_brain_phantom(
    shape: tuple = (121, 145, 121),
    rng: np.random.Generator | None = None,
) -> BrainPhantom:
    """Generate a synthetic brain volume with anatomically realistic features.

    Parameters
    ----------
    shape : tuple
        (X, Y, Z) voxel dimensions.  Default (121, 145, 121) matches
        MNI 1.5 mm space.
    rng : numpy Generator, optional

    Returns
    -------
    BrainPhantom
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nx, ny, nz = shape
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # ── 1. Cerebrum outer boundary ───────────────────────────────
    # Asymmetric radii: frontal broader, occipital narrower,
    # temporal lobe bulge, slight inferior tilt anteriorly.
    r_x = _RX_BASE + 0.05 * np.tanh(2.5 * Y) + 0.03 * (1 - Z.clip(0))
    r_y = _RY_BASE + 0.03 * Z                   # longer A-P at top
    r_z = _RZ_BASE - 0.06 * Y.clip(0)           # flatter posteriorly

    cerebrum_dist = (X / r_x) ** 2 + (Y / r_y) ** 2 + (Z / r_z) ** 2

    # Temporal lobe bulge: widen brain laterally in inferior-anterior
    temporal_bulge = np.exp(-((Z + 0.25) ** 2 / 0.06 + (Y - 0.15) ** 2 / 0.08))
    cerebrum_dist -= 0.12 * temporal_bulge

    # Sulcal indentations
    sulcal = _sulcal_field(shape, rng, n_harmonics=10, base_freq=2.5)
    sulcal_depth = 0.18
    threshold = 1.0 - sulcal_depth * sulcal
    cerebrum = cerebrum_dist <= threshold

    # ── 2. Cerebellum (posterior-inferior) ────────────────────────
    cerebellum = _ellipsoid(X, Y, Z, 0.0, -0.52, -0.55, 0.42, 0.22, 0.18)
    # Cerebellar fissure (vermis gap)
    cb_fissure = np.abs(X) < 0.025
    cerebellum = cerebellum & ~cb_fissure

    # ── 3. Brainstem (inferior midline) ──────────────────────────
    brainstem = _ellipsoid(X, Y, Z, 0.0, -0.20, -0.58, 0.08, 0.10, 0.16)

    # ── 4. Combine into full brain mask ──────────────────────────
    brain_raw = cerebrum | cerebellum | brainstem

    # Interhemispheric fissure — only in cerebrum, widens dorsally
    fissure_width = 0.010 + 0.012 * Z.clip(0)
    fissure = (np.abs(X) < fissure_width) & (Z > -0.30)  # not in brainstem
    brain_mask = brain_raw & ~fissure

    # ── 5. Lateral ventricles (C-shaped) ─────────────────────────
    # Body of lateral ventricle
    vent_body_L = _ellipsoid(X, Y, Z, -_VENT_X_OFF, -0.02, 0.0,
                             _VENT_RX, _VENT_RY, _VENT_RZ)
    vent_body_R = _ellipsoid(X, Y, Z, _VENT_X_OFF, -0.02, 0.0,
                             _VENT_RX, _VENT_RY, _VENT_RZ)
    # Frontal horn (anterior extension, tilted down)
    vent_front_L = _ellipsoid(X, Y, Z, -0.06, 0.18, -0.04,
                              0.03, 0.08, 0.05)
    vent_front_R = _ellipsoid(X, Y, Z, 0.06, 0.18, -0.04,
                              0.03, 0.08, 0.05)
    # Occipital horn (posterior extension)
    vent_occ_L = _ellipsoid(X, Y, Z, -0.07, -0.22, -0.03,
                            0.025, 0.06, 0.035)
    vent_occ_R = _ellipsoid(X, Y, Z, 0.07, -0.22, -0.03,
                            0.025, 0.06, 0.035)
    # Temporal horn (inferior-lateral extension)
    vent_temp_L = _ellipsoid(X, Y, Z, -0.16, -0.05, -0.18,
                             0.03, 0.10, 0.03)
    vent_temp_R = _ellipsoid(X, Y, Z, 0.16, -0.05, -0.18,
                             0.03, 0.10, 0.03)

    ventricles = ((vent_body_L | vent_body_R |
                   vent_front_L | vent_front_R |
                   vent_occ_L | vent_occ_R |
                   vent_temp_L | vent_temp_R) & brain_mask)

    # Third ventricle (midline slit)
    third_vent = _ellipsoid(X, Y, Z, 0.0, -0.04, -0.02,
                            0.012, 0.06, 0.05)
    ventricles = ventricles | (third_vent & brain_mask)

    brain_no_vent = brain_mask & ~ventricles

    # ── 6. Subcortical grey matter structures ────────────────────
    # Caudate nucleus (medial to ventricle, C-shaped head)
    caudate_L = _ellipsoid(X, Y, Z, -0.06, 0.06, 0.02,
                           _CAUDATE_RX, _CAUDATE_RY, _CAUDATE_RZ)
    caudate_R = _ellipsoid(X, Y, Z, 0.06, 0.06, 0.02,
                           _CAUDATE_RX, _CAUDATE_RY, _CAUDATE_RZ)
    # Putamen (lateral to caudate)
    putamen_L = _ellipsoid(X, Y, Z, -0.18, 0.02, -0.02,
                           _PUTAMEN_RX, _PUTAMEN_RY, _PUTAMEN_RZ)
    putamen_R = _ellipsoid(X, Y, Z, 0.18, 0.02, -0.02,
                           _PUTAMEN_RX, _PUTAMEN_RY, _PUTAMEN_RZ)
    # Thalamus (posterior and medial)
    thal_L = _ellipsoid(X, Y, Z, -0.08, -0.10, -0.04,
                        _THALAMUS_RX, _THALAMUS_RY, _THALAMUS_RZ)
    thal_R = _ellipsoid(X, Y, Z, 0.08, -0.10, -0.04,
                        _THALAMUS_RX, _THALAMUS_RY, _THALAMUS_RZ)

    subcortical_gm = ((caudate_L | caudate_R |
                       putamen_L | putamen_R |
                       thal_L | thal_R) & brain_no_vent)

    # ── 7. Corpus callosum (midline WM bridge) ──────────────────
    # Curved band connecting hemispheres (visible in sagittal)
    cc_body = _ellipsoid(X, Y, Z, 0.0, -0.02, 0.12,
                         0.04, 0.28, 0.03)
    # Genu (anterior thickening)
    cc_genu = _ellipsoid(X, Y, Z, 0.0, 0.22, 0.06,
                         0.03, 0.04, 0.06)
    # Splenium (posterior thickening)
    cc_splenium = _ellipsoid(X, Y, Z, 0.0, -0.28, 0.06,
                             0.03, 0.04, 0.06)
    corpus_callosum = (cc_body | cc_genu | cc_splenium) & brain_no_vent

    # ── 8. Probabilistic tissue masks ────────────────────────────
    dist_from_surface = distance_transform_edt(brain_no_vent).astype(np.float64)

    # Cortical GM ribbon — modulated by sulcal field
    cortical_depth = _CORTEX_MIN_DEPTH + _CORTEX_DEPTH_RANGE * sulcal
    gm_prob = np.clip(1.0 - (dist_from_surface - 0.5) / cortical_depth, 0, 1)
    gm_prob[~brain_no_vent] = 0.0
    # Add subcortical GM as bright islands
    gm_prob[subcortical_gm] = np.maximum(gm_prob[subcortical_gm], 0.85)

    # WM core
    wm_prob = np.clip((dist_from_surface - _WM_ONSET) / _WM_RISE, 0, 1)
    wm_prob[~brain_no_vent] = 0.0
    wm_prob[ventricles] = 0.0
    # Corpus callosum is definite WM
    wm_prob[corpus_callosum] = np.maximum(wm_prob[corpus_callosum], 0.90)
    # Subcortical GM overrides WM
    wm_prob[subcortical_gm] = np.minimum(wm_prob[subcortical_gm], 0.10)

    # Cerebellar GM (outer cerebellar cortex)
    cb_dist = distance_transform_edt(cerebellum & brain_no_vent).astype(np.float64)
    cb_gm = np.clip(1.0 - (cb_dist - 0.5) / 3.0, 0, 1)
    cb_gm[~(cerebellum & brain_no_vent)] = 0.0
    # Merge cerebellar GM with cortical GM
    in_cb = cerebellum & brain_no_vent
    gm_prob[in_cb] = np.maximum(gm_prob[in_cb], cb_gm[in_cb])

    # Smooth for PV boundaries
    gm_prob = gaussian_filter(gm_prob, sigma=1.0)
    wm_prob = gaussian_filter(wm_prob, sigma=1.0)

    # Normalise so GM + WM <= 1 within parenchyma
    total = gm_prob + wm_prob + 1e-12
    gm_prob = np.where(brain_no_vent, gm_prob / total, 0.0)
    wm_prob = np.where(brain_no_vent, wm_prob / total, 0.0)

    # CSF
    csf_prob = np.zeros(shape, dtype=np.float64)
    csf_prob[ventricles] = 1.0
    # Sulcal CSF in surface gaps
    surface_gap = brain_mask & ~brain_no_vent
    csf_prob[surface_gap] = 0.3

    # ── 9. Synthetic T1-weighted image ───────────────────────────
    t1w = np.zeros(shape, dtype=np.float64)
    t1w += wm_prob * 210.0        # WM bright
    t1w += gm_prob * 130.0        # GM mid-grey
    t1w += csf_prob * 25.0        # CSF dark
    # Subcortical GM slightly different intensity than cortex
    t1w[subcortical_gm] = 115.0 + rng.normal(0, 3, shape)[subcortical_gm]
    # Corpus callosum very bright WM
    t1w[corpus_callosum] = 220.0 + rng.normal(0, 3, shape)[corpus_callosum]
    # Brainstem moderate intensity
    bs_mask = brainstem & brain_no_vent & ~subcortical_gm
    t1w[bs_mask] = 150.0 + rng.normal(0, 5, shape)[bs_mask]

    # Rician noise + smoothing
    t1w += rng.normal(0, 5.0, shape)
    t1w = gaussian_filter(t1w, sigma=0.5)
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
        subcortical_gm=subcortical_gm,
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
