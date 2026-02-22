"""
Shared test fixtures for the ASL QC toolbox.

Generates synthetic NIfTI-like numpy arrays that simulate realistic
ASL data, CBF maps, tissue masks, and motion parameters.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Spatial dimensions for all synthetic volumes
NX, NY, NZ = 32, 32, 20
N_VOLS = 40  # 20 control + 20 label


# ---------------------------------------------------------------------------
# Brain mask
# ---------------------------------------------------------------------------

@pytest.fixture
def brain_mask() -> np.ndarray:
    """Ellipsoidal brain mask."""
    mask = np.zeros((NX, NY, NZ), dtype=bool)
    cx, cy, cz = NX // 2, NY // 2, NZ // 2
    rx, ry, rz = NX // 3, NY // 3, NZ // 3
    for x in range(NX):
        for y in range(NY):
            for z in range(NZ):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1:
                    mask[x, y, z] = True
    return mask


# ---------------------------------------------------------------------------
# Tissue masks (GM, WM subsets of brain)
# ---------------------------------------------------------------------------

@pytest.fixture
def gm_mask(brain_mask: np.ndarray) -> np.ndarray:
    """GM mask: outer shell of the brain mask."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(brain_mask, iterations=2)
    gm = brain_mask & ~eroded
    return gm


@pytest.fixture
def wm_mask(brain_mask: np.ndarray) -> np.ndarray:
    """WM mask: inner core of the brain mask."""
    from scipy.ndimage import binary_erosion
    wm = binary_erosion(brain_mask, iterations=2)
    return wm


# ---------------------------------------------------------------------------
# CBF maps
# ---------------------------------------------------------------------------

@pytest.fixture
def good_cbf_map(gm_mask: np.ndarray, wm_mask: np.ndarray) -> np.ndarray:
    """High-quality CBF map: GM ~ 60, WM ~ 25, no negatives."""
    rng = np.random.default_rng(42)
    cbf = np.zeros((NX, NY, NZ), dtype=np.float64)
    cbf[gm_mask] = 60.0 + rng.normal(0, 5, size=int(np.sum(gm_mask)))
    cbf[wm_mask] = 25.0 + rng.normal(0, 3, size=int(np.sum(wm_mask)))
    return cbf


@pytest.fixture
def noisy_cbf_map(brain_mask: np.ndarray) -> np.ndarray:
    """Noisy CBF map: lots of negative values and high variance."""
    rng = np.random.default_rng(123)
    cbf = rng.normal(10, 50, size=(NX, NY, NZ))
    cbf[~brain_mask] = 0.0
    return cbf


# ---------------------------------------------------------------------------
# 4-D ASL timeseries
# ---------------------------------------------------------------------------

@pytest.fixture
def asl_timeseries(brain_mask: np.ndarray) -> np.ndarray:
    """Synthetic 4-D ASL timeseries with alternating control/label."""
    rng = np.random.default_rng(42)
    data = np.zeros((NX, NY, NZ, N_VOLS), dtype=np.float64)
    for t in range(N_VOLS):
        vol = rng.normal(1000, 50, size=(NX, NY, NZ))
        if t % 2 == 0:
            # Control: higher signal
            vol[brain_mask] += 10
        else:
            # Label: lower signal (inverted)
            vol[brain_mask] -= 10
        data[..., t] = vol
    return data


@pytest.fixture
def asl_context_types() -> list:
    """Matching context for ``asl_timeseries``."""
    return ["control" if t % 2 == 0 else "label" for t in range(N_VOLS)]


# ---------------------------------------------------------------------------
# Motion parameters
# ---------------------------------------------------------------------------

@pytest.fixture
def low_motion_params() -> np.ndarray:
    """Low-motion parameters (N×6): small translations/rotations."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.05, size=(N_VOLS, 6))


@pytest.fixture
def high_motion_params() -> np.ndarray:
    """High-motion parameters with a large spike."""
    rng = np.random.default_rng(42)
    params = rng.normal(0, 0.05, size=(N_VOLS, 6))
    params[15, :3] += 5.0  # 5mm translation spike
    return params


# ---------------------------------------------------------------------------
# Temporary directory for file I/O tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# BIDS fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aslcontext_tsv(tmp_dir: Path) -> Path:
    """Write a valid aslcontext.tsv file."""
    p = tmp_dir / "aslcontext.tsv"
    lines = ["volume_type\n"]
    for t in range(N_VOLS):
        lines.append("control\n" if t % 2 == 0 else "label\n")
    p.write_text("".join(lines))
    return p


@pytest.fixture
def asl_json(tmp_dir: Path) -> Path:
    """Write a valid ASL JSON sidecar."""
    import json
    p = tmp_dir / "sub-01_asl.json"
    data = {
        "ArterialSpinLabelingType": "PCASL",
        "PostLabelingDelay": 1.8,
        "BackgroundSuppression": False,
        "RepetitionTimePreparation": 4.0,
        "LabelingDuration": 1.8,
        "M0Type": "Separate",
        "MagneticFieldStrength": 3.0,
        "Manufacturer": "Siemens",
    }
    p.write_text(json.dumps(data, indent=2))
    return p
