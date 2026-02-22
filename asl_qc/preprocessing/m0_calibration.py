"""
M0 calibration image assessment.

Evaluates the quality and usability of the M0 (equilibrium magnetisation)
image used for absolute CBF quantification, including intensity range
checks, artifact flagging, and automated preconditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class M0Result:
    """Result of M0 calibration assessment."""

    source: str
    """How the M0 was obtained: ``'dedicated'``, ``'control_average'``, or ``'absent'``."""
    is_usable: bool
    """``True`` if the M0 is suitable for absolute quantification."""
    intensity_ratio: float
    """max / median intensity ratio within the brain mask."""
    artifact_flagged: bool
    """``True`` if the intensity ratio exceeds the threshold."""
    smoothed_m0: Optional[np.ndarray]
    """Preconditioned (smoothed + low-signal masked) M0 image, or ``None``."""
    low_signal_mask: Optional[np.ndarray]
    """Boolean mask of voxels below the low-signal threshold."""
    message: str = ""


def assess_m0(
    m0_data: Optional[np.ndarray],
    brain_mask: np.ndarray,
    *,
    has_dedicated_m0: bool = True,
    background_suppression: bool = False,
    control_average: Optional[np.ndarray] = None,
    max_median_ratio_threshold: float = 10.0,
    smoothing_fwhm_mm: float = 5.0,
    voxel_size_mm: float = 2.0,
    low_signal_fraction: float = 0.10,
) -> M0Result:
    """Assess the M0 calibration image for ASL quantification.

    Parameters
    ----------
    m0_data : np.ndarray or None
        Dedicated M0 scan (3-D). ``None`` if not available.
    brain_mask : np.ndarray
        Boolean 3-D brain mask.
    has_dedicated_m0 : bool
        Whether a dedicated M0 scan was acquired.
    background_suppression : bool
        Whether background suppression was active during the ASL scan.
    control_average : np.ndarray or None
        Temporal mean of control images (used as M0 proxy if no dedicated
        scan and no background suppression).
    max_median_ratio_threshold : float
        Flag artifacts if max/median within the mask exceeds this.
    smoothing_fwhm_mm : float
        FWHM of Gaussian smoothing applied to the M0.
    voxel_size_mm : float
        Isotropic voxel size for sigma calculation.
    low_signal_fraction : float
        Mask voxels below this fraction of global mean M0.

    Returns
    -------
    M0Result
    """
    mask = brain_mask.astype(bool)

    # ---- Source identification ----
    if m0_data is not None and has_dedicated_m0:
        source = "dedicated"
        m0 = m0_data.copy()
    elif not background_suppression and control_average is not None:
        source = "control_average"
        m0 = control_average.copy()
    else:
        return M0Result(
            source="absent",
            is_usable=False,
            intensity_ratio=0.0,
            artifact_flagged=False,
            smoothed_m0=None,
            low_signal_mask=None,
            message=(
                "No dedicated M0 scan and background suppression is active. "
                "Absolute CBF quantification is not possible; flagged for "
                "relative quantification only."
            ),
        )

    # ---- Intensity range & artifact check ----
    masked_vals = m0[mask]
    if len(masked_vals) == 0 or np.median(masked_vals) == 0:
        return M0Result(
            source=source,
            is_usable=False,
            intensity_ratio=0.0,
            artifact_flagged=True,
            smoothed_m0=None,
            low_signal_mask=None,
            message="M0 image has zero or empty signal within brain mask.",
        )

    max_val = np.max(masked_vals)
    median_val = np.median(masked_vals)
    intensity_ratio = float(max_val / median_val) if median_val != 0 else float("inf")
    artifact_flagged = intensity_ratio > max_median_ratio_threshold

    # ---- Preconditioning: Gaussian smoothing ----
    sigma = smoothing_fwhm_mm / (2.355 * voxel_size_mm)
    smoothed = gaussian_filter(m0, sigma=sigma)

    # ---- Low-signal masking ----
    global_mean = np.mean(smoothed[mask]) if np.any(mask) else 1.0
    low_threshold = low_signal_fraction * global_mean
    low_signal_mask = smoothed < low_threshold
    smoothed[low_signal_mask] = 0.0  # Prevent division-by-zero downstream

    msg_parts = [f"M0 source: {source}."]
    if artifact_flagged:
        msg_parts.append(
            f"WARNING: intensity ratio ({intensity_ratio:.1f}) exceeds "
            f"threshold ({max_median_ratio_threshold}). Possible coil "
            "proximity flare or susceptibility artifact."
        )
    msg_parts.append(
        f"Smoothed with {smoothing_fwhm_mm} mm FWHM. "
        f"{int(np.sum(low_signal_mask))} voxels masked as low-signal."
    )

    return M0Result(
        source=source,
        is_usable=not artifact_flagged,
        intensity_ratio=intensity_ratio,
        artifact_flagged=artifact_flagged,
        smoothed_m0=smoothed,
        low_signal_mask=low_signal_mask,
        message=" ".join(msg_parts),
    )
