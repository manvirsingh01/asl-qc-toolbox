"""
Signal-to-noise metrics for ASL timeseries.

Provides temporal SNR (tSNR) and RMS difference computations that assess
the stability of the acquisition and the detectability of the perfusion
signal above the thermal noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SNRResult:
    """Container for tSNR and RMS difference metrics."""

    global_tsnr: float
    """Mean tSNR across the brain mask."""
    median_tsnr: float
    """Median tSNR across the brain mask."""
    rms_difference: float
    """Global RMS difference between successive control-label pairs."""
    tsnr_map: np.ndarray
    """3-D voxel-wise tSNR map."""


def compute_temporal_snr(
    timeseries_4d: np.ndarray,
    brain_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Compute voxel-wise temporal SNR.

    tSNR(v) = mean(v, t) / std(v, t)

    Parameters
    ----------
    timeseries_4d : np.ndarray
        (X, Y, Z, T) timeseries.
    brain_mask : np.ndarray
        Boolean 3-D mask.

    Returns
    -------
    tsnr_map : np.ndarray
        3-D voxel-wise tSNR.
    global_tsnr : float
        Mean of tSNR within the mask.
    median_tsnr : float
        Median of tSNR within the mask.
    """
    mask = brain_mask.astype(bool)

    temporal_mean = np.mean(timeseries_4d, axis=-1)
    temporal_std = np.std(timeseries_4d, axis=-1, ddof=1)

    # Avoid division by zero
    tsnr_map = np.zeros_like(temporal_mean)
    valid = (temporal_std > 1e-12) & mask
    tsnr_map[valid] = temporal_mean[valid] / temporal_std[valid]

    masked_tsnr = tsnr_map[mask]
    global_tsnr = float(np.mean(masked_tsnr)) if len(masked_tsnr) > 0 else 0.0
    median_tsnr = float(np.median(masked_tsnr)) if len(masked_tsnr) > 0 else 0.0

    return tsnr_map, global_tsnr, median_tsnr


def compute_rms_difference(
    control_vols: np.ndarray,
    label_vols: np.ndarray,
    brain_mask: np.ndarray,
) -> float:
    """Compute the global RMS difference across control-label pairs.

    Parameters
    ----------
    control_vols : np.ndarray
        4-D (X, Y, Z, N_pairs) control volumes.
    label_vols : np.ndarray
        4-D (X, Y, Z, N_pairs) label volumes (same shape).
    brain_mask : np.ndarray
        Boolean 3-D mask.

    Returns
    -------
    float
        Global RMS difference.
    """
    mask = brain_mask.astype(bool)
    n_pairs = control_vols.shape[-1]

    if n_pairs == 0:
        return 0.0

    rms_vals = np.empty(n_pairs)
    for i in range(n_pairs):
        diff = control_vols[..., i][mask] - label_vols[..., i][mask]
        rms_vals[i] = np.sqrt(np.mean(diff ** 2))

    return float(np.mean(rms_vals))


def compute_snr_result(
    timeseries_4d: np.ndarray,
    brain_mask: np.ndarray,
    control_vols: np.ndarray | None = None,
    label_vols: np.ndarray | None = None,
) -> SNRResult:
    """Full SNR analysis.

    Parameters
    ----------
    timeseries_4d : np.ndarray
        (X, Y, Z, T) timeseries.
    brain_mask : np.ndarray
        Boolean 3-D mask.
    control_vols, label_vols : np.ndarray, optional
        Separate control and label volumes for RMS difference.

    Returns
    -------
    SNRResult
    """
    tsnr_map, global_tsnr, median_tsnr = compute_temporal_snr(
        timeseries_4d, brain_mask
    )

    rms_diff = 0.0
    if control_vols is not None and label_vols is not None:
        rms_diff = compute_rms_difference(control_vols, label_vols, brain_mask)

    return SNRResult(
        global_tsnr=global_tsnr,
        median_tsnr=median_tsnr,
        rms_difference=rms_diff,
        tsnr_map=tsnr_map,
    )
