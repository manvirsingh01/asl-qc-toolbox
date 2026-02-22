"""
Core QC metrics for raw BIDS ASL MRI data.

Evaluates the raw ASL timeseries *before* CBF quantification: volume
consistency, control–label balance, signal intensity, temporal stability,
NIfTI header sanity, and BIDS metadata completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CoreQCResult:
    """Container for core QC metrics computed on raw BIDS ASL data."""

    # ---- Volume structure ----
    n_volumes_nifti: int
    """Number of volumes in the 4-D NIfTI."""
    n_volumes_context: Optional[int] = None
    """Number of volumes declared in aslcontext.tsv (None if not provided)."""
    volume_count_match: Optional[bool] = None
    """True if NIfTI and aslcontext volume counts agree."""
    n_controls: int = 0
    """Number of control volumes."""
    n_labels: int = 0
    """Number of label volumes."""
    control_label_balanced: bool = True
    """True if n_controls == n_labels."""

    # ---- Signal intensity ----
    mean_signal: float = 0.0
    """Mean voxel intensity within the brain mask."""
    temporal_std_mean: float = 0.0
    """Mean temporal standard deviation within the brain mask."""
    global_tsnr: float = 0.0
    """Global temporal SNR (mean / std) within the brain mask."""

    # ---- Control–label difference ----
    mean_control_label_diff: float = 0.0
    """Mean absolute control–label difference (perfusion signal magnitude)."""

    # ---- M0 signal ----
    m0_mean_signal: Optional[float] = None
    """Mean M0 signal within the brain mask (None if no M0)."""
    m0_asl_ratio: Optional[float] = None
    """Ratio of M0 mean to ASL mean (expected > 1)."""

    # ---- Data integrity ----
    has_nan: bool = False
    """True if NaN values detected in the timeseries."""
    has_inf: bool = False
    """True if Inf values detected in the timeseries."""
    nan_voxel_count: int = 0
    """Number of voxels containing at least one NaN across time."""
    inf_voxel_count: int = 0
    """Number of voxels containing at least one Inf across time."""

    # ---- NIfTI header ----
    voxel_sizes: Optional[Tuple[float, ...]] = None
    """Voxel dimensions in mm (from NIfTI header)."""
    repetition_time: Optional[float] = None
    """TR in seconds (from NIfTI header pixdim[4])."""

    # ---- Metadata completeness ----
    missing_bids_fields: List[str] = field(default_factory=list)
    """List of required BIDS sidecar fields that are missing."""

    # ---- Advisory warnings ----
    warnings: List[str] = field(default_factory=list)
    """Human-readable warnings raised during QC evaluation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_data_integrity(
    timeseries_4d: np.ndarray,
) -> Tuple[bool, bool, int, int]:
    """Detect NaN and Inf values in the timeseries.

    Returns (has_nan, has_inf, nan_voxel_count, inf_voxel_count).
    """
    nan_mask = np.isnan(timeseries_4d)
    inf_mask = np.isinf(timeseries_4d)

    has_nan = bool(np.any(nan_mask))
    has_inf = bool(np.any(inf_mask))

    # Count spatial voxels that have at least one bad value across time
    nan_voxel_count = int(np.any(nan_mask, axis=-1).sum()) if has_nan else 0
    inf_voxel_count = int(np.any(inf_mask, axis=-1).sum()) if has_inf else 0

    return has_nan, has_inf, nan_voxel_count, inf_voxel_count


def _compute_signal_stats(
    timeseries_4d: np.ndarray,
    brain_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute mean signal, temporal std, and global tSNR within mask.

    Returns (mean_signal, temporal_std_mean, global_tsnr).
    """
    mask = brain_mask.astype(bool)
    temporal_mean = np.mean(timeseries_4d, axis=-1)
    temporal_std = np.std(timeseries_4d, axis=-1, ddof=1)

    masked_mean = temporal_mean[mask]
    masked_std = temporal_std[mask]

    mean_signal = float(np.mean(masked_mean)) if len(masked_mean) > 0 else 0.0
    temporal_std_mean = float(np.mean(masked_std)) if len(masked_std) > 0 else 0.0

    # tSNR per voxel, then take the global mean
    valid = (masked_std > 1e-12)
    if np.any(valid):
        tsnr_vals = masked_mean[valid] / masked_std[valid]
        global_tsnr = float(np.mean(tsnr_vals))
    else:
        global_tsnr = 0.0

    return mean_signal, temporal_std_mean, global_tsnr


def _compute_control_label_diff(
    timeseries_4d: np.ndarray,
    brain_mask: np.ndarray,
    control_indices: List[int],
    label_indices: List[int],
) -> float:
    """Mean absolute difference between paired control and label volumes."""
    mask = brain_mask.astype(bool)
    n_pairs = min(len(control_indices), len(label_indices))
    if n_pairs == 0:
        return 0.0

    diffs = []
    for i in range(n_pairs):
        ctrl_vol = timeseries_4d[..., control_indices[i]][mask]
        lbl_vol = timeseries_4d[..., label_indices[i]][mask]
        diffs.append(float(np.mean(np.abs(ctrl_vol - lbl_vol))))

    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_core_qc(
    timeseries_4d: np.ndarray,
    brain_mask: np.ndarray,
    *,
    volume_types: Optional[List[str]] = None,
    voxel_sizes: Optional[Tuple[float, ...]] = None,
    repetition_time: Optional[float] = None,
    missing_bids_fields: Optional[List[str]] = None,
) -> CoreQCResult:
    """Compute core QC metrics on a raw BIDS ASL timeseries.

    Parameters
    ----------
    timeseries_4d : np.ndarray
        (X, Y, Z, T) raw ASL timeseries.
    brain_mask : np.ndarray
        Boolean 3-D mask.
    volume_types : list of str, optional
        Ordered volume types from ``aslcontext.tsv``
        (e.g. ``['control', 'label', 'control', 'label', ...]``).
    voxel_sizes : tuple of float, optional
        Voxel dimensions in mm from the NIfTI header.
    repetition_time : float, optional
        TR in seconds from the NIfTI header.
    missing_bids_fields : list of str, optional
        Names of required BIDS fields that are absent.

    Returns
    -------
    CoreQCResult
    """
    warnings: List[str] = []
    n_vols = timeseries_4d.shape[-1] if timeseries_4d.ndim == 4 else 1

    # ---- Volume structure ----
    n_volumes_context: Optional[int] = None
    volume_count_match: Optional[bool] = None
    n_controls = 0
    n_labels = 0
    control_indices: List[int] = []
    label_indices: List[int] = []
    m0_indices: List[int] = []

    if volume_types is not None:
        n_volumes_context = len(volume_types)
        volume_count_match = (n_vols == n_volumes_context)
        if not volume_count_match:
            warnings.append(
                f"Volume count mismatch: NIfTI has {n_vols} volumes but "
                f"aslcontext.tsv declares {n_volumes_context}."
            )
        control_indices = [i for i, v in enumerate(volume_types) if v == "control"]
        label_indices = [i for i, v in enumerate(volume_types) if v == "label"]
        m0_indices = [i for i, v in enumerate(volume_types) if v == "m0scan"]
        n_controls = len(control_indices)
        n_labels = len(label_indices)

    control_label_balanced = (n_controls == n_labels) if volume_types else True
    if not control_label_balanced:
        warnings.append(
            f"Control–label imbalance: {n_controls} controls vs {n_labels} labels."
        )

    # ---- Data integrity ----
    has_nan, has_inf, nan_voxel_count, inf_voxel_count = _check_data_integrity(
        timeseries_4d
    )
    if has_nan:
        warnings.append(f"NaN values detected in {nan_voxel_count} voxels.")
    if has_inf:
        warnings.append(f"Inf values detected in {inf_voxel_count} voxels.")

    # ---- Signal statistics ----
    mean_signal, temporal_std_mean, global_tsnr = _compute_signal_stats(
        timeseries_4d, brain_mask
    )
    if mean_signal < 1.0:
        warnings.append(
            f"Very low mean signal ({mean_signal:.2f}). Possible dropout or "
            "empty acquisition."
        )

    # ---- Control–label difference ----
    mean_cl_diff = 0.0
    if control_indices and label_indices:
        mean_cl_diff = _compute_control_label_diff(
            timeseries_4d, brain_mask, control_indices, label_indices
        )
        if mean_cl_diff < 1e-3:
            warnings.append(
                "Control–label difference is near zero — perfusion signal "
                "may be absent."
            )

    # ---- M0 signal ----
    m0_mean_signal: Optional[float] = None
    m0_asl_ratio: Optional[float] = None
    if m0_indices and timeseries_4d.ndim == 4:
        mask = brain_mask.astype(bool)
        m0_vols = [timeseries_4d[..., i][mask] for i in m0_indices
                    if i < n_vols]
        if m0_vols:
            m0_mean_signal = float(np.mean(np.concatenate(m0_vols)))
            if mean_signal > 1e-12:
                m0_asl_ratio = m0_mean_signal / mean_signal

    # ---- Voxel size warnings ----
    if voxel_sizes is not None:
        spatial = voxel_sizes[:3] if len(voxel_sizes) >= 3 else voxel_sizes
        for i, sz in enumerate(spatial):
            if sz < 0.5 or sz > 10.0:
                warnings.append(
                    f"Unusual voxel size along axis {i}: {sz:.2f} mm."
                )

    # ---- Metadata completeness ----
    bids_missing = missing_bids_fields or []
    if bids_missing:
        warnings.append(
            f"Missing required BIDS fields: {', '.join(bids_missing)}."
        )

    return CoreQCResult(
        n_volumes_nifti=n_vols,
        n_volumes_context=n_volumes_context,
        volume_count_match=volume_count_match,
        n_controls=n_controls,
        n_labels=n_labels,
        control_label_balanced=control_label_balanced,
        mean_signal=mean_signal,
        temporal_std_mean=temporal_std_mean,
        global_tsnr=global_tsnr,
        mean_control_label_diff=mean_cl_diff,
        m0_mean_signal=m0_mean_signal,
        m0_asl_ratio=m0_asl_ratio,
        has_nan=has_nan,
        has_inf=has_inf,
        nan_voxel_count=nan_voxel_count,
        inf_voxel_count=inf_voxel_count,
        voxel_sizes=voxel_sizes,
        repetition_time=repetition_time,
        missing_bids_fields=bids_missing,
        warnings=warnings,
    )
