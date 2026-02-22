"""
Control-label pattern verification for ASL timeseries.

Validates that the temporal interleaving of control and label volumes
matches the declared BIDS ``aslcontext.tsv`` by analysing the global
mean intensity periodicity of the 4-D acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PatternResult:
    """Result of control-label pattern verification."""

    is_valid: bool
    """``True`` if the physical data matches the declared BIDS context."""
    phase_shift_detected: bool
    """``True`` if the intensity zig-zag is inverted w.r.t. the metadata."""
    intensity_timecourse: np.ndarray = field(repr=False)
    """Per-volume global mean intensity within the brain mask."""
    autocorrelation_value: float = 0.0
    """Lag-1 autocorrelation of the intensity timecourse."""
    message: str = ""


def _compute_volume_means(data_4d: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Return mean intensity within *brain_mask* for each volume."""
    mask = brain_mask.astype(bool)
    n_vols = data_4d.shape[-1]
    means = np.empty(n_vols)
    for t in range(n_vols):
        vol = data_4d[..., t]
        means[t] = np.mean(vol[mask]) if np.any(mask) else 0.0
    return means


def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a 1-D signal."""
    if len(x) < 3:
        return 0.0
    x_centered = x - np.mean(x)
    denom = np.sum(x_centered ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(x_centered[:-1] * x_centered[1:]) / denom)


def verify_control_label_pattern(
    data_4d: np.ndarray,
    asl_context_types: List[str],
    brain_mask: np.ndarray,
    min_amplitude: float = 0.005,
) -> PatternResult:
    """Verify that the control-label alternation in the 4-D data is consistent
    with the declared BIDS volume types.

    Parameters
    ----------
    data_4d : np.ndarray
        4-D ASL timeseries (X, Y, Z, T).
    asl_context_types : list of str
        Ordered volume types from ``aslcontext.tsv``
        (``"control"``, ``"label"``, ``"m0scan"``, …).
    brain_mask : np.ndarray
        Boolean 3-D mask.
    min_amplitude : float
        Minimum fractional amplitude of the zig-zag pattern to consider
        the data valid.

    Returns
    -------
    PatternResult
    """
    n_vols = data_4d.shape[-1]

    # Sanity: volume count must match context length
    if n_vols != len(asl_context_types):
        return PatternResult(
            is_valid=False,
            phase_shift_detected=False,
            intensity_timecourse=np.array([]),
            message=(
                f"Volume count mismatch: data has {n_vols} volumes but "
                f"aslcontext declares {len(asl_context_types)}."
            ),
        )

    # Compute per-volume means
    means = _compute_volume_means(data_4d, brain_mask)

    # Only analyse control / label volumes (ignore m0scan, deltam, cbf)
    cl_indices = [
        i for i, vt in enumerate(asl_context_types) if vt in ("control", "label")
    ]

    if len(cl_indices) < 4:
        return PatternResult(
            is_valid=True,  # Too few volumes to verify pattern
            phase_shift_detected=False,
            intensity_timecourse=means,
            message="Fewer than 4 control/label volumes; pattern check skipped.",
        )

    cl_means = means[cl_indices]
    cl_types = [asl_context_types[i] for i in cl_indices]

    # Expected pattern: control=high, label=low → alternating sign
    expected_sign = np.array(
        [1.0 if vt == "control" else -1.0 for vt in cl_types]
    )

    # Normalise means to zero-mean
    cl_centered = cl_means - np.mean(cl_means)

    # Check amplitude
    amplitude = (np.max(cl_means) - np.min(cl_means)) / (np.mean(cl_means) + 1e-12)
    if amplitude < min_amplitude:
        return PatternResult(
            is_valid=False,
            phase_shift_detected=False,
            intensity_timecourse=means,
            message=(
                f"Zig-zag amplitude too low ({amplitude:.4f} < {min_amplitude}). "
                "Possible background suppression failure or flat signal."
            ),
        )

    # Correlation between expected sign pattern and actual centred means
    corr = np.corrcoef(expected_sign, cl_centered)[0, 1]

    # Lag-1 autocorrelation (should be strongly negative for alternating data)
    ac = _lag1_autocorrelation(cl_means)

    if corr > 0.5 and ac < -0.3:
        # Data matches expectations
        return PatternResult(
            is_valid=True,
            phase_shift_detected=False,
            intensity_timecourse=means,
            autocorrelation_value=ac,
            message="Control-label pattern verified successfully.",
        )
    elif corr < -0.5 and ac < -0.3:
        # Phase is inverted (label=high, control=low)
        return PatternResult(
            is_valid=False,
            phase_shift_detected=True,
            intensity_timecourse=means,
            autocorrelation_value=ac,
            message=(
                "Pattern Desynchronization Error: intensity pattern is "
                "inverted relative to declared aslcontext. The data labels "
                "and controls appear swapped."
            ),
        )
    else:
        return PatternResult(
            is_valid=False,
            phase_shift_detected=False,
            intensity_timecourse=means,
            autocorrelation_value=ac,
            message=(
                f"Ambiguous control-label pattern (correlation={corr:.3f}, "
                f"autocorrelation={ac:.3f}). Possible sequence interruption "
                "or non-standard labeling scheme."
            ),
        )
