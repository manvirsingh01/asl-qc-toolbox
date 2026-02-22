"""
SCORE and ENABLE outlier-rejection algorithms for ASL data.

**SCORE** (Structural Correlation-based Outlier Rejection) evaluates each
control-label difference image against the median perfusion map and
discards pairs whose correlation falls below an adaptive threshold.

**ENABLE** (ENhancement of Automated Blood fLow Estimates) sorts
multi-PLD volumes by temporal quality and strips the worst contributors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ScoreResult:
    """Result of SCORE outlier rejection."""

    n_input_pairs: int
    n_retained_pairs: int
    retained_indices: List[int]
    rejected_indices: List[int]
    correlation_values: np.ndarray = field(repr=False)
    threshold_used: float
    message: str = ""


@dataclass
class EnableResult:
    """Result of ENABLE volume sorting."""

    sorted_indices: List[int]
    quality_scores: np.ndarray = field(repr=False)
    n_stripped: int
    message: str = ""


def score_rejection(
    delta_m_pairs: np.ndarray,
    brain_mask: np.ndarray,
    correlation_threshold: float = 0.6,
    min_retained_fraction: float = 0.50,
) -> ScoreResult:
    """Apply the SCORE algorithm to reject outlier ΔM pairs.

    Parameters
    ----------
    delta_m_pairs : np.ndarray
        4-D array (X, Y, Z, N_pairs) where each volume is a single
        control-label difference image (ΔM).
    brain_mask : np.ndarray
        Boolean 3-D brain mask.
    correlation_threshold : float
        Minimum Pearson *r* against the median perfusion map.
    min_retained_fraction : float
        Never discard more than ``(1 - min_retained_fraction)`` of pairs.

    Returns
    -------
    ScoreResult
    """
    mask = brain_mask.astype(bool)
    n_pairs = delta_m_pairs.shape[-1]

    if n_pairs < 3:
        return ScoreResult(
            n_input_pairs=n_pairs,
            n_retained_pairs=n_pairs,
            retained_indices=list(range(n_pairs)),
            rejected_indices=[],
            correlation_values=np.ones(n_pairs),
            threshold_used=correlation_threshold,
            message="Fewer than 3 pairs; SCORE skipped.",
        )

    # Compute median perfusion map
    median_map = np.median(delta_m_pairs, axis=-1)
    median_flat = median_map[mask]

    # Correlate each pair against the median
    correlations = np.empty(n_pairs)
    for i in range(n_pairs):
        pair_flat = delta_m_pairs[..., i][mask]
        r = np.corrcoef(median_flat, pair_flat)[0, 1]
        correlations[i] = r if np.isfinite(r) else 0.0

    # Determine which pairs to reject
    rejected = np.where(correlations < correlation_threshold)[0].tolist()

    # Enforce minimum retention
    min_keep = max(2, int(np.ceil(n_pairs * min_retained_fraction)))
    if n_pairs - len(rejected) < min_keep:
        # Sort by correlation and keep the best min_keep
        sorted_idx = np.argsort(correlations)[::-1]
        retained = sorted(sorted_idx[:min_keep].tolist())
        rejected = sorted(sorted_idx[min_keep:].tolist())
    else:
        retained = sorted(set(range(n_pairs)) - set(rejected))

    return ScoreResult(
        n_input_pairs=n_pairs,
        n_retained_pairs=len(retained),
        retained_indices=retained,
        rejected_indices=rejected,
        correlation_values=correlations,
        threshold_used=correlation_threshold,
        message=(
            f"SCORE retained {len(retained)}/{n_pairs} pairs "
            f"(threshold r={correlation_threshold:.2f})."
        ),
    )


def enable_sorting(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
    strip_fraction: float = 0.20,
) -> EnableResult:
    """Apply the ENABLE algorithm to sort and strip low-quality volumes.

    Volumes are ranked by their temporal stability (inverse of their
    contribution to global variance). The worst ``strip_fraction`` are
    removed.

    Parameters
    ----------
    data_4d : np.ndarray
        4-D timeseries (X, Y, Z, T).
    brain_mask : np.ndarray
        Boolean 3-D mask.
    strip_fraction : float
        Fraction of volumes to strip (0–1).

    Returns
    -------
    EnableResult
    """
    mask = brain_mask.astype(bool)
    n_vols = data_4d.shape[-1]

    if n_vols < 3:
        return EnableResult(
            sorted_indices=list(range(n_vols)),
            quality_scores=np.ones(n_vols),
            n_stripped=0,
            message="Fewer than 3 volumes; ENABLE skipped.",
        )

    # Compute quality score: inverse of each volume's deviation from
    # the temporal mean
    mean_vol = np.mean(data_4d, axis=-1)
    quality = np.empty(n_vols)
    for t in range(n_vols):
        diff = data_4d[..., t][mask] - mean_vol[mask]
        rms_dev = np.sqrt(np.mean(diff ** 2))
        quality[t] = 1.0 / (rms_dev + 1e-12)

    # Sort by quality (best first)
    sorted_idx = np.argsort(quality)[::-1].tolist()

    # Strip worst fraction
    n_strip = int(np.floor(n_vols * strip_fraction))
    n_strip = min(n_strip, n_vols - 2)  # Keep at least 2

    return EnableResult(
        sorted_indices=sorted_idx,
        quality_scores=quality,
        n_stripped=n_strip,
        message=(
            f"ENABLE sorted {n_vols} volumes; "
            f"stripping {n_strip} lowest-quality volumes."
        ),
    )
