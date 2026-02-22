"""
Histogram analysis of ASL CBF distributions.

Extracts distributional moments (skewness, kurtosis) and percentile
bounds from the voxel intensity histogram within a tissue mask, providing
secondary markers for vascular artifacts and noise contamination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class HistogramResult:
    """Summary statistics from the CBF histogram within a tissue mask."""

    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    percentile_5: float
    percentile_95: float
    iqr: float
    n_voxels: int


def analyze_histogram(
    cbf_map: np.ndarray,
    tissue_mask: np.ndarray,
) -> HistogramResult:
    """Compute histogram-derived statistics for CBF within *tissue_mask*.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    tissue_mask : np.ndarray
        Boolean 3-D mask (e.g. GM mask).

    Returns
    -------
    HistogramResult
    """
    mask = tissue_mask.astype(bool)
    vals = cbf_map[mask].astype(np.float64)

    if len(vals) < 4:
        return HistogramResult(
            mean=0.0,
            median=0.0,
            std=0.0,
            skewness=0.0,
            kurtosis=0.0,
            percentile_5=0.0,
            percentile_95=0.0,
            iqr=0.0,
            n_voxels=len(vals),
        )

    return HistogramResult(
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        std=float(np.std(vals, ddof=1)),
        skewness=float(sp_stats.skew(vals, bias=False)),
        kurtosis=float(sp_stats.kurtosis(vals, bias=False)),  # Excess kurtosis
        percentile_5=float(np.percentile(vals, 5)),
        percentile_95=float(np.percentile(vals, 95)),
        iqr=float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        n_voxels=len(vals),
    )
