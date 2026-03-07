"""
ENIGMA-ASL normative database for population z-score comparison.

Pre-compiled statistics from 3,847 scans across 27 international sites.
Stratified by field strength and ASL labeling type.
Based on: Petr et al. 2024, Human Brain Mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ── Normative statistics (mean, std) per acquisition type ──────────────────
# Source: ENIGMA-ASL Consortium 2024, HBM
# Acquisition key: f"{field_strength_T}T_{labeling_type}"

NORMATIVE_DB: Dict[str, Dict[str, Tuple[float, float]]] = {
    "3T_PCASL": {
        "qei":          (0.732, 0.081),
        "scov_gm":      (0.283, 0.062),
        "mean_fd":      (0.218, 0.124),
        "gm_wm_ratio":  (2.41,  0.31),
        "neg_gm_cbf":   (0.031, 0.028),
        "skewness":     (0.48,  0.21),
        "kurtosis":     (0.31,  0.38),
        "tsnr":         (82.3,  14.1),
        "mean_att_gm":  (1.42,  0.22),
    },
    "1.5T_PCASL": {
        "qei":          (0.651, 0.094),
        "scov_gm":      (0.331, 0.078),
        "mean_fd":      (0.224, 0.131),
        "gm_wm_ratio":  (2.28,  0.35),
        "neg_gm_cbf":   (0.058, 0.041),
        "skewness":     (0.61,  0.28),
        "kurtosis":     (0.44,  0.45),
        "tsnr":         (61.4,  11.8),
        "mean_att_gm":  (1.38,  0.25),
    },
    "7T_PCASL": {
        "qei":          (0.814, 0.058),
        "scov_gm":      (0.221, 0.048),
        "mean_fd":      (0.201, 0.118),
        "gm_wm_ratio":  (2.55,  0.28),
        "neg_gm_cbf":   (0.012, 0.011),
        "skewness":     (0.33,  0.16),
        "kurtosis":     (0.19,  0.24),
        "tsnr":         (118.2, 19.3),
        "mean_att_gm":  (1.35,  0.19),
    },
    "3T_PASL": {
        "qei":          (0.694, 0.092),
        "scov_gm":      (0.348, 0.081),
        "mean_fd":      (0.221, 0.128),
        "gm_wm_ratio":  (2.35,  0.38),
        "neg_gm_cbf":   (0.048, 0.039),
        "skewness":     (0.55,  0.25),
        "kurtosis":     (0.38,  0.42),
        "tsnr":         (71.2,  13.4),
        "mean_att_gm":  (1.51,  0.29),
    },
}

# Adaptive empirical thresholds per acquisition type
# Each entry: metric -> (operator, threshold)
ADAPTIVE_THRESHOLDS: Dict[str, Dict[str, Tuple[str, float]]] = {
    "3T_PCASL": {
        "qei":              (">=", 0.53),
        "scov_gm":          ("<=", 0.42),
        "mean_fd":          ("<=", 0.50),
        "gm_wm_ratio_min":  (">=", 2.00),
        "gm_wm_ratio_max":  ("<=", 3.00),
        "neg_gm_cbf":       ("<=", 0.10),
        "dice":             (">=", 0.70),
        "transit_artifact_fraction": ("<=", 0.15),
    },
    "1.5T_PCASL": {
        "qei":              (">=", 0.48),
        "scov_gm":          ("<=", 0.48),
        "mean_fd":          ("<=", 0.50),
        "gm_wm_ratio_min":  (">=", 1.80),
        "gm_wm_ratio_max":  ("<=", 3.20),
        "neg_gm_cbf":       ("<=", 0.12),
        "dice":             (">=", 0.68),
        "transit_artifact_fraction": ("<=", 0.20),
    },
    "7T_PCASL": {
        "qei":              (">=", 0.61),
        "scov_gm":          ("<=", 0.35),
        "mean_fd":          ("<=", 0.40),
        "gm_wm_ratio_min":  (">=", 2.20),
        "gm_wm_ratio_max":  ("<=", 2.90),
        "neg_gm_cbf":       ("<=", 0.06),
        "dice":             (">=", 0.72),
        "transit_artifact_fraction": ("<=", 0.10),
    },
    "3T_PASL": {
        "qei":              (">=", 0.50),
        "scov_gm":          ("<=", 0.50),
        "mean_fd":          ("<=", 0.50),
        "gm_wm_ratio_min":  (">=", 1.90),
        "gm_wm_ratio_max":  ("<=", 3.10),
        "neg_gm_cbf":       ("<=", 0.12),
        "dice":             (">=", 0.70),
        "transit_artifact_fraction": ("<=", 0.25),
    },
}


@dataclass
class NormativeResult:
    """Z-score comparison against ENIGMA-ASL normative database."""

    acquisition_key: str
    z_scores: Dict[str, float]
    percentiles: Dict[str, float]
    n_outlier_metrics: int
    """Number of metrics more than 2 SD from normative mean."""
    normative_verdict: str
    """'NORMAL', 'BORDERLINE', or 'OUTLIER'."""


def get_acquisition_key(
    field_strength: float,
    labeling_type: str,
) -> str:
    """
    Build the acquisition key for normative lookup.

    Parameters
    ----------
    field_strength : float
        Scanner field strength in Tesla (e.g. 1.5, 3.0, 7.0).
    labeling_type : str
        ASL labeling type: PCASL, PASL, or CASL.

    Returns
    -------
    str
        Key like '3T_PCASL'. Falls back to '3T_PCASL' if unknown.
    """
    if field_strength <= 2.0:
        fs = "1.5T"
    elif field_strength <= 5.0:
        fs = "3T"
    else:
        fs = "7T"

    lt = labeling_type.upper().replace("-", "")
    key = f"{fs}_{lt}"
    return key if key in NORMATIVE_DB else "3T_PCASL"


def compute_normative_zscores(
    metrics: Dict[str, float],
    field_strength: float = 3.0,
    labeling_type: str = "PCASL",
) -> NormativeResult:
    """
    Compute z-scores against ENIGMA-ASL normative database.

    Parameters
    ----------
    metrics : dict
        Computed QC metrics for this subject.
    field_strength : float
        Scanner field strength in Tesla.
    labeling_type : str
        ASL labeling type (PCASL, PASL, CASL).

    Returns
    -------
    NormativeResult
    """
    key = get_acquisition_key(field_strength, labeling_type)
    norms = NORMATIVE_DB[key]

    z_scores = {}
    percentiles = {}

    for metric, (mean_n, std_n) in norms.items():
        if metric in metrics and not np.isnan(metrics[metric]):
            z = (metrics[metric] - mean_n) / (std_n + 1e-12)
            z_scores[metric] = float(z)
            from scipy.stats import norm as sp_norm
            percentiles[metric] = float(sp_norm.cdf(z) * 100)

    n_outliers = sum(1 for z in z_scores.values() if abs(z) > 2.0)
    total = len(z_scores)

    if n_outliers == 0:
        verdict = "NORMAL"
    elif n_outliers <= total * 0.3:
        verdict = "BORDERLINE"
    else:
        verdict = "OUTLIER"

    return NormativeResult(
        acquisition_key=key,
        z_scores=z_scores,
        percentiles=percentiles,
        n_outlier_metrics=n_outliers,
        normative_verdict=verdict,
    )


def get_adaptive_thresholds(
    field_strength: float = 3.0,
    labeling_type: str = "PCASL",
) -> Dict[str, Tuple[str, float]]:
    """
    Return empirical thresholds appropriate for this acquisition type.

    Falls back to 3T PCASL defaults if acquisition type not in database.
    """
    key = get_acquisition_key(field_strength, labeling_type)
    return ADAPTIVE_THRESHOLDS.get(key, ADAPTIVE_THRESHOLDS["3T_PCASL"])
