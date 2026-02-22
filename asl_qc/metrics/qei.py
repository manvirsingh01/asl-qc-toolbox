"""
Quality Evaluation Index (QEI) for ASL CBF maps.

The QEI is a scalar metric ∈ [0, 1] that synthesises structural similarity,
spatial variability, and negative-voxel proportion into a single quality
score.  Values closer to 1 indicate superior image quality.

Reference
---------
Automated Quality Evaluation Index for Arterial Spin Labeling Derived
Cerebral Blood Flow Maps (Shirzadi et al., ISMRM 2017 / NeuroImage 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr


@dataclass
class QEIResult:
    """Container for QEI computation results."""

    qei: float
    """Final Quality Evaluation Index ∈ [0, 1]."""
    structural_similarity: float
    """Pearson *r* between actual CBF and structural pseudo-CBF (C_ss)."""
    index_of_dispersion: float
    """Normalised pooled variance across tissue classes (C_v)."""
    negative_gm_fraction: float
    """Proportion of GM voxels with negative CBF (C_neg)."""
    mean_gm_cbf: float
    """Mean CBF within the GM mask."""


def compute_qei(
    cbf_map: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
    csf_mask: np.ndarray | None = None,
    *,
    gm_weight: float = 2.5,
    wm_weight: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> QEIResult:
    """Compute the Quality Evaluation Index for a CBF map.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D absolute CBF map (ml/100 g/min).
    gm_mask, wm_mask : np.ndarray
        Binary (or probabilistic) 3-D tissue masks.
    csf_mask : np.ndarray, optional
        Binary CSF mask (used for dispersion calculation only).
    gm_weight, wm_weight : float
        Weights for constructing the structural pseudo-CBF map.
    alpha, beta, gamma : float
        Exponents in the QEI formula.

    Returns
    -------
    QEIResult
    """
    gm = gm_mask.astype(bool)
    wm = wm_mask.astype(bool)

    # ---- Structural Similarity (C_ss) ----
    # Build structural pseudo-CBF: GM × 2.5 + WM × 1.0
    sp_cbf = np.zeros_like(cbf_map, dtype=np.float64)
    sp_cbf[gm] = gm_weight
    sp_cbf[wm] = wm_weight

    # ROI = union of GM and WM
    roi = gm | wm
    if np.sum(roi) < 10:
        # Degenerate case
        return QEIResult(
            qei=0.0,
            structural_similarity=0.0,
            index_of_dispersion=float("inf"),
            negative_gm_fraction=1.0,
            mean_gm_cbf=0.0,
        )

    cbf_roi = cbf_map[roi]
    sp_roi = sp_cbf[roi]

    if np.std(cbf_roi) < 1e-12 or np.std(sp_roi) < 1e-12:
        c_ss = 0.0
    else:
        c_ss, _ = pearsonr(cbf_roi, sp_roi)
        c_ss = max(c_ss, 0.0)  # Clamp to non-negative

    # ---- Index of Dispersion (C_v) ----
    gm_cbf = cbf_map[gm]
    wm_cbf = cbf_map[wm]
    mean_gm = float(np.mean(gm_cbf)) if len(gm_cbf) > 0 else 1.0

    vars_list = []
    counts = []
    if len(gm_cbf) > 1:
        vars_list.append(np.var(gm_cbf, ddof=1))
        counts.append(len(gm_cbf))
    if len(wm_cbf) > 1:
        vars_list.append(np.var(wm_cbf, ddof=1))
        counts.append(len(wm_cbf))
    if csf_mask is not None:
        csf = csf_mask.astype(bool)
        csf_cbf = cbf_map[csf]
        if len(csf_cbf) > 1:
            vars_list.append(np.var(csf_cbf, ddof=1))
            counts.append(len(csf_cbf))

    if vars_list:
        total_n = sum(counts)
        pooled_var = sum(v * n for v, n in zip(vars_list, counts)) / total_n
        c_v = pooled_var / (abs(mean_gm) + 1e-12)
    else:
        c_v = float("inf")

    # ---- Negative GM CBF (C_neg) ----
    if len(gm_cbf) > 0:
        c_neg = float(np.sum(gm_cbf < 0) / len(gm_cbf))
    else:
        c_neg = 1.0

    # ---- QEI formula ----
    # QEI = (C_ss)^α × exp(-C_v)^β × (1 - C_neg)^γ
    qei = (c_ss ** alpha) * (np.exp(-c_v) ** beta) * ((1.0 - c_neg) ** gamma)
    qei = float(np.clip(qei, 0.0, 1.0))

    return QEIResult(
        qei=qei,
        structural_similarity=float(c_ss),
        index_of_dispersion=float(c_v),
        negative_gm_fraction=float(c_neg),
        mean_gm_cbf=float(mean_gm),
    )
