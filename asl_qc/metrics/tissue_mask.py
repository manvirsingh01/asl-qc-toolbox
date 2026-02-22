"""
Tissue mask quality and spatial overlap metrics.

Evaluates the fidelity of structural segmentation and co-registration
between the T1-weighted anatomy and the ASL space using the Dice
Similarity Coefficient, Jaccard Index, and physiological GM/WM CBF ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TissueMaskResult:
    """Container for tissue mask quality metrics."""

    dice_coefficient: float
    """Dice Similarity Coefficient between subject mask and reference."""
    jaccard_index: float
    """Jaccard Index (intersection over union)."""
    gm_wm_cbf_ratio: float
    """Ratio of mean GM CBF to mean WM CBF."""
    mean_gm_cbf: float
    mean_wm_cbf: float
    message: str = ""


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the Dice Similarity Coefficient.

    DSC = 2|A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Boolean 3-D masks.

    Returns
    -------
    float
        DSC ∈ [0, 1].
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 0.0
    return float(2 * intersection / total)


def compute_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the Jaccard Index.

    JI = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Boolean 3-D masks.

    Returns
    -------
    float
        JI ∈ [0, 1].
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_gm_wm_ratio(
    cbf_map: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute the GM/WM CBF ratio.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    gm_mask, wm_mask : np.ndarray
        Boolean 3-D masks.

    Returns
    -------
    ratio : float
    mean_gm : float
    mean_wm : float
    """
    gm = gm_mask.astype(bool)
    wm = wm_mask.astype(bool)

    gm_vals = cbf_map[gm]
    wm_vals = cbf_map[wm]

    mean_gm = float(np.mean(gm_vals)) if len(gm_vals) > 0 else 0.0
    mean_wm = float(np.mean(wm_vals)) if len(wm_vals) > 0 else 0.0

    if abs(mean_wm) < 1e-12:
        ratio = float("inf")
    else:
        ratio = mean_gm / mean_wm

    return ratio, mean_gm, mean_wm


def compute_tissue_mask_result(
    cbf_map: np.ndarray,
    subject_mask: np.ndarray,
    reference_mask: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
) -> TissueMaskResult:
    """Full tissue mask quality assessment.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    subject_mask : np.ndarray
        Subject-space tissue mask (e.g. brain mask in ASL space).
    reference_mask : np.ndarray
        Reference/atlas mask (e.g. MNI template mask).
    gm_mask, wm_mask : np.ndarray
        Tissue probability maps thresholded to binary.

    Returns
    -------
    TissueMaskResult
    """
    dice = compute_dice(subject_mask, reference_mask)
    jaccard = compute_jaccard(subject_mask, reference_mask)
    ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf_map, gm_mask, wm_mask)

    msg_parts = []
    if dice < 0.70:
        msg_parts.append(
            f"WARNING: Dice coefficient ({dice:.3f}) below 0.70 — "
            "possible normalization failure."
        )
    if ratio < 2.0 or ratio > 3.0:
        msg_parts.append(
            f"WARNING: GM/WM ratio ({ratio:.2f}) outside expected range [2.0, 3.0]."
        )

    return TissueMaskResult(
        dice_coefficient=dice,
        jaccard_index=jaccard,
        gm_wm_cbf_ratio=ratio,
        mean_gm_cbf=mean_gm,
        mean_wm_cbf=mean_wm,
        message=" ".join(msg_parts) if msg_parts else "Tissue masks within normal parameters.",
    )
