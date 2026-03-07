"""
Partial Volume Correction (PVC) QC metrics.

Assesses the impact of partial volume effects on CBF estimates and
evaluates whether PVC was applied appropriately. Based on:
Chappell et al. 2021, MRM — PVC for ASL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PVCQCResult:
    """QC results comparing non-PVC and PVC CBF maps."""

    gm_cbf_uncorrected: float
    gm_cbf_pvc: float
    wm_cbf_uncorrected: float
    wm_cbf_pvc: float
    ratio_uncorrected: float
    ratio_pvc: float
    pvc_impact_gm: float
    """Absolute change in GM CBF due to PVC (ml/100g/min)."""
    pvc_impact_ratio: float
    """Change in GM/WM ratio due to PVC."""
    pvc_needed: bool
    """True if PVC substantially changes the CBF estimates."""
    message: str = ""


def compute_pvc_qc(
    cbf_uncorrected: np.ndarray,
    cbf_pvc: np.ndarray,
    gm_probability: np.ndarray,
    wm_probability: np.ndarray,
    *,
    pvc_threshold: float = 0.20,
) -> PVCQCResult:
    """
    Assess the impact of partial volume correction.

    Compares probability-weighted CBF means before and after PVC.
    If PVC changes GM CBF by more than pvc_threshold fraction,
    it is flagged as necessary.

    Parameters
    ----------
    cbf_uncorrected : np.ndarray
        CBF map without PVC.
    cbf_pvc : np.ndarray
        PVC-corrected CBF map (e.g. ExploreASL CBF_PVC2).
    gm_probability : np.ndarray
        Continuous GM probability map [0,1].
    wm_probability : np.ndarray
        Continuous WM probability map [0,1].
    pvc_threshold : float
        Fractional CBF change threshold for flagging PVC as needed.

    Returns
    -------
    PVCQCResult
    """
    gm_sum = np.sum(gm_probability) + 1e-12
    wm_sum = np.sum(wm_probability) + 1e-12

    gm_cbf_raw = float(np.sum(cbf_uncorrected * gm_probability) / gm_sum)
    wm_cbf_raw = float(np.sum(cbf_uncorrected * wm_probability) / wm_sum)
    gm_cbf_pvc = float(np.sum(cbf_pvc * gm_probability) / gm_sum)
    wm_cbf_pvc = float(np.sum(cbf_pvc * wm_probability) / wm_sum)

    ratio_raw = gm_cbf_raw / (wm_cbf_raw + 1e-12)
    ratio_pvc = gm_cbf_pvc / (wm_cbf_pvc + 1e-12)

    pvc_impact_gm = gm_cbf_pvc - gm_cbf_raw
    pvc_impact_ratio = ratio_pvc - ratio_raw

    pvc_needed = abs(pvc_impact_gm / (gm_cbf_raw + 1e-12)) > pvc_threshold

    msg = (
        f"PVC changed GM CBF by {pvc_impact_gm:+.1f} ml/100g/min "
        f"({pvc_impact_gm/(gm_cbf_raw+1e-12)*100:+.1f}%). "
    )
    if pvc_needed:
        msg += "PVC substantially affects CBF — use PVC-corrected map for analysis."
    else:
        msg += "PVC has minor effect — non-corrected map acceptable."

    return PVCQCResult(
        gm_cbf_uncorrected=gm_cbf_raw,
        gm_cbf_pvc=gm_cbf_pvc,
        wm_cbf_uncorrected=wm_cbf_raw,
        wm_cbf_pvc=wm_cbf_pvc,
        ratio_uncorrected=ratio_raw,
        ratio_pvc=ratio_pvc,
        pvc_impact_gm=pvc_impact_gm,
        pvc_impact_ratio=pvc_impact_ratio,
        pvc_needed=pvc_needed,
        message=msg,
    )
