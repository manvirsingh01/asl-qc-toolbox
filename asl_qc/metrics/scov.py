"""
Spatial Coefficient of Variation (sCoV) for ASL CBF maps.

sCoV = σ(CBF) / μ(CBF) within a region of interest, used to detect
arterial transit time (ATT) artifacts characterised by intense focal
hyperintensity and deflated tissue perfusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SCoVResult:
    """Container for sCoV results."""

    global_scov: float
    """sCoV computed over the entire ROI mask."""
    regional_scov: Optional[Dict[str, float]] = None
    """Per-region sCoV if an atlas was provided."""


def compute_scov(cbf_map: np.ndarray, roi_mask: np.ndarray) -> float:
    """Compute the spatial Coefficient of Variation within *roi_mask*.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    roi_mask : np.ndarray
        Boolean 3-D mask (e.g. GM mask).

    Returns
    -------
    float
        sCoV = σ / μ  (returns ``inf`` if μ ≈ 0).
    """
    mask = roi_mask.astype(bool)
    vals = cbf_map[mask]
    if len(vals) < 2:
        return float("inf")
    mu = np.mean(vals)
    if abs(mu) < 1e-12:
        return float("inf")
    return float(np.std(vals, ddof=1) / abs(mu))


def compute_regional_scov(
    cbf_map: np.ndarray,
    atlas_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute sCoV for each region in an atlas.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    atlas_masks : dict
        Mapping from region name → boolean 3-D mask.

    Returns
    -------
    dict
        Region name → sCoV value.
    """
    return {name: compute_scov(cbf_map, mask) for name, mask in atlas_masks.items()}


def compute_scov_result(
    cbf_map: np.ndarray,
    roi_mask: np.ndarray,
    atlas_masks: Optional[Dict[str, np.ndarray]] = None,
) -> SCoVResult:
    """Full sCoV analysis.

    Parameters
    ----------
    cbf_map : np.ndarray
        3-D CBF map.
    roi_mask : np.ndarray
        Primary ROI mask (e.g. GM).
    atlas_masks : dict, optional
        Per-region masks for regional analysis.

    Returns
    -------
    SCoVResult
    """
    global_scov = compute_scov(cbf_map, roi_mask)
    regional = None
    if atlas_masks is not None:
        regional = compute_regional_scov(cbf_map, atlas_masks)
    return SCoVResult(global_scov=global_scov, regional_scov=regional)
