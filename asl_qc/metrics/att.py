"""
Arterial Transit Time (ATT) based QC metrics.

Evaluates ATT maps produced by ExploreASL (multi-PLD sequences) to
detect transit artifact contamination in CBF maps. Based on:
Woods et al. 2023, JCBFM — Multi-delay ASL quality control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ATTQCResult:
    """QC metrics derived from the arterial transit time map."""

    mean_att_gm: float
    """Mean ATT within gray matter (seconds). Normal: 1.2-1.8s at 3T."""

    att_std_gm: float
    """Standard deviation of ATT in gray matter."""

    transit_artifact_fraction: float
    """Fraction of GM voxels where ATT exceeds the maximum PLD.
    These voxels have uninverted labeling -> CBF underestimated."""

    long_att_fraction: float
    """Fraction of GM voxels with ATT > 2.0s (pathologically long)."""

    att_scov: float
    """Spatial CoV of ATT within GM -- high = heterogeneous transit."""

    is_att_problematic: bool
    """True if transit artifacts likely corrupt CBF map."""

    message: str = ""


def compute_att_qc(
    att_map: np.ndarray,
    gm_mask: np.ndarray,
    pld_values: Optional[List[float]] = None,
    *,
    long_att_threshold_s: float = 2.0,
    problematic_fraction: float = 0.15,
) -> ATTQCResult:
    """
    Compute QC metrics from an ATT map.

    Parameters
    ----------
    att_map : np.ndarray
        3D arterial transit time map in seconds.
        Produced by ExploreASL multi-PLD fitting.
    gm_mask : np.ndarray
        Binary GM mask.
    pld_values : list of float, optional
        Post-labeling delay values used in acquisition (seconds).
        If provided, voxels with ATT > max(PLD) are flagged as artifacts.
    long_att_threshold_s : float
        ATT above this value considered pathologically long.
    problematic_fraction : float
        Fraction threshold above which ATT artifacts are flagged.

    Returns
    -------
    ATTQCResult
    """
    gm = gm_mask.astype(bool)
    att_gm = att_map[gm]

    if len(att_gm) < 10:
        return ATTQCResult(
            mean_att_gm=0.0, att_std_gm=0.0,
            transit_artifact_fraction=0.0,
            long_att_fraction=0.0,
            att_scov=0.0,
            is_att_problematic=False,
            message="Insufficient GM voxels for ATT QC.",
        )

    mean_att = float(np.mean(att_gm))
    std_att = float(np.std(att_gm, ddof=1))
    mu = mean_att if abs(mean_att) > 1e-12 else 1e-12
    att_scov = std_att / abs(mu)

    # Transit artifact: ATT > max PLD -> labeled blood not yet in tissue
    if pld_values:
        max_pld = max(pld_values)
        transit_frac = float(np.mean(att_gm > max_pld))
    else:
        # Conservative estimate: ATT > 2.5s exceeds most PCASL protocols
        transit_frac = float(np.mean(att_gm > 2.5))

    long_frac = float(np.mean(att_gm > long_att_threshold_s))
    is_problematic = transit_frac > problematic_fraction

    messages = [f"Mean GM ATT = {mean_att:.3f}s."]
    if is_problematic:
        messages.append(
            f"WARNING: {transit_frac*100:.1f}% of GM voxels have "
            "ATT exceeding PLD — CBF underestimated in these regions."
        )
    if long_frac > 0.20:
        messages.append(
            f"WARNING: {long_frac*100:.1f}% of GM voxels have "
            f"ATT > {long_att_threshold_s}s — consider delayed PLD protocol."
        )

    return ATTQCResult(
        mean_att_gm=mean_att,
        att_std_gm=std_att,
        transit_artifact_fraction=transit_frac,
        long_att_fraction=long_frac,
        att_scov=att_scov,
        is_att_problematic=is_problematic,
        message=" ".join(messages),
    )
