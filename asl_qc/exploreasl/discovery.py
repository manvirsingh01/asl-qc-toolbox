"""
ExploreASL output file auto-discovery.

Automatically locates all required input files from an ExploreASL
derivatives directory, eliminating the need to specify paths manually.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class ExploreASLPaths:
    """All file paths discovered from an ExploreASL subject directory."""

    # Core CBF maps
    cbf_map: Optional[Path] = None
    cbf_pvc_map: Optional[Path] = None
    att_map: Optional[Path] = None

    # Tissue masks (from T1w processing)
    gm_mask: Optional[Path] = None
    wm_mask: Optional[Path] = None
    csf_mask: Optional[Path] = None
    brain_mask: Optional[Path] = None
    wmh_mask: Optional[Path] = None

    # ASL timeseries and calibration
    asl_timeseries: Optional[Path] = None
    m0_image: Optional[Path] = None
    motion_params: Optional[Path] = None

    # BIDS metadata
    aslcontext_tsv: Optional[Path] = None
    asl_json: Optional[Path] = None

    # ExploreASL QC
    xasl_qc_json: Optional[Path] = None

    # Population space
    cbf_mni: Optional[Path] = None

    # Missing files list
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)


def discover_exploreasl_outputs(
    subject_dir: str | Path,
    subject_id: Optional[str] = None,
    session_label: str = "ASL_1",
) -> ExploreASLPaths:
    """
    Auto-discover all ExploreASL output files for a subject.

    Parameters
    ----------
    subject_dir : path-like
        Path to ExploreASL derivatives subject directory.
        e.g. /derivatives/ExploreASL/sub-01
    subject_id : str, optional
        Subject identifier. Inferred from directory name if not provided.
    session_label : str
        ASL session label used by ExploreASL (default: ASL_1).

    Returns
    -------
    ExploreASLPaths
        Dataclass with all discovered file paths.
    """
    subject_dir = Path(subject_dir)
    if subject_id is None:
        subject_id = subject_dir.name

    paths = ExploreASLPaths()

    # ASL session directory
    asl_dir = subject_dir / f"{session_label}"
    t1w_dir = subject_dir / "T1w"

    # ── CBF maps ──────────────────────────────────────────────────────────
    cbf_candidates = [
        asl_dir / f"{subject_id}_{session_label}_CBF.nii.gz",
        asl_dir / "CBF.nii.gz",
        asl_dir / "mean_control.nii.gz",
    ]
    for c in cbf_candidates:
        if c.exists():
            paths.cbf_map = c
            break
    if paths.cbf_map is None:
        paths.missing_required.append("cbf_map")

    # PVC-corrected CBF
    pvc_path = asl_dir / f"{subject_id}_{session_label}_CBF_PVC2.nii.gz"
    if pvc_path.exists():
        paths.cbf_pvc_map = pvc_path
    else:
        paths.missing_optional.append("cbf_pvc_map")

    # ATT map (multi-PLD only)
    att_path = asl_dir / f"{subject_id}_{session_label}_ATT.nii.gz"
    if att_path.exists():
        paths.att_map = att_path
    else:
        paths.missing_optional.append("att_map")

    # ── Tissue masks ──────────────────────────────────────────────────────
    mask_map = {
        "gm_mask":    [t1w_dir / f"{subject_id}_T1w_pGM.nii.gz",
                       t1w_dir / "pGM.nii.gz"],
        "wm_mask":    [t1w_dir / f"{subject_id}_T1w_pWM.nii.gz",
                       t1w_dir / "pWM.nii.gz"],
        "csf_mask":   [t1w_dir / f"{subject_id}_T1w_pCSF.nii.gz",
                       t1w_dir / "pCSF.nii.gz"],
        "brain_mask": [t1w_dir / f"{subject_id}_T1w_BrainMask.nii.gz",
                       t1w_dir / "BrainMask.nii.gz"],
        "wmh_mask":   [t1w_dir / f"{subject_id}_T1w_WMH_SEGM.nii.gz",
                       t1w_dir / "WMH_SEGM.nii.gz"],
    }
    required_masks = {"gm_mask", "wm_mask", "brain_mask"}

    for attr, candidates in mask_map.items():
        found = False
        for c in candidates:
            if c.exists():
                setattr(paths, attr, c)
                found = True
                break
        if not found:
            if attr in required_masks:
                paths.missing_required.append(attr)
            else:
                paths.missing_optional.append(attr)

    # ── ASL timeseries ────────────────────────────────────────────────────
    ts_candidates = [
        asl_dir / f"{subject_id}_{session_label}.nii.gz",
        asl_dir / "ASL4D.nii.gz",
    ]
    for c in ts_candidates:
        if c.exists():
            paths.asl_timeseries = c
            break
    if paths.asl_timeseries is None:
        paths.missing_optional.append("asl_timeseries")

    # ── M0 image ──────────────────────────────────────────────────────────
    m0_candidates = [
        asl_dir / f"{subject_id}_{session_label}_M0.nii.gz",
        asl_dir / "M0.nii.gz",
    ]
    for c in m0_candidates:
        if c.exists():
            paths.m0_image = c
            break
    if paths.m0_image is None:
        paths.missing_optional.append("m0_image")

    # ── Motion parameters ─────────────────────────────────────────────────
    motion_candidates = [
        asl_dir / f"rp_{subject_id}_{session_label}.txt",
        asl_dir / "rp_ASL4D.txt",
    ]
    for c in motion_candidates:
        if c.exists():
            paths.motion_params = c
            break
    if paths.motion_params is None:
        paths.missing_optional.append("motion_params")

    # ── BIDS metadata ─────────────────────────────────────────────────────
    bids_candidates = {
        "aslcontext_tsv": [
            asl_dir / f"{subject_id}_{session_label}_aslcontext.tsv",
            asl_dir / "aslcontext.tsv",
        ],
        "asl_json": [
            asl_dir / f"{subject_id}_{session_label}.json",
            asl_dir / "ASL4D.json",
        ],
    }
    for attr, candidates in bids_candidates.items():
        for c in candidates:
            if c.exists():
                setattr(paths, attr, c)
                break

    # ── ExploreASL QC JSON ────────────────────────────────────────────────
    xasl_qc = asl_dir / f"{subject_id}_{session_label}_qc_collection.json"
    if xasl_qc.exists():
        paths.xasl_qc_json = xasl_qc

    # ── Population space CBF ──────────────────────────────────────────────
    pop_dir = subject_dir.parent / "Population"
    pop_cbf = pop_dir / f"{subject_id}_{session_label}_CBF.nii.gz"
    if pop_cbf.exists():
        paths.cbf_mni = pop_cbf

    return paths


def load_exploreasl_qc(xasl_qc_path: Path) -> Dict:
    """
    Load ExploreASL's own QC JSON collection.

    ExploreASL saves basic QC flags in a JSON file per subject.
    These can complement (not replace) the metrics computed by this toolbox.

    Returns
    -------
    dict
        ExploreASL QC values, or empty dict if file absent/unreadable.
    """
    if xasl_qc_path is None or not xasl_qc_path.exists():
        return {}
    try:
        with open(xasl_qc_path, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}
