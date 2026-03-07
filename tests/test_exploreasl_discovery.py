"""Tests for ExploreASL output auto-discovery."""

from pathlib import Path

from asl_qc.exploreasl.discovery import (
    discover_exploreasl_outputs,
    load_exploreasl_qc,
)


def test_discovery_finds_cbf(tmp_path):
    """Auto-discovery finds CBF map in standard ExploreASL layout."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    cbf = asl_dir / "sub-01_ASL_1_CBF.nii.gz"
    cbf.write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert paths.cbf_map == cbf
    assert "cbf_map" not in paths.missing_required


def test_discovery_missing_required(tmp_path):
    """Missing CBF map reported in missing_required."""
    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert "cbf_map" in paths.missing_required


def test_discovery_finds_tissue_masks(tmp_path):
    """Auto-discovery finds GM, WM, brain masks."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    # Create CBF
    (asl_dir / "sub-01_ASL_1_CBF.nii.gz").write_bytes(b"fake")
    # Create tissue masks
    (t1w_dir / "sub-01_T1w_pGM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pWM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_BrainMask.nii.gz").write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert paths.gm_mask is not None
    assert paths.wm_mask is not None
    assert paths.brain_mask is not None
    assert len(paths.missing_required) == 0


def test_discovery_optional_files(tmp_path):
    """Optional files (PVC, ATT, WMH) reported as missing_optional."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    (asl_dir / "sub-01_ASL_1_CBF.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pGM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pWM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_BrainMask.nii.gz").write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert "cbf_pvc_map" in paths.missing_optional
    assert "att_map" in paths.missing_optional
    assert "wmh_mask" in paths.missing_optional


def test_discovery_fallback_filenames(tmp_path):
    """Fallback filenames (CBF.nii.gz, pGM.nii.gz) are discovered."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    (asl_dir / "CBF.nii.gz").write_bytes(b"fake")
    (t1w_dir / "pGM.nii.gz").write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert paths.cbf_map == asl_dir / "CBF.nii.gz"
    assert paths.gm_mask == t1w_dir / "pGM.nii.gz"


def test_discovery_infers_subject_id(tmp_path):
    """Subject ID inferred from directory name."""
    sub_dir = tmp_path / "sub-042"
    sub_dir.mkdir()
    (sub_dir / "ASL_1").mkdir()
    (sub_dir / "T1w").mkdir()

    paths = discover_exploreasl_outputs(sub_dir)
    # Should use sub-042 as subject_id (affects filename patterns)
    assert "cbf_map" in paths.missing_required


def test_discovery_motion_params(tmp_path):
    """Motion parameters file discovered."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    mp = asl_dir / "rp_sub-01_ASL_1.txt"
    mp.write_text("0 0 0 0 0 0\n")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    assert paths.motion_params == mp


def test_load_exploreasl_qc_valid(tmp_path):
    """Load a valid ExploreASL QC JSON."""
    import json
    qc_path = tmp_path / "qc.json"
    qc_data = {"SpatialCoV": 0.3, "MeanGMCBF": 55.2}
    qc_path.write_text(json.dumps(qc_data))

    result = load_exploreasl_qc(qc_path)
    assert result["SpatialCoV"] == 0.3
    assert result["MeanGMCBF"] == 55.2


def test_load_exploreasl_qc_missing():
    """Missing QC JSON returns empty dict."""
    from pathlib import Path
    result = load_exploreasl_qc(Path("/nonexistent/path.json"))
    assert result == {}


def test_load_exploreasl_qc_none():
    """None path returns empty dict."""
    result = load_exploreasl_qc(None)
    assert result == {}
