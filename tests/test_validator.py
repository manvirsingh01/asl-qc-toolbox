"""Tests for ExploreASL output validator."""

from asl_qc.exploreasl.discovery import discover_exploreasl_outputs
from asl_qc.exploreasl.validator import validate_exploreasl_outputs


def test_valid_outputs(tmp_path):
    """Complete ExploreASL outputs pass validation."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    (asl_dir / "sub-01_ASL_1_CBF.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pGM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pWM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_BrainMask.nii.gz").write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    result = validate_exploreasl_outputs(paths)
    assert result.is_valid
    assert len(result.errors) == 0


def test_invalid_missing_cbf(tmp_path):
    """Missing CBF map fails validation."""
    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    result = validate_exploreasl_outputs(paths)
    assert not result.is_valid
    assert any("cbf_map" in e for e in result.errors)


def test_warnings_for_optional(tmp_path):
    """Optional missing files produce warnings, not errors."""
    asl_dir = tmp_path / "ASL_1"
    t1w_dir = tmp_path / "T1w"
    asl_dir.mkdir()
    t1w_dir.mkdir()

    (asl_dir / "sub-01_ASL_1_CBF.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pGM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_pWM.nii.gz").write_bytes(b"fake")
    (t1w_dir / "sub-01_T1w_BrainMask.nii.gz").write_bytes(b"fake")

    paths = discover_exploreasl_outputs(tmp_path, "sub-01")
    result = validate_exploreasl_outputs(paths)
    assert result.is_valid
    assert len(result.warnings) > 0
