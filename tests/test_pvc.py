"""Tests for PVC impact assessment."""

import numpy as np

from asl_qc.pvc import compute_pvc_qc


def test_pvc_no_impact(brain_mask):
    """Identical maps produce zero PVC impact."""
    rng = np.random.default_rng(42)
    cbf = rng.normal(50, 5, brain_mask.shape)
    gm_prob = brain_mask.astype(float) * 0.8
    wm_prob = brain_mask.astype(float) * 0.2
    result = compute_pvc_qc(cbf, cbf, gm_prob, wm_prob)
    assert abs(result.pvc_impact_gm) < 0.01
    assert not result.pvc_needed


def test_pvc_large_impact(brain_mask):
    """Large PVC correction flagged as needed."""
    rng = np.random.default_rng(42)
    cbf_raw = rng.normal(40, 5, brain_mask.shape)
    cbf_pvc = rng.normal(65, 5, brain_mask.shape)
    gm_prob = brain_mask.astype(float)
    wm_prob = 1.0 - gm_prob
    result = compute_pvc_qc(cbf_raw, cbf_pvc, gm_prob, wm_prob)
    assert result.pvc_needed


def test_pvc_ratios(brain_mask):
    """GM/WM ratios are computed before and after PVC."""
    rng = np.random.default_rng(42)
    cbf_raw = np.ones(brain_mask.shape) * 40
    cbf_pvc = np.ones(brain_mask.shape) * 50
    gm_prob = brain_mask.astype(float) * 0.7
    wm_prob = brain_mask.astype(float) * 0.3
    result = compute_pvc_qc(cbf_raw, cbf_pvc, gm_prob, wm_prob)
    assert result.ratio_uncorrected > 0
    assert result.ratio_pvc > 0


def test_pvc_message_content(brain_mask):
    """Result message contains PVC impact info."""
    cbf = np.ones(brain_mask.shape) * 50
    gm_prob = brain_mask.astype(float) * 0.8
    wm_prob = brain_mask.astype(float) * 0.2
    result = compute_pvc_qc(cbf, cbf, gm_prob, wm_prob)
    assert "PVC" in result.message


def test_pvc_custom_threshold(brain_mask):
    """Custom PVC threshold is respected."""
    rng = np.random.default_rng(42)
    cbf_raw = rng.normal(50, 5, brain_mask.shape)
    cbf_pvc = rng.normal(55, 5, brain_mask.shape)  # ~10% change
    gm_prob = brain_mask.astype(float)
    wm_prob = np.zeros_like(gm_prob)
    # Very low threshold -> should flag as needed
    result = compute_pvc_qc(cbf_raw, cbf_pvc, gm_prob, wm_prob, pvc_threshold=0.01)
    assert result.pvc_needed
