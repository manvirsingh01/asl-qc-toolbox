"""Tests for spatial Coefficient of Variation (sCoV)."""

import numpy as np
from asl_qc.metrics.scov import compute_scov, compute_regional_scov


def test_uniform_cbf_low_scov(brain_mask):
    """Uniform CBF → low sCoV."""
    cbf = np.ones(brain_mask.shape) * 50
    cbf[~brain_mask] = 0
    scov = compute_scov(cbf, brain_mask)
    assert scov < 0.01


def test_heterogeneous_cbf_high_scov(brain_mask):
    """Heterogeneous CBF → high sCoV."""
    rng = np.random.default_rng(42)
    cbf = rng.normal(50, 40, size=brain_mask.shape)
    cbf[~brain_mask] = 0
    scov = compute_scov(cbf, brain_mask)
    assert scov > 0.3


def test_scov_inf_for_zero_mean(brain_mask):
    """Zero mean → sCoV = inf."""
    cbf = np.zeros(brain_mask.shape)
    scov = compute_scov(cbf, brain_mask)
    assert scov == float("inf")


def test_regional_scov(brain_mask, gm_mask, wm_mask):
    rng = np.random.default_rng(42)
    cbf = rng.normal(50, 10, size=brain_mask.shape)
    cbf[~brain_mask] = 0
    regions = {"GM": gm_mask, "WM": wm_mask}
    result = compute_regional_scov(cbf, regions)
    assert "GM" in result
    assert "WM" in result
    assert all(v >= 0 for v in result.values())
