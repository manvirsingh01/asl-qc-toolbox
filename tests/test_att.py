"""Tests for ATT-based QC metrics."""

import numpy as np

from asl_qc.metrics.att import compute_att_qc


def test_att_qc_normal(brain_mask):
    """Normal ATT (1.4s) produces no problems."""
    att = np.ones(brain_mask.shape) * 1.4
    gm = brain_mask
    result = compute_att_qc(att, gm, pld_values=[1.8])
    assert not result.is_att_problematic
    assert result.transit_artifact_fraction < 0.01
    assert abs(result.mean_att_gm - 1.4) < 0.01


def test_att_qc_artifact(brain_mask):
    """ATT > max PLD in most voxels is flagged as problematic."""
    att = np.ones(brain_mask.shape) * 3.0
    result = compute_att_qc(att, brain_mask, pld_values=[1.8, 2.5])
    assert result.is_att_problematic
    assert result.transit_artifact_fraction > 0.5


def test_att_qc_no_pld(brain_mask):
    """Without PLD values, uses conservative 2.5s threshold."""
    att = np.ones(brain_mask.shape) * 1.4
    result = compute_att_qc(att, brain_mask)
    assert not result.is_att_problematic
    assert result.transit_artifact_fraction < 0.01


def test_att_qc_long_fraction(brain_mask):
    """Long ATT fraction computed correctly."""
    att = np.ones(brain_mask.shape) * 2.5  # > default 2.0s threshold
    result = compute_att_qc(att, brain_mask)
    assert result.long_att_fraction > 0.99


def test_att_qc_insufficient_voxels():
    """Insufficient GM voxels returns safe defaults."""
    att = np.zeros((5, 5, 5))
    gm = np.zeros((5, 5, 5), dtype=bool)
    gm[0, 0, 0] = True  # Only 1 voxel
    result = compute_att_qc(att, gm)
    assert not result.is_att_problematic
    assert result.message == "Insufficient GM voxels for ATT QC."


def test_att_scov(brain_mask):
    """ATT sCoV is computed correctly for uniform ATT."""
    rng = np.random.default_rng(42)
    att = np.ones(brain_mask.shape) * 1.5 + rng.normal(0, 0.1, brain_mask.shape)
    result = compute_att_qc(att, brain_mask)
    # For nearly uniform ATT, sCoV should be small
    assert result.att_scov < 0.2
