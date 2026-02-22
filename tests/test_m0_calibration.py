"""Tests for M0 calibration assessment."""

import numpy as np
from asl_qc.preprocessing.m0_calibration import assess_m0


def test_dedicated_m0_normal(brain_mask):
    rng = np.random.default_rng(42)
    m0 = rng.normal(5000, 200, size=brain_mask.shape)
    m0[~brain_mask] = 0
    result = assess_m0(m0, brain_mask, has_dedicated_m0=True)
    assert result.source == "dedicated"
    assert result.is_usable
    assert not result.artifact_flagged
    assert result.smoothed_m0 is not None


def test_m0_with_artifact(brain_mask):
    rng = np.random.default_rng(42)
    m0 = rng.normal(5000, 200, size=brain_mask.shape)
    m0[~brain_mask] = 0
    # Add massive flare
    m0[10, 10, 10] = 500000
    result = assess_m0(m0, brain_mask, has_dedicated_m0=True, max_median_ratio_threshold=10)
    assert result.artifact_flagged
    assert not result.is_usable


def test_m0_absent_with_bgsup(brain_mask):
    result = assess_m0(
        None, brain_mask,
        has_dedicated_m0=False,
        background_suppression=True,
    )
    assert result.source == "absent"
    assert not result.is_usable
    assert "relative" in result.message.lower()


def test_m0_from_control_average(brain_mask):
    rng = np.random.default_rng(42)
    ctrl_avg = rng.normal(5000, 200, size=brain_mask.shape)
    ctrl_avg[~brain_mask] = 0
    result = assess_m0(
        None, brain_mask,
        has_dedicated_m0=False,
        background_suppression=False,
        control_average=ctrl_avg,
    )
    assert result.source == "control_average"
    assert result.is_usable
