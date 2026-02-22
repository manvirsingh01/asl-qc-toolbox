"""Tests for control-label pattern verification."""

import numpy as np
from asl_qc.preprocessing.control_label import verify_control_label_pattern


def test_valid_pattern(asl_timeseries, asl_context_types, brain_mask):
    result = verify_control_label_pattern(
        asl_timeseries, asl_context_types, brain_mask
    )
    assert result.is_valid
    assert not result.phase_shift_detected
    assert "verified" in result.message.lower() or "pattern" in result.message.lower()


def test_volume_count_mismatch(asl_timeseries, brain_mask):
    wrong_context = ["control", "label"]  # Only 2 types but 40 volumes
    result = verify_control_label_pattern(
        asl_timeseries, wrong_context, brain_mask
    )
    assert not result.is_valid
    assert "mismatch" in result.message.lower()


def test_inverted_pattern(asl_timeseries, brain_mask):
    """Swap control/label in context → should detect phase shift."""
    inverted_types = [
        "label" if t % 2 == 0 else "control"
        for t in range(asl_timeseries.shape[-1])
    ]
    result = verify_control_label_pattern(
        asl_timeseries, inverted_types, brain_mask
    )
    # Should detect desynchronization
    assert not result.is_valid or result.phase_shift_detected


def test_flat_signal(brain_mask):
    """Flat signal → low amplitude → should fail."""
    rng = np.random.default_rng(99)
    flat = rng.normal(1000, 0.001, size=(32, 32, 20, 10))
    types = ["control" if i % 2 == 0 else "label" for i in range(10)]
    result = verify_control_label_pattern(flat, types, brain_mask)
    assert not result.is_valid
    assert "amplitude" in result.message.lower() or "ambiguous" in result.message.lower()
