"""Tests for tSNR and RMS difference."""

import numpy as np
from asl_qc.metrics.snr import compute_temporal_snr, compute_rms_difference


def test_tsnr_stable_signal(brain_mask):
    """Stable signal → high tSNR."""
    rng = np.random.default_rng(42)
    data = rng.normal(1000, 10, size=(*brain_mask.shape, 20))
    tsnr_map, global_tsnr, median_tsnr = compute_temporal_snr(data, brain_mask)
    assert global_tsnr > 50  # 1000/10 ≈ 100
    assert tsnr_map.shape == brain_mask.shape


def test_tsnr_noisy_signal(brain_mask):
    """Noisy signal → low tSNR."""
    rng = np.random.default_rng(42)
    data = rng.normal(10, 100, size=(*brain_mask.shape, 20))
    _, global_tsnr, _ = compute_temporal_snr(data, brain_mask)
    assert global_tsnr < 5


def test_rms_difference_identical(brain_mask):
    """Identical volumes → RMS = 0."""
    vol = np.ones((*brain_mask.shape, 5))
    rms = compute_rms_difference(vol, vol, brain_mask)
    assert rms == 0.0


def test_rms_difference_known(brain_mask):
    """Known offset → predictable RMS."""
    ctrl = np.ones((*brain_mask.shape, 3)) * 100
    lbl = np.ones((*brain_mask.shape, 3)) * 90
    rms = compute_rms_difference(ctrl, lbl, brain_mask)
    assert rms > 0
    np.testing.assert_allclose(rms, 10.0, atol=0.1)
