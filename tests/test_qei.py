"""Tests for Quality Evaluation Index (QEI)."""

import numpy as np
from asl_qc.metrics.qei import compute_qei


def test_perfect_cbf_high_qei(gm_mask, wm_mask):
    """Perfect CBF map (GM=60, WM=25) → high QEI."""
    rng = np.random.default_rng(42)
    cbf = np.zeros(gm_mask.shape, dtype=np.float64)
    cbf[gm_mask] = 60.0 + rng.normal(0, 2, size=int(np.sum(gm_mask)))
    cbf[wm_mask] = 25.0 + rng.normal(0, 1, size=int(np.sum(wm_mask)))

    result = compute_qei(cbf, gm_mask, wm_mask)
    assert result.qei > 0.5
    assert result.structural_similarity > 0.3
    assert result.negative_gm_fraction < 0.05


def test_noisy_cbf_low_qei(noisy_cbf_map, gm_mask, wm_mask):
    """Noisy CBF with many negatives → low QEI."""
    result = compute_qei(noisy_cbf_map, gm_mask, wm_mask)
    assert result.qei < 0.5
    assert result.negative_gm_fraction > 0.1


def test_qei_between_0_and_1(good_cbf_map, gm_mask, wm_mask):
    result = compute_qei(good_cbf_map, gm_mask, wm_mask)
    assert 0.0 <= result.qei <= 1.0


def test_qei_zero_for_empty_masks():
    """Empty masks → QEI = 0."""
    cbf = np.zeros((10, 10, 10))
    gm = np.zeros((10, 10, 10), dtype=bool)
    wm = np.zeros((10, 10, 10), dtype=bool)
    result = compute_qei(cbf, gm, wm)
    assert result.qei == 0.0
