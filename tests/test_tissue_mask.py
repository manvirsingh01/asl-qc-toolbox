"""Tests for tissue mask quality metrics (Dice, Jaccard, GM/WM ratio)."""

import numpy as np
from asl_qc.metrics.tissue_mask import (
    compute_dice,
    compute_jaccard,
    compute_gm_wm_ratio,
)


def test_dice_identical():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    assert compute_dice(mask, mask) == 1.0


def test_dice_disjoint():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[:5] = True
    b = np.zeros((10, 10, 10), dtype=bool)
    b[5:] = True
    assert compute_dice(a, b) == 0.0


def test_dice_partial():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[:6] = True
    b = np.zeros((10, 10, 10), dtype=bool)
    b[4:] = True
    dice = compute_dice(a, b)
    assert 0 < dice < 1


def test_jaccard_identical():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    assert compute_jaccard(mask, mask) == 1.0


def test_jaccard_disjoint():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[:5] = True
    b = np.zeros((10, 10, 10), dtype=bool)
    b[5:] = True
    assert compute_jaccard(a, b) == 0.0


def test_gm_wm_ratio_expected(gm_mask, wm_mask):
    """GM=60, WM=25 → ratio ≈ 2.4."""
    cbf = np.zeros(gm_mask.shape)
    cbf[gm_mask] = 60
    cbf[wm_mask] = 25
    ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf, gm_mask, wm_mask)
    assert 2.0 <= ratio <= 3.0
    assert mean_gm == 60.0
    assert mean_wm == 25.0
