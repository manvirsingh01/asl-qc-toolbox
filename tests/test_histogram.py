"""Tests for histogram analysis."""

import numpy as np
from asl_qc.metrics.histogram import analyze_histogram


def test_gaussian_histogram(brain_mask):
    """Gaussian-distributed CBF → near-zero skewness and kurtosis."""
    rng = np.random.default_rng(42)
    cbf = rng.normal(50, 10, size=brain_mask.shape)
    result = analyze_histogram(cbf, brain_mask)
    assert abs(result.skewness) < 0.5
    assert abs(result.kurtosis) < 1.5
    assert result.percentile_5 < result.percentile_95
    assert result.n_voxels > 0


def test_skewed_histogram(brain_mask):
    """Right-skewed CBF → positive skewness."""
    rng = np.random.default_rng(42)
    cbf = rng.exponential(20, size=brain_mask.shape)
    result = analyze_histogram(cbf, brain_mask)
    assert result.skewness > 0.5


def test_empty_mask():
    """Empty mask → zeroed results."""
    cbf = np.ones((10, 10, 10))
    mask = np.zeros((10, 10, 10), dtype=bool)
    result = analyze_histogram(cbf, mask)
    assert result.n_voxels == 0
    assert result.mean == 0.0
