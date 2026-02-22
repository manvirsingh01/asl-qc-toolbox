"""Tests for motion tracking (FD and DVARS)."""

import numpy as np
from asl_qc.preprocessing.motion import (
    compute_framewise_displacement,
    compute_dvars,
    summarize_motion,
)


def test_fd_zero_motion():
    """Zero motion → FD = 0."""
    params = np.zeros((10, 6))
    fd = compute_framewise_displacement(params)
    assert len(fd) == 9
    np.testing.assert_allclose(fd, 0.0)


def test_fd_known_displacement():
    """Known translation → expected FD."""
    params = np.zeros((3, 6))
    params[1, 0] = 1.0  # +1mm in X at second time point
    fd = compute_framewise_displacement(params)
    assert len(fd) == 2
    assert fd[0] == pytest.approx(1.0)


def test_fd_rotation():
    """Rotation converted to arc length."""
    params = np.zeros((3, 6))
    params[1, 3] = 0.01  # Small rotation in rad
    fd = compute_framewise_displacement(params, radius_mm=50)
    assert fd[0] == pytest.approx(0.5, abs=0.01)  # 0.01 rad × 50mm


def test_dvars_stable(asl_timeseries, brain_mask):
    dvars = compute_dvars(asl_timeseries, brain_mask)
    assert len(dvars) == asl_timeseries.shape[-1] - 1
    assert np.all(dvars >= 0)


def test_summarize_low_motion(asl_timeseries, brain_mask):
    # Use truly small motion: 0.001mm translations, 0.00001 rad rotations
    rng = np.random.default_rng(42)
    small_params = np.zeros((40, 6))
    small_params[:, :3] = rng.normal(0, 0.001, size=(40, 3))
    small_params[:, 3:] = rng.normal(0, 0.00001, size=(40, 3))
    fd = compute_framewise_displacement(small_params)
    dvars = compute_dvars(asl_timeseries, brain_mask)
    ms = summarize_motion(fd, dvars, fd_spike_threshold=0.5)
    assert ms.mean_fd < 0.5
    assert ms.n_fd_spikes == 0


def test_summarize_high_motion(high_motion_params, asl_timeseries, brain_mask):
    fd = compute_framewise_displacement(high_motion_params)
    dvars = compute_dvars(asl_timeseries, brain_mask)
    ms = summarize_motion(fd, dvars, fd_spike_threshold=0.5)
    assert ms.max_fd > 1.0
    assert ms.n_fd_spikes >= 1


import pytest
