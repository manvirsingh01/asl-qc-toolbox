"""Tests for core QC metrics on raw BIDS ASL data."""

import numpy as np
import pytest

from asl_qc.metrics.core import CoreQCResult, compute_core_qc


# Spatial dimensions (match conftest.py)
NX, NY, NZ = 32, 32, 20
N_VOLS = 40


# ---------------------------------------------------------------------------
# Volume structure
# ---------------------------------------------------------------------------

def test_volume_count_match(brain_mask, asl_timeseries, asl_context_types):
    """NIfTI and aslcontext volume counts agree."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, volume_types=asl_context_types
    )
    assert result.volume_count_match is True
    assert result.n_volumes_nifti == N_VOLS
    assert result.n_volumes_context == N_VOLS


def test_volume_count_mismatch(brain_mask, asl_timeseries):
    """Mismatch between NIfTI and aslcontext volume counts triggers warning."""
    short_context = ["control", "label"] * 10  # 20 instead of 40
    result = compute_core_qc(
        asl_timeseries, brain_mask, volume_types=short_context
    )
    assert result.volume_count_match is False
    assert any("mismatch" in w.lower() for w in result.warnings)


def test_control_label_balanced(brain_mask, asl_timeseries, asl_context_types):
    """Equal control/label counts → balanced."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, volume_types=asl_context_types
    )
    assert result.n_controls == 20
    assert result.n_labels == 20
    assert result.control_label_balanced is True


def test_control_label_imbalanced(brain_mask, asl_timeseries):
    """Unequal control/label counts → warning."""
    unbalanced = ["control"] * 25 + ["label"] * 15
    result = compute_core_qc(
        asl_timeseries, brain_mask, volume_types=unbalanced
    )
    assert result.control_label_balanced is False
    assert any("imbalance" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Signal statistics
# ---------------------------------------------------------------------------

def test_signal_stats_reasonable(brain_mask, asl_timeseries):
    """Synthetic ASL data yields positive signal and tSNR."""
    result = compute_core_qc(asl_timeseries, brain_mask)
    assert result.mean_signal > 0
    assert result.temporal_std_mean > 0
    assert result.global_tsnr > 0


def test_low_signal_warns(brain_mask):
    """Near-zero data → low signal warning."""
    data = np.zeros((NX, NY, NZ, 10), dtype=np.float64)
    result = compute_core_qc(data, brain_mask)
    assert result.mean_signal < 1.0
    assert any("low mean signal" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def test_nan_detection(brain_mask):
    """NaN in timeseries is detected."""
    rng = np.random.default_rng(99)
    data = rng.normal(1000, 50, size=(NX, NY, NZ, 10))
    data[5, 5, 5, 0] = np.nan
    result = compute_core_qc(data, brain_mask)
    assert result.has_nan is True
    assert result.nan_voxel_count >= 1
    assert any("nan" in w.lower() for w in result.warnings)


def test_inf_detection(brain_mask):
    """Inf in timeseries is detected."""
    rng = np.random.default_rng(99)
    data = rng.normal(1000, 50, size=(NX, NY, NZ, 10))
    data[5, 5, 5, 0] = np.inf
    result = compute_core_qc(data, brain_mask)
    assert result.has_inf is True
    assert result.inf_voxel_count >= 1
    assert any("inf" in w.lower() for w in result.warnings)


def test_clean_data_no_nan_inf(brain_mask, asl_timeseries):
    """Clean data → no NaN/Inf flags."""
    result = compute_core_qc(asl_timeseries, brain_mask)
    assert result.has_nan is False
    assert result.has_inf is False


# ---------------------------------------------------------------------------
# Control–label difference
# ---------------------------------------------------------------------------

def test_control_label_diff_positive(brain_mask, asl_timeseries, asl_context_types):
    """Alternating control/label data shows a positive difference."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, volume_types=asl_context_types
    )
    assert result.mean_control_label_diff > 0


# ---------------------------------------------------------------------------
# M0 signal
# ---------------------------------------------------------------------------

def test_m0_signal_detected(brain_mask):
    """M0 volumes are identified and their signal is reported."""
    rng = np.random.default_rng(42)
    # 5 vols: m0, control, label, control, label
    data = rng.normal(1000, 50, size=(NX, NY, NZ, 5))
    data[..., 0] *= 3  # M0 much brighter
    vol_types = ["m0scan", "control", "label", "control", "label"]
    result = compute_core_qc(data, brain_mask, volume_types=vol_types)
    assert result.m0_mean_signal is not None
    assert result.m0_asl_ratio is not None
    assert result.m0_asl_ratio > 1.0


# ---------------------------------------------------------------------------
# NIfTI header / voxel sizes
# ---------------------------------------------------------------------------

def test_voxel_size_warnings(brain_mask, asl_timeseries):
    """Unusual voxel size triggers a warning."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, voxel_sizes=(0.1, 3.0, 3.0)
    )
    assert any("unusual voxel size" in w.lower() for w in result.warnings)


def test_normal_voxel_sizes_no_warning(brain_mask, asl_timeseries):
    """Typical voxel sizes produce no warnings."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, voxel_sizes=(3.0, 3.0, 5.0)
    )
    assert not any("voxel size" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------

def test_missing_bids_fields_warning(brain_mask, asl_timeseries):
    """Missing BIDS fields are reported."""
    result = compute_core_qc(
        asl_timeseries, brain_mask,
        missing_bids_fields=["PostLabelingDelay", "ArterialSpinLabelingType"],
    )
    assert len(result.missing_bids_fields) == 2
    assert any("missing required bids" in w.lower() for w in result.warnings)


def test_no_missing_bids_fields(brain_mask, asl_timeseries):
    """No missing fields → no metadata warning."""
    result = compute_core_qc(
        asl_timeseries, brain_mask, missing_bids_fields=[]
    )
    assert len(result.missing_bids_fields) == 0
    assert not any("bids" in w.lower() for w in result.warnings)
