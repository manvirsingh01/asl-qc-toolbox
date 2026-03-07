"""Tests for ENIGMA-ASL normative database and adaptive thresholds."""

from asl_qc.thresholds.normative import (
    compute_normative_zscores,
    get_acquisition_key,
    get_adaptive_thresholds,
)


def test_acquisition_key_rounding():
    """Field strength is correctly binned."""
    assert get_acquisition_key(2.89, "PCASL") == "3T_PCASL"
    assert get_acquisition_key(1.5, "PCASL") == "1.5T_PCASL"
    assert get_acquisition_key(7.0, "PCASL") == "7T_PCASL"
    assert get_acquisition_key(3.0, "PASL") == "3T_PASL"


def test_acquisition_key_fallback():
    """Unknown labeling type falls back to 3T_PCASL."""
    assert get_acquisition_key(3.0, "UNKNOWN") == "3T_PCASL"


def test_normative_zscores_normal():
    """Metrics near normative mean produce small z-scores."""
    metrics = {"qei": 0.732, "scov_gm": 0.283, "mean_fd": 0.218}
    result = compute_normative_zscores(metrics, 3.0, "PCASL")
    assert all(abs(z) < 1.0 for z in result.z_scores.values())
    assert result.normative_verdict == "NORMAL"


def test_normative_zscores_outlier():
    """Metrics far from normative mean produce OUTLIER verdict."""
    metrics = {
        "qei": 0.2,       # very low
        "scov_gm": 0.8,   # very high
        "mean_fd": 1.5,    # way too much motion
        "gm_wm_ratio": 0.5,  # abnormally low
    }
    result = compute_normative_zscores(metrics, 3.0, "PCASL")
    assert result.n_outlier_metrics > 0
    assert result.normative_verdict in ("BORDERLINE", "OUTLIER")


def test_normative_zscores_missing_metrics():
    """Missing metrics are excluded from z-score computation."""
    metrics = {"qei": 0.732}  # only one metric
    result = compute_normative_zscores(metrics, 3.0, "PCASL")
    assert len(result.z_scores) == 1
    assert "qei" in result.z_scores


def test_adaptive_thresholds_exist():
    """Adaptive thresholds returned for standard acquisition types."""
    thresh = get_adaptive_thresholds(3.0, "PCASL")
    assert "qei" in thresh
    assert "scov_gm" in thresh
    assert "mean_fd" in thresh


def test_adaptive_thresholds_stricter_at_7T():
    """7T has stricter QEI minimum than 3T."""
    thresh_3T = get_adaptive_thresholds(3.0, "PCASL")
    thresh_7T = get_adaptive_thresholds(7.0, "PCASL")
    assert thresh_7T["qei"][1] > thresh_3T["qei"][1]


def test_adaptive_thresholds_relaxed_at_1_5T():
    """1.5T has more relaxed sCoV threshold than 3T."""
    thresh_3T = get_adaptive_thresholds(3.0, "PCASL")
    thresh_1_5T = get_adaptive_thresholds(1.5, "PCASL")
    assert thresh_1_5T["scov_gm"][1] > thresh_3T["scov_gm"][1]


def test_adaptive_thresholds_pasl():
    """PASL thresholds are more relaxed than PCASL at 3T."""
    thresh_pcasl = get_adaptive_thresholds(3.0, "PCASL")
    thresh_pasl = get_adaptive_thresholds(3.0, "PASL")
    assert thresh_pasl["scov_gm"][1] > thresh_pcasl["scov_gm"][1]


def test_normative_percentiles():
    """Percentiles are between 0 and 100."""
    metrics = {"qei": 0.732, "scov_gm": 0.283}
    result = compute_normative_zscores(metrics, 3.0, "PCASL")
    for p in result.percentiles.values():
        assert 0 <= p <= 100
