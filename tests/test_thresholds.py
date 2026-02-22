"""Tests for empirical and ML thresholds."""

import numpy as np
from asl_qc.thresholds.empirical import apply_empirical_thresholds
from asl_qc.thresholds.ml_outlier import (
    build_feature_vector,
    build_feature_matrix,
    train_outlier_model,
    predict_outlier,
    FEATURE_KEYS,
)


# -- Empirical thresholds --

def test_all_pass():
    metrics = {
        "qei": 0.80,
        "scov_gm": 0.30,
        "mean_fd": 0.2,
        "gm_wm_ratio": 2.5,
        "neg_gm_cbf": 0.02,
        "dice": 0.85,
    }
    result = apply_empirical_thresholds(metrics)
    assert result.overall_pass
    assert result.n_failed == 0


def test_qei_fail():
    metrics = {
        "qei": 0.30,  # Below 0.53
        "scov_gm": 0.30,
        "mean_fd": 0.2,
        "gm_wm_ratio": 2.5,
        "neg_gm_cbf": 0.02,
        "dice": 0.85,
    }
    result = apply_empirical_thresholds(metrics)
    assert not result.overall_pass
    assert result.n_failed >= 1


def test_multiple_failures():
    metrics = {
        "qei": 0.10,
        "scov_gm": 0.80,
        "mean_fd": 2.0,
        "gm_wm_ratio": 1.0,
        "neg_gm_cbf": 0.50,
        "dice": 0.30,
    }
    result = apply_empirical_thresholds(metrics)
    assert not result.overall_pass
    assert result.n_failed >= 4


# -- ML outlier detection --

def test_feature_vector():
    metrics = {"qei": 0.8, "scov_gm": 0.3, "mean_fd": 0.2}
    fv = build_feature_vector(metrics)
    assert len(fv) == len(FEATURE_KEYS)
    assert fv[0] == 0.8  # qei
    assert np.isnan(fv[-1])  # rms_difference not provided


def test_train_and_predict():
    rng = np.random.default_rng(42)
    # Normal population
    n_samples = 50
    metrics_list = []
    for _ in range(n_samples):
        m = {
            "qei": rng.normal(0.75, 0.05),
            "scov_gm": rng.normal(0.30, 0.03),
            "mean_fd": rng.normal(0.2, 0.05),
            "gm_wm_ratio": rng.normal(2.5, 0.2),
            "neg_gm_cbf": rng.normal(0.03, 0.01),
            "dice": rng.normal(0.85, 0.03),
            "skewness": rng.normal(0.5, 0.2),
            "kurtosis": rng.normal(0.3, 0.2),
            "tsnr": rng.normal(80, 10),
            "rms_difference": rng.normal(5, 1),
        }
        metrics_list.append(m)

    X = build_feature_matrix(metrics_list)
    model = train_outlier_model(X, method="isolation_forest")

    # Test normal sample
    normal = build_feature_vector(metrics_list[0])
    pred = predict_outlier(normal, model)
    assert not pred.is_outlier

    # Test extreme outlier
    outlier = build_feature_vector({
        "qei": 0.01,
        "scov_gm": 2.0,
        "mean_fd": 5.0,
        "gm_wm_ratio": 0.5,
        "neg_gm_cbf": 0.9,
        "dice": 0.1,
        "skewness": 5.0,
        "kurtosis": 20.0,
        "tsnr": 1.0,
        "rms_difference": 100.0,
    })
    pred_out = predict_outlier(outlier, model)
    assert pred_out.is_outlier
