"""
Machine-learning-based multi-dimensional outlier detection for ASL QC.

Projects the full suite of QC metrics into a high-dimensional feature
space and learns the boundary of "acceptable" variance using unsupervised
anomaly detection (Isolation Forest or One-Class SVM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import joblib

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

# Canonical order of features for consistent model I/O
FEATURE_KEYS: List[str] = [
    "qei",
    "scov_gm",
    "mean_fd",
    "gm_wm_ratio",
    "neg_gm_cbf",
    "dice",
    "skewness",
    "kurtosis",
    "tsnr",
    "rms_difference",
]


def build_feature_vector(metrics: Dict[str, float]) -> np.ndarray:
    """Convert a metrics dictionary to a 1-D feature vector.

    Missing keys are filled with ``np.nan``.

    Parameters
    ----------
    metrics : dict
        Metric name → value.

    Returns
    -------
    np.ndarray
        (N_features,) array.
    """
    return np.array([metrics.get(k, np.nan) for k in FEATURE_KEYS], dtype=np.float64)


def build_feature_matrix(metrics_list: List[Dict[str, float]]) -> np.ndarray:
    """Build a (N_samples, N_features) matrix from a list of metrics dicts."""
    return np.vstack([build_feature_vector(m) for m in metrics_list])


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

@dataclass
class OutlierModel:
    """Wrapper around a trained anomaly detection model."""

    model: Any
    scaler: StandardScaler
    method: str
    feature_keys: List[str]


def train_outlier_model(
    feature_matrix: np.ndarray,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
) -> OutlierModel:
    """Train an unsupervised outlier detection model.

    Parameters
    ----------
    feature_matrix : np.ndarray
        (N_samples, N_features) normative training data.
    method : str
        ``"isolation_forest"`` or ``"one_class_svm"``.
    contamination : float
        Expected fraction of outliers in the training data.
    n_estimators : int
        Number of trees (Isolation Forest only).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    OutlierModel
    """
    # Handle NaNs by imputing column median
    col_medians = np.nanmedian(feature_matrix, axis=0)
    for j in range(feature_matrix.shape[1]):
        nan_mask = np.isnan(feature_matrix[:, j])
        feature_matrix[nan_mask, j] = col_medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    if method == "isolation_forest":
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
    elif method == "one_class_svm":
        model = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=contamination,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'isolation_forest' or 'one_class_svm'.")

    model.fit(X_scaled)

    return OutlierModel(
        model=model,
        scaler=scaler,
        method=method,
        feature_keys=list(FEATURE_KEYS),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class OutlierPrediction:
    """Result of outlier prediction for a single sample."""

    is_outlier: bool
    anomaly_score: float
    """Anomaly score (lower = more anomalous for Isolation Forest)."""
    label: int
    """Raw model label: 1 = inlier, -1 = outlier."""


def predict_outlier(
    feature_vector: np.ndarray,
    outlier_model: OutlierModel,
) -> OutlierPrediction:
    """Predict whether a single scan is an outlier.

    Parameters
    ----------
    feature_vector : np.ndarray
        (N_features,) vector (from ``build_feature_vector``).
    outlier_model : OutlierModel
        Trained model.

    Returns
    -------
    OutlierPrediction
    """
    # Impute NaNs with 0 (will be scaled)
    fv = feature_vector.copy()
    fv[np.isnan(fv)] = 0.0

    X = outlier_model.scaler.transform(fv.reshape(1, -1))
    label = int(outlier_model.model.predict(X)[0])
    score = float(outlier_model.model.decision_function(X)[0])

    return OutlierPrediction(
        is_outlier=(label == -1),
        anomaly_score=score,
        label=label,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(outlier_model: OutlierModel, path: str | Path) -> None:
    """Save a trained model to disk using ``joblib``."""
    if not _HAS_JOBLIB:
        raise ImportError("joblib is required to save models: pip install joblib")
    joblib.dump(outlier_model, str(path))


def load_model(path: str | Path) -> OutlierModel:
    """Load a trained model from disk."""
    if not _HAS_JOBLIB:
        raise ImportError("joblib is required to load models: pip install joblib")
    return joblib.load(str(path))
