"""
Empirical linear thresholds for ASL QC classification.

Provides literature-based gate thresholds for individual QC metrics and
a function that evaluates a complete metrics dictionary against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Default thresholds (from literature: Shirzadi et al., ISMRM/NeuroImage)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    # metric_key: (comparison_operator, threshold_value)
    "qei":              (">=", 0.53),
    "scov_gm":          ("<=", 0.42),
    "mean_fd":          ("<=", 0.50),
    "gm_wm_ratio_min":  (">=", 2.00),
    "gm_wm_ratio_max":  ("<=", 3.00),
    "neg_gm_cbf":       ("<=", 0.10),
    "dice":             (">=", 0.70),
}


@dataclass
class ThresholdVerdict:
    """Result of applying a single threshold."""

    metric_name: str
    value: float
    threshold: float
    operator: str
    passed: bool


@dataclass
class EmpiricalResult:
    """Result of empirical threshold evaluation."""

    overall_pass: bool
    verdicts: List[ThresholdVerdict] = field(default_factory=list)
    n_passed: int = 0
    n_failed: int = 0


def _evaluate(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == ">":
        return value > threshold
    elif operator == "<":
        return value < threshold
    elif operator == "==":
        return value == threshold
    raise ValueError(f"Unknown operator: {operator}")


def apply_empirical_thresholds(
    metrics: Dict[str, float],
    thresholds: Dict[str, Tuple[str, float]] | None = None,
) -> EmpiricalResult:
    """Evaluate all metrics against empirical thresholds.

    Parameters
    ----------
    metrics : dict
        Mapping of metric key → numeric value.  Expected keys:

        - ``qei``
        - ``scov_gm``
        - ``mean_fd``
        - ``gm_wm_ratio``  (checked against both min and max bounds)
        - ``neg_gm_cbf``
        - ``dice``
    thresholds : dict, optional
        Custom thresholds (defaults to ``DEFAULT_THRESHOLDS``).

    Returns
    -------
    EmpiricalResult
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    verdicts: List[ThresholdVerdict] = []

    for key, (op, thresh) in thresholds.items():
        # Map gm_wm_ratio to the correct check
        if key == "gm_wm_ratio_min":
            value = metrics.get("gm_wm_ratio", float("nan"))
            name = "gm_wm_ratio (min)"
        elif key == "gm_wm_ratio_max":
            value = metrics.get("gm_wm_ratio", float("nan"))
            name = "gm_wm_ratio (max)"
        else:
            value = metrics.get(key, float("nan"))
            name = key

        import math
        if math.isnan(value):
            verdicts.append(
                ThresholdVerdict(
                    metric_name=name,
                    value=value,
                    threshold=thresh,
                    operator=op,
                    passed=False,
                )
            )
            continue

        passed = _evaluate(value, op, thresh)
        verdicts.append(
            ThresholdVerdict(
                metric_name=name,
                value=value,
                threshold=thresh,
                operator=op,
                passed=passed,
            )
        )

    n_passed = sum(1 for v in verdicts if v.passed)
    n_failed = len(verdicts) - n_passed

    return EmpiricalResult(
        overall_pass=(n_failed == 0),
        verdicts=verdicts,
        n_passed=n_passed,
        n_failed=n_failed,
    )
