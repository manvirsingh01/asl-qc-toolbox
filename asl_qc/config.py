"""
YAML-based pipeline configuration loader.

Loads default config, merges with user overrides, and exposes a
``PipelineConfig`` dataclass for type-safe access throughout the toolbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ControlLabelConfig:
    autocorrelation_lag: int = 1
    min_pattern_amplitude: float = 0.005


@dataclass
class M0Config:
    max_median_ratio_threshold: float = 10.0
    smoothing_fwhm_mm: float = 5.0
    low_signal_fraction: float = 0.10


@dataclass
class MotionConfig:
    fd_radius_mm: float = 50.0
    fd_spike_threshold_mm: float = 0.5
    dvars_spike_threshold: float = 1.5


@dataclass
class ScoreConfig:
    correlation_threshold: float = 0.6
    min_retained_fraction: float = 0.50


@dataclass
class PreprocessingConfig:
    control_label: ControlLabelConfig = field(default_factory=ControlLabelConfig)
    m0: M0Config = field(default_factory=M0Config)
    motion: MotionConfig = field(default_factory=MotionConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)


@dataclass
class QEIConfig:
    gm_weight: float = 2.5
    wm_weight: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


@dataclass
class SCoVConfig:
    use_gm_mask: bool = True


@dataclass
class HistogramConfig:
    nbins: int = 100


@dataclass
class TissueMaskConfig:
    gm_threshold: float = 0.5
    wm_threshold: float = 0.5


@dataclass
class MetricsConfig:
    qei: QEIConfig = field(default_factory=QEIConfig)
    scov: SCoVConfig = field(default_factory=SCoVConfig)
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)


@dataclass
class EmpiricalThresholds:
    qei_min: float = 0.53
    scov_max: float = 0.42
    mean_fd_max_mm: float = 0.5
    gm_wm_ratio_min: float = 2.0
    gm_wm_ratio_max: float = 3.0
    neg_gm_cbf_max: float = 0.10
    dice_min: float = 0.70


@dataclass
class MLThresholds:
    method: str = "isolation_forest"
    contamination: float = 0.05
    n_estimators: int = 100
    random_state: int = 42


@dataclass
class ThresholdsConfig:
    empirical: EmpiricalThresholds = field(default_factory=EmpiricalThresholds)
    ml: MLThresholds = field(default_factory=MLThresholds)


@dataclass
class ReportingConfig:
    json: bool = True
    html: bool = True
    include_plots: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the ASL QC pipeline."""

    name: str = "asl-qc-toolbox"
    version: str = "1.0.0"
    verbose: bool = True
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    """Recursively convert a nested dict to the corresponding dataclass tree."""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # Resolve string annotations
        if isinstance(ft, str):
            ft = eval(ft)  # noqa: S307
        if hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(user_config_path: Optional[str] = None) -> PipelineConfig:
    """Load the pipeline configuration.

    Parameters
    ----------
    user_config_path : str, optional
        Path to a user-provided YAML file whose values override the defaults.

    Returns
    -------
    PipelineConfig
        Fully merged configuration dataclass.
    """
    # Load defaults
    with open(_DEFAULT_CONFIG_PATH, "r") as fh:
        base: Dict[str, Any] = yaml.safe_load(fh) or {}

    # Merge user overrides
    if user_config_path is not None:
        with open(user_config_path, "r") as fh:
            overrides: Dict[str, Any] = yaml.safe_load(fh) or {}
        _deep_merge(base, overrides)

    # Flatten 'pipeline' key into top-level
    pipeline_section = base.pop("pipeline", {})
    base.update(pipeline_section)

    return _dict_to_dataclass(PipelineConfig, base)
