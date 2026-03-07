"""
ExploreASL output format helpers.

Provides utilities for reading ExploreASL-specific file formats
(e.g. motion parameter files, QC collection JSON).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def read_motion_params(path: str | Path) -> np.ndarray:
    """Read an ExploreASL motion parameter file (N×6 text).

    Parameters
    ----------
    path : path-like
        Path to ``rp_*.txt`` file produced by ExploreASL/SPM.
        Columns: 3 translations (mm) + 3 rotations (rad).

    Returns
    -------
    np.ndarray
        (N, 6) motion parameter matrix.
    """
    return np.loadtxt(str(path))


def read_exploreasl_stats_csv(path: str | Path) -> Dict[str, float]:
    """Read an ExploreASL ROI statistics CSV.

    Parameters
    ----------
    path : path-like
        Path to a ``*_stats.csv`` file from the ExploreASL Population/Stats directory.

    Returns
    -------
    dict
        Mapping of ROI name → mean CBF value.
    """
    import csv
    result: Dict[str, float] = {}
    with open(str(path), "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key, value in row.items():
                try:
                    result[key.strip()] = float(value)
                except (ValueError, TypeError):
                    continue
    return result
