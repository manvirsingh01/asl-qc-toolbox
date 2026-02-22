"""
JSON report generator for ASL QC results.

Serialises all metric results, pass/fail verdicts, and provenance metadata
into a structured JSON file suitable for machine consumption and
downstream statistical analysis.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and dataclasses."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


def generate_json_report(
    metrics: Dict[str, Any],
    verdicts: Dict[str, Any],
    input_files: Dict[str, str],
    config_path: Optional[str] = None,
    output_path: str | Path = "qc_report.json",
) -> Path:
    """Generate a structured JSON QC report.

    Parameters
    ----------
    metrics : dict
        All computed QC metric results.
    verdicts : dict
        Threshold verdicts (empirical and/or ML).
    input_files : dict
        Mapping of input type → file path for provenance.
    config_path : str, optional
        Path to the configuration YAML used.
    output_path : path-like
        Where to write the JSON report.

    Returns
    -------
    Path
        Absolute path to the generated report.
    """
    output_path = Path(output_path)

    # Provenance
    config_hash = ""
    if config_path:
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    report = {
        "provenance": {
            "toolbox": "asl-qc-toolbox",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "input_files": input_files,
        },
        "metrics": metrics,
        "verdicts": verdicts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2, cls=_NumpyEncoder)

    return output_path.resolve()
