"""
BIDS metadata parser for ASL MRI datasets.

Handles ``aslcontext.tsv`` files and ASL JSON sidecars to extract labeling
scheme, post-labeling delay, background suppression, and volume ordering.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

VALID_VOLUME_TYPES = {"control", "label", "m0scan", "deltam", "cbf"}


@dataclass
class ASLContext:
    """Parsed representation of an ``aslcontext.tsv``."""

    volume_types: List[str]
    """Ordered list of volume type strings (e.g. ``['control', 'label', ...]``)."""

    @property
    def n_volumes(self) -> int:
        return len(self.volume_types)

    @property
    def n_controls(self) -> int:
        return self.volume_types.count("control")

    @property
    def n_labels(self) -> int:
        return self.volume_types.count("label")

    @property
    def has_m0(self) -> bool:
        return "m0scan" in self.volume_types

    def control_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.volume_types) if v == "control"]

    def label_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.volume_types) if v == "label"]

    def m0_indices(self) -> List[int]:
        return [i for i, v in enumerate(self.volume_types) if v == "m0scan"]


@dataclass
class ASLMetadata:
    """Parsed representation of an ASL BIDS JSON sidecar."""

    labeling_type: Optional[str] = None  # PCASL, PASL, CASL
    post_labeling_delay: Optional[float] = None  # seconds
    background_suppression: Optional[bool] = None
    repetition_time_preparation: Optional[float] = None  # seconds
    labeling_duration: Optional[float] = None  # seconds
    m0_type: Optional[str] = None  # Separate, Included, Estimate, Absent
    magnetic_field_strength: Optional[float] = None  # Tesla
    manufacturer: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_asl_context(filepath: str | Path) -> ASLContext:
    """Parse an ``aslcontext.tsv`` file.

    Parameters
    ----------
    filepath : path-like
        Path to the TSV file.  Expected to have a ``volume_type`` column.

    Returns
    -------
    ASLContext

    Raises
    ------
    ValueError
        If the file is empty or contains unrecognised volume types.
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"aslcontext.tsv not found: {filepath}")

    volume_types: List[str] = []
    with open(filepath, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            vt = row.get("volume_type", "").strip().lower()
            if vt not in VALID_VOLUME_TYPES:
                raise ValueError(
                    f"Unrecognised volume type '{vt}' in {filepath}. "
                    f"Valid types: {VALID_VOLUME_TYPES}"
                )
            volume_types.append(vt)

    if not volume_types:
        raise ValueError(f"aslcontext.tsv is empty: {filepath}")

    return ASLContext(volume_types=volume_types)


def parse_asl_metadata(filepath: str | Path) -> ASLMetadata:
    """Parse an ASL BIDS JSON sidecar.

    Parameters
    ----------
    filepath : path-like
        Path to the JSON sidecar (e.g. ``sub-01_asl.json``).

    Returns
    -------
    ASLMetadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ASL JSON sidecar not found: {filepath}")

    with open(filepath, "r") as fh:
        data: Dict[str, Any] = json.load(fh)

    # Map common BIDS keys → dataclass fields
    pld = data.get("PostLabelingDelay")
    if isinstance(pld, list):
        pld = pld[0]  # Multi-PLD: take first for summary

    return ASLMetadata(
        labeling_type=data.get("ArterialSpinLabelingType"),
        post_labeling_delay=pld,
        background_suppression=data.get("BackgroundSuppression"),
        repetition_time_preparation=data.get("RepetitionTimePreparation"),
        labeling_duration=data.get("LabelingDuration"),
        m0_type=data.get("M0Type"),
        magnetic_field_strength=data.get("MagneticFieldStrength"),
        manufacturer=data.get("Manufacturer"),
        raw=data,
    )


def validate_metadata(metadata: ASLMetadata, required_fields: List[str] | None = None) -> List[str]:
    """Validate that required BIDS fields are present.

    Returns a list of missing field names (empty if all present).
    """
    if required_fields is None:
        required_fields = [
            "RepetitionTimePreparation",
            "ArterialSpinLabelingType",
            "PostLabelingDelay",
        ]
    missing = []
    for f in required_fields:
        if metadata.raw.get(f) is None:
            missing.append(f)
    return missing
