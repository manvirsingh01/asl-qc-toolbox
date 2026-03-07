"""
ExploreASL output completeness validator.

Checks that all required ExploreASL output files exist and are readable
before the QC pipeline runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .discovery import ExploreASLPaths


@dataclass
class ValidationResult:
    """Result of ExploreASL output validation."""

    is_valid: bool
    """True if all required files are present."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_exploreasl_outputs(paths: ExploreASLPaths) -> ValidationResult:
    """
    Validate that discovered ExploreASL outputs are sufficient for QC.

    Parameters
    ----------
    paths : ExploreASLPaths
        Discovered file paths from :func:`discover_exploreasl_outputs`.

    Returns
    -------
    ValidationResult
    """
    errors: List[str] = []
    warnings: List[str] = []

    if paths.missing_required:
        for m in paths.missing_required:
            errors.append(f"Required file missing: {m}")

    if paths.missing_optional:
        for m in paths.missing_optional:
            warnings.append(f"Optional file missing: {m}")

    # Check that found files are actually readable
    for attr in ["cbf_map", "gm_mask", "wm_mask", "brain_mask"]:
        p = getattr(paths, attr, None)
        if p is not None and not p.is_file():
            errors.append(f"{attr} path exists but is not a file: {p}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
    )
