"""
Example: Working with BIDS-formatted ASL data
==============================================

Shows how to parse BIDS metadata, verify control-label patterns,
and assess M0 calibration images before running QC metrics.
"""

import json
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────
# 1. PARSE BIDS METADATA
# ─────────────────────────────────────────────────────────

from asl_qc.bids import parse_asl_context, parse_asl_metadata, validate_metadata

# For real data, point these to your BIDS files:
# context = parse_asl_context("/path/to/sub-01_aslcontext.tsv")
# metadata = parse_asl_metadata("/path/to/sub-01_asl.json")

# For this example, we create temporary files:
import tempfile, os

tmp = tempfile.mkdtemp()

# Create aslcontext.tsv
with open(f"{tmp}/aslcontext.tsv", "w") as f:
    f.write("volume_type\n")
    for i in range(20):
        f.write("control\n" if i % 2 == 0 else "label\n")

# Create ASL JSON sidecar
sidecar = {
    "ArterialSpinLabelingType": "PCASL",
    "PostLabelingDelay": 1.8,
    "BackgroundSuppression": False,
    "RepetitionTimePreparation": 4.0,
    "LabelingDuration": 1.8,
    "M0Type": "Separate",
    "MagneticFieldStrength": 3.0,
}
with open(f"{tmp}/sub-01_asl.json", "w") as f:
    json.dump(sidecar, f)

# Parse
context = parse_asl_context(f"{tmp}/aslcontext.tsv")
metadata = parse_asl_metadata(f"{tmp}/sub-01_asl.json")

print(f"Volumes: {context.n_volumes} ({context.n_controls} control, {context.n_labels} label)")
print(f"Labeling: {metadata.labeling_type}, PLD: {metadata.post_labeling_delay}s")
print(f"Field Strength: {metadata.magnetic_field_strength}T")

# Check required fields
missing = validate_metadata(metadata)
if missing:
    print(f"⚠️  Missing BIDS fields: {missing}")
else:
    print("✅ All required BIDS fields present")


# ─────────────────────────────────────────────────────────
# 2. CONTROL-LABEL PATTERN VERIFICATION
# ─────────────────────────────────────────────────────────

from asl_qc.preprocessing.control_label import verify_control_label_pattern

# Create synthetic 4D timeseries with control/label alternation
rng = np.random.default_rng(42)
shape = (32, 32, 20)
brain_mask = np.zeros(shape, dtype=bool)
brain_mask[6:26, 6:26, 4:16] = True

data_4d = np.zeros((*shape, 20))
for t in range(20):
    vol = rng.normal(1000, 50, size=shape)
    if t % 2 == 0:  # Control = higher signal
        vol[brain_mask] += 10
    else:            # Label = lower signal
        vol[brain_mask] -= 10
    data_4d[..., t] = vol

pattern = verify_control_label_pattern(
    data_4d, context.volume_types, brain_mask
)
print(f"\nPattern Valid: {pattern.is_valid}")
print(f"Phase Shift: {pattern.phase_shift_detected}")
print(f"Message: {pattern.message}")


# ─────────────────────────────────────────────────────────
# 3. M0 CALIBRATION ASSESSMENT
# ─────────────────────────────────────────────────────────

from asl_qc.preprocessing.m0_calibration import assess_m0

m0_image = rng.normal(5000, 200, size=shape)
m0_image[~brain_mask] = 0

m0_result = assess_m0(
    m0_image, brain_mask,
    has_dedicated_m0=True,
    smoothing_fwhm_mm=5.0,
)
print(f"\nM0 Source: {m0_result.source}")
print(f"Usable: {m0_result.is_usable}")
print(f"Artifact: {m0_result.artifact_flagged}")
print(f"Message: {m0_result.message}")


# ─────────────────────────────────────────────────────────
# 4. MOTION TRACKING
# ─────────────────────────────────────────────────────────

from asl_qc.preprocessing.motion import (
    compute_framewise_displacement,
    compute_dvars,
    summarize_motion,
)

# Simulate motion parameters (N x 6: tx, ty, tz, rx, ry, rz)
motion_params = rng.normal(0, 0.01, size=(20, 6))  # Tiny motion

fd = compute_framewise_displacement(motion_params, radius_mm=50)
dvars = compute_dvars(data_4d, brain_mask)
motion = summarize_motion(fd, dvars)

print(f"\nMean FD:   {motion.mean_fd:.4f} mm")
print(f"Max FD:    {motion.max_fd:.4f} mm")
print(f"FD Spikes: {motion.n_fd_spikes}")
print(f"Mean DVARS: {motion.mean_dvars:.2f}")

# Clean up temp files
import shutil
shutil.rmtree(tmp)
