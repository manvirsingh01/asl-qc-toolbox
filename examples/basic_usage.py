"""
Example: Using individual QC metrics on your own data
=====================================================

This script shows how to use each metric module independently.
You can copy any section into your own analysis script.
"""

import numpy as np
import nibabel as nib

# ─────────────────────────────────────────────────────────
# 1. LOAD YOUR DATA  (replace paths with your own files)
# ─────────────────────────────────────────────────────────

# cbf_img = nib.load("path/to/your/cbf.nii.gz")
# cbf_data = np.asarray(cbf_img.dataobj, dtype=np.float64)
#
# gm_img = nib.load("path/to/your/gm_mask.nii.gz")
# gm_mask = np.asarray(gm_img.dataobj) > 0.5
#
# wm_img = nib.load("path/to/your/wm_mask.nii.gz")
# wm_mask = np.asarray(wm_img.dataobj) > 0.5

# For this example, we create synthetic data:
from scipy.ndimage import binary_erosion

rng = np.random.default_rng(42)
shape = (32, 32, 20)
brain = np.zeros(shape, dtype=bool)
brain[6:26, 6:26, 4:16] = True
gm_mask = brain & ~binary_erosion(brain, iterations=2)
wm_mask = binary_erosion(brain, iterations=2)
cbf_data = np.zeros(shape)
cbf_data[gm_mask] = 60 + rng.normal(0, 5, size=int(np.sum(gm_mask)))
cbf_data[wm_mask] = 25 + rng.normal(0, 3, size=int(np.sum(wm_mask)))


# ─────────────────────────────────────────────────────────
# 2. QUALITY EVALUATION INDEX (QEI)
# ─────────────────────────────────────────────────────────

from asl_qc.metrics.qei import compute_qei

qei_result = compute_qei(cbf_data, gm_mask, wm_mask)
print(f"QEI = {qei_result.qei:.4f}")
# Access sub-components:
#   qei_result.structural_similarity
#   qei_result.index_of_dispersion
#   qei_result.negative_gm_fraction


# ─────────────────────────────────────────────────────────
# 3. SPATIAL COEFFICIENT OF VARIATION (sCoV)
# ─────────────────────────────────────────────────────────

from asl_qc.metrics.scov import compute_scov

scov_value = compute_scov(cbf_data, gm_mask)
print(f"sCoV = {scov_value:.4f}")

# For multiple regions:
from asl_qc.metrics.scov import compute_regional_scov
regions = {"GM": gm_mask, "WM": wm_mask}
regional = compute_regional_scov(cbf_data, regions)
print(f"Regional sCoV = {regional}")


# ─────────────────────────────────────────────────────────
# 4. HISTOGRAM ANALYSIS
# ─────────────────────────────────────────────────────────

from asl_qc.metrics.histogram import analyze_histogram

hist = analyze_histogram(cbf_data, gm_mask)
print(f"Skewness = {hist.skewness:.4f}")
print(f"Kurtosis = {hist.kurtosis:.4f}")
print(f"5th-95th percentile range: [{hist.percentile_5:.1f}, {hist.percentile_95:.1f}]")


# ─────────────────────────────────────────────────────────
# 5. TISSUE MASK QUALITY (Dice / Jaccard / GM-WM Ratio)
# ─────────────────────────────────────────────────────────

from asl_qc.metrics.tissue_mask import compute_dice, compute_gm_wm_ratio

# Compare two masks (e.g. subject vs atlas)
dice = compute_dice(gm_mask, gm_mask)  # Perfect overlap → 1.0
print(f"Dice = {dice:.4f}")

ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf_data, gm_mask, wm_mask)
print(f"GM/WM Ratio = {ratio:.2f}  (GM={mean_gm:.1f}, WM={mean_wm:.1f})")


# ─────────────────────────────────────────────────────────
# 6. APPLY THRESHOLDS  (pass / fail each metric)
# ─────────────────────────────────────────────────────────

from asl_qc.thresholds.empirical import apply_empirical_thresholds

all_metrics = {
    "qei": qei_result.qei,
    "scov_gm": scov_value,
    "mean_fd": 0.25,          # From your motion parameters
    "gm_wm_ratio": ratio,
    "neg_gm_cbf": qei_result.negative_gm_fraction,
    "dice": dice,
}

verdict = apply_empirical_thresholds(all_metrics)
print(f"\nOverall: {'PASS' if verdict.overall_pass else 'FAIL'}")
for v in verdict.verdicts:
    print(f"  {v.metric_name}: {'✅' if v.passed else '❌'} ({v.value:.3f} {v.operator} {v.threshold:.3f})")
