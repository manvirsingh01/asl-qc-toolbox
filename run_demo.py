#!/usr/bin/env python3
"""
ASL QC Toolbox — Quick Start Demo
==================================

Run this script to see the toolbox in action with synthetic data.
No real MRI data needed!

Usage:
    python run_demo.py
"""

import numpy as np
from scipy.ndimage import binary_erosion

# ── Step 1: Generate synthetic brain data ──────────────────────────
print("🧠 ASL QC Toolbox — Demo\n")
print("Step 1: Generating synthetic brain data...")

rng = np.random.default_rng(42)
shape = (32, 32, 20)

# Brain mask (ellipsoid)
mask = np.zeros(shape, dtype=bool)
cx, cy, cz = 16, 16, 10
for x in range(32):
    for y in range(32):
        for z in range(20):
            if ((x - cx) / 10) ** 2 + ((y - cy) / 10) ** 2 + ((z - cz) / 6) ** 2 <= 1:
                mask[x, y, z] = True

# Tissue masks
gm_mask = mask & ~binary_erosion(mask, iterations=2)
wm_mask = binary_erosion(mask, iterations=2)

# CBF map: GM ~ 60 ml/100g/min, WM ~ 25 ml/100g/min
cbf = np.zeros(shape)
cbf[gm_mask] = 60 + rng.normal(0, 5, size=int(np.sum(gm_mask)))
cbf[wm_mask] = 25 + rng.normal(0, 3, size=int(np.sum(wm_mask)))

print(f"   Brain mask: {np.sum(mask)} voxels")
print(f"   GM voxels:  {np.sum(gm_mask)}")
print(f"   WM voxels:  {np.sum(wm_mask)}")

# ── Step 2: Compute QC Metrics ─────────────────────────────────────
print("\nStep 2: Computing QC metrics...\n")

# Quality Evaluation Index
from asl_qc.metrics.qei import compute_qei

qei = compute_qei(cbf, gm_mask, wm_mask)
print(f"   QEI Score:              {qei.qei:.4f}")
print(f"   Structural Similarity:  {qei.structural_similarity:.4f}")
print(f"   Negative GM Fraction:   {qei.negative_gm_fraction:.4f}")
print(f"   Mean GM CBF:            {qei.mean_gm_cbf:.1f} ml/100g/min")

# Spatial Coefficient of Variation
from asl_qc.metrics.scov import compute_scov

scov = compute_scov(cbf, gm_mask)
print(f"\n   sCoV (GM):              {scov:.4f}")

# Histogram Analysis
from asl_qc.metrics.histogram import analyze_histogram

hist = analyze_histogram(cbf, gm_mask)
print(f"\n   Histogram Skewness:     {hist.skewness:.4f}")
print(f"   Histogram Kurtosis:     {hist.kurtosis:.4f}")
print(f"   5th Percentile:         {hist.percentile_5:.1f}")
print(f"   95th Percentile:        {hist.percentile_95:.1f}")

# GM/WM Ratio
from asl_qc.metrics.tissue_mask import compute_gm_wm_ratio

ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf, gm_mask, wm_mask)
print(f"\n   GM/WM CBF Ratio:        {ratio:.2f}")

# ── Step 3: Apply Thresholds ───────────────────────────────────────
print("\nStep 3: Applying quality thresholds...\n")

from asl_qc.thresholds.empirical import apply_empirical_thresholds

metrics = {
    "qei": qei.qei,
    "scov_gm": scov,
    "mean_fd": 0.3,  # Simulated low motion
    "gm_wm_ratio": ratio,
    "neg_gm_cbf": qei.negative_gm_fraction,
    "dice": 0.85,  # Simulated good registration
}

result = apply_empirical_thresholds(metrics)

for v in result.verdicts:
    status = "✅ PASS" if v.passed else "❌ FAIL"
    print(f"   {v.metric_name:25s}  {v.value:8.4f}  {v.operator} {v.threshold:.3f}  {status}")

print(f"\n   {'✅ OVERALL PASS' if result.overall_pass else '❌ OVERALL FAIL'}")
print(f"   {result.n_passed}/{result.n_passed + result.n_failed} metrics passed")

# ── Step 4: ML Outlier Detection ───────────────────────────────────
print("\nStep 4: Training ML outlier detector...\n")

from asl_qc.thresholds.ml_outlier import (
    build_feature_vector,
    build_feature_matrix,
    train_outlier_model,
    predict_outlier,
)

# Simulate a normative population
population = []
for _ in range(50):
    population.append({
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
    })

X = build_feature_matrix(population)
model = train_outlier_model(X, method="isolation_forest")

# Test our sample
our_metrics = {**metrics, "skewness": hist.skewness, "kurtosis": hist.kurtosis, "tsnr": 80, "rms_difference": 5}
pred = predict_outlier(build_feature_vector(our_metrics), model)
print(f"   ML Verdict:  {'🚨 OUTLIER' if pred.is_outlier else '✅ INLIER'}")
print(f"   Anomaly Score: {pred.anomaly_score:.4f}")

# ── Done ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("🎉 Demo complete! All modules working correctly.")
print("=" * 55)
print("\nNext steps:")
print("  • Edit this script to load your own NIfTI data")
print("  • Use 'asl-qc --help' for the full CLI")
print("  • See examples/ for more usage patterns")
print("  • Run 'python -m pytest tests/ -v' for the test suite")
# ── Step 5: HTML Report ───────────────────────────────────────────
from asl_qc.reporting.html_report import generate_html_report
from datetime import datetime
import os

html_path = generate_html_report(
    subject_id="sub-demo",
    timestamp=datetime.now().isoformat(),
    overall_pass=result.overall_pass,
    verdicts=[{
        "metric_name": v.metric_name,
        "value": v.value,
        "threshold": v.threshold,
        "operator": v.operator,
        "passed": v.passed
    } for v in result.verdicts],
    summary_stats=[
        {"value": f"{qei.qei:.3f}", "label": "QEI"},
        {"value": f"{scov:.3f}", "label": "sCoV (GM)"},
        {"value": "0.300 mm", "label": "Mean FD"},
        {"value": f"{ratio:.2f}", "label": "GM/WM Ratio"},
    ],
    input_files={"cbf_map": "synthetic_cbf", "gm_mask": "synthetic_gm"},
    ml_verdict={"method": "isolation_forest", "anomaly_score": pred.anomaly_score, "is_outlier": pred.is_outlier},
    output_path="demo_report.html"
)
print(f"\n   Report generated at: file://{os.path.abspath('demo_report.html')}")
