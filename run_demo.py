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

# ── Step 1: Generate synthetic brain data ──────────────────────────
print("[ASL-QC] ASL QC Toolbox -- Demo\n")
print("Step 1: Generating synthetic brain data (121x145x121)...")

from asl_qc.phantom import (
    generate_brain_phantom, generate_cbf_map,
    generate_temporal_sd, generate_motion_params, generate_m0_image,
)

rng = np.random.default_rng(42)
shape = (121, 145, 121)

phantom = generate_brain_phantom(shape=shape, rng=rng)
mask = phantom.brain_mask
gm_mask = phantom.gm_mask > 0.5
wm_mask = phantom.wm_mask > 0.5

# CBF map: GM ~ 60 ml/100g/min, WM ~ 22 ml/100g/min
cbf = generate_cbf_map(phantom, gm_cbf=60, wm_cbf=22, gm_noise=3, wm_noise=2, rng=rng)

# Probabilistic maps for visualisation
gm_prob = phantom.gm_mask
wm_prob = phantom.wm_mask

# Additional ExploreASL QC data
temporal_sd = generate_temporal_sd(phantom, rng=rng)
motion_params = generate_motion_params(n_volumes=80, max_displacement=0.3, rng=rng)
m0 = generate_m0_image(phantom, rng=rng)

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
    status = "[PASS]" if v.passed else "[FAIL]"
    print(f"   {v.metric_name:25s}  {v.value:8.4f}  {v.operator} {v.threshold:.3f}  {status}")

print(f"\n   {'[PASS] OVERALL PASS' if result.overall_pass else '[FAIL] OVERALL FAIL'}")
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
print(f"   ML Verdict:  {'[ALERT] OUTLIER' if pred.is_outlier else '[PASS] INLIER'}")
print(f"   Anomaly Score: {pred.anomaly_score:.4f}")

# ── Done ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("[DONE] Demo complete! All modules working correctly.")
print("=" * 55)
print("\nNext steps:")
print("  • Edit this script to load your own NIfTI data")
print("  • Use 'asl-qc --help' for the full CLI")
print("  • See examples/ for more usage patterns")
print("  • Run 'python -m pytest tests/ -v' for the test suite")
# ── Step 5: HTML Report with ExploreASL QC Images ─────────────────
from asl_qc.reporting.html_report import generate_html_report
from asl_qc.reporting.exploreasl_images import generate_all_qc_images
from datetime import datetime
import os

print("\n   Rendering ExploreASL QC images at 300 DPI...")

images = generate_all_qc_images(
    t1w=phantom.t1w,
    cbf=cbf,
    gm_prob=gm_prob,
    wm_prob=wm_prob,
    brain_mask=mask,
    temporal_sd=temporal_sd,
    motion_params=motion_params,
    m0=m0,
    cbf_gm_values=cbf[gm_mask],
    hist_stats={
        "percentile_5": hist.percentile_5,
        "percentile_95": hist.percentile_95,
        "skewness": hist.skewness,
        "kurtosis": hist.kurtosis,
        "caption_extra": "Tight distribution centred at ~60 ml/100g/min.",
    },
    cbf_vmin=-10,
    cbf_vmax=80,
)

# Build data summary sections
data_summary = [
    {
        "title": "Synthetic Brain Geometry",
        "rows": [
            {"name": "Volume shape", "value": f"{shape}"},
            {"name": "Total brain voxels", "value": f"{int(np.sum(mask))}"},
            {"name": "GM voxels", "value": f"{int(np.sum(gm_mask))}"},
            {"name": "WM voxels", "value": f"{int(np.sum(wm_mask))}"},
        ],
    },
    {
        "title": "CBF Map Statistics",
        "rows": [
            {"name": "GM CBF generation", "value": "60 +/- 3 ml/100g/min"},
            {"name": "WM CBF generation", "value": "22 +/- 2 ml/100g/min"},
            {"name": "Measured mean GM CBF", "value": f"{mean_gm:.1f} ml/100g/min"},
            {"name": "Measured mean WM CBF", "value": f"{mean_wm:.1f} ml/100g/min"},
            {"name": "GM/WM ratio", "value": f"{ratio:.2f} (expected 2.0-3.0)"},
            {"name": "Negative GM voxel fraction", "value": f"{qei.negative_gm_fraction:.4f} ({qei.negative_gm_fraction*100:.1f}%)"},
        ],
    },
    {
        "title": "ExploreASL QC Reference Values",
        "rows": [
            {"name": "Expected GM CBF (PVC0)", "value": "~52 ml/100g/min"},
            {"name": "Expected GM CBF (PVC2)", "value": "~65 ml/100g/min"},
            {"name": "sCoV threshold", "value": "<= 0.42"},
            {"name": "QEI threshold", "value": ">= 0.53"},
        ],
    },
    {
        "title": "QEI Sub-Scores",
        "rows": [
            {"name": "QEI (composite)", "value": f"{qei.qei:.4f}"},
            {"name": "Structural similarity (C_ss)", "value": f"{qei.structural_similarity:.4f}"},
            {"name": "Index of dispersion (C_v)", "value": f"{qei.index_of_dispersion:.4f}"},
            {"name": "Negative GM fraction (C_neg)", "value": f"{qei.negative_gm_fraction:.4f}"},
        ],
    },
    {
        "title": "Histogram Analysis (GM)",
        "rows": [
            {"name": "Mean", "value": f"{hist.mean:.1f}"},
            {"name": "Median", "value": f"{hist.median:.1f}"},
            {"name": "Std Dev", "value": f"{hist.std:.1f}"},
            {"name": "Skewness", "value": f"{hist.skewness:.4f}"},
            {"name": "Kurtosis", "value": f"{hist.kurtosis:.4f}"},
            {"name": "5th percentile", "value": f"{hist.percentile_5:.1f}"},
            {"name": "95th percentile", "value": f"{hist.percentile_95:.1f}"},
            {"name": "IQR", "value": f"{hist.iqr:.1f}"},
            {"name": "Voxels analysed", "value": f"{hist.n_voxels}"},
        ],
    },
    {
        "title": "Simulated Acquisition Parameters",
        "rows": [
            {"name": "Mean FD (framewise displacement)", "value": "0.300 mm (low motion)"},
            {"name": "Dice coefficient (registration)", "value": "0.850 (good alignment)"},
            {"name": "tSNR", "value": "80 (good)"},
            {"name": "RMS difference", "value": "5 (low)"},
        ],
    },
]

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
    input_files={"cbf_map": "synthetic_cbf", "gm_mask": "synthetic_gm", "wm_mask": "synthetic_wm", "brain_mask": "synthetic_brain"},
    ml_verdict={"method": "isolation_forest", "anomaly_score": pred.anomaly_score, "is_outlier": pred.is_outlier},
    data_summary=data_summary,
    images=images,
    output_path="demo_report.html"
)
print(f"\n   Report generated at: file://{os.path.abspath('demo_report.html')}")
