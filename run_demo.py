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
print("[ASL-QC] ASL QC Toolbox -- Demo\n")
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
# ── Step 5: HTML Report ───────────────────────────────────────────
from asl_qc.reporting.html_report import generate_html_report
from datetime import datetime
import os
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _render_slice(volume, title, cmap="RdBu_r", vmin=None, vmax=None, mask_overlay=None):
    """Render axial/coronal/sagittal mid-slices to a base64 PNG."""
    mid = [s // 2 for s in volume.shape]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor="#1e293b")
    slices = [
        volume[mid[0], :, :],   # sagittal
        volume[:, mid[1], :],   # coronal
        volume[:, :, mid[2]],   # axial
    ]
    labels = ["Sagittal", "Coronal", "Axial"]
    for ax, sl, lbl in zip(axes, slices, labels):
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        if mask_overlay is not None:
            overlay_slices = [
                mask_overlay[mid[0], :, :],
                mask_overlay[:, mid[1], :],
                mask_overlay[:, :, mid[2]],
            ]
            ov = overlay_slices[labels.index(lbl)]
            ax.contour(ov.T, levels=[0.5], colors="lime", linewidths=0.8, origin="lower")
        ax.set_title(lbl, color="#e2e8f0", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, color="#e2e8f0", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
    fig.subplots_adjust(right=0.92, top=0.88)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


print("\n   Rendering slice images...")

images = [
    {
        "title": "CBF Map",
        "base64": _render_slice(cbf, "CBF Map (ml/100g/min)",
                                cmap="RdBu_r", vmin=-10, vmax=80,
                                mask_overlay=gm_mask.astype(float)),
        "caption": "CBF with GM contour (green). Good contrast between GM (~60) and WM (~25).",
    },
    {
        "title": "GM Mask",
        "base64": _render_slice(gm_mask.astype(float), "Grey Matter Mask",
                                cmap="Greens", vmin=0, vmax=1),
        "caption": "Binary GM tissue mask used for QC metrics.",
    },
    {
        "title": "WM Mask",
        "base64": _render_slice(wm_mask.astype(float), "White Matter Mask",
                                cmap="Blues", vmin=0, vmax=1),
        "caption": "Binary WM tissue mask.",
    },
    {
        "title": "Brain Mask",
        "base64": _render_slice(mask.astype(float), "Brain Mask",
                                cmap="Oranges", vmin=0, vmax=1),
        "caption": "Full brain mask (GM + WM).",
    },
]

# CBF histogram plot
fig_hist, ax_hist = plt.subplots(figsize=(6, 3), facecolor="#1e293b")
gm_vals = cbf[gm_mask]
ax_hist.hist(gm_vals, bins=40, color="#6366f1", alpha=0.8, edgecolor="#334155")
ax_hist.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="CBF = 0")
ax_hist.axvline(hist.percentile_5, color="#f59e0b", linestyle=":", linewidth=1, label=f"P5 = {hist.percentile_5:.0f}")
ax_hist.axvline(hist.percentile_95, color="#22c55e", linestyle=":", linewidth=1, label=f"P95 = {hist.percentile_95:.0f}")
ax_hist.set_xlabel("CBF (ml/100g/min)", color="#e2e8f0", fontsize=9)
ax_hist.set_ylabel("Voxel count", color="#e2e8f0", fontsize=9)
ax_hist.set_title("GM CBF Histogram", color="#e2e8f0", fontsize=11, fontweight="bold")
ax_hist.tick_params(colors="#94a3b8", labelsize=7)
ax_hist.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
for spine in ax_hist.spines.values():
    spine.set_color("#334155")
ax_hist.set_facecolor("#0f172a")
fig_hist.tight_layout()
buf_h = io.BytesIO()
fig_hist.savefig(buf_h, format="png", dpi=120, facecolor=fig_hist.get_facecolor())
plt.close(fig_hist)
buf_h.seek(0)
images.append({
    "title": "GM CBF Histogram",
    "base64": base64.b64encode(buf_h.read()).decode("ascii"),
    "caption": f"Skewness={hist.skewness:.3f}, Kurtosis={hist.kurtosis:.3f}. "
               f"Tight distribution centred at ~60 ml/100g/min.",
})

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
            {"name": "GM CBF generation", "value": "60 ± 5 ml/100g/min"},
            {"name": "WM CBF generation", "value": "25 ± 3 ml/100g/min"},
            {"name": "Measured mean GM CBF", "value": f"{mean_gm:.1f} ml/100g/min"},
            {"name": "Measured mean WM CBF", "value": f"{mean_wm:.1f} ml/100g/min"},
            {"name": "GM/WM ratio", "value": f"{ratio:.2f} (expected 2.0–3.0)"},
            {"name": "Negative GM voxel fraction", "value": f"{qei.negative_gm_fraction:.4f} ({qei.negative_gm_fraction*100:.1f}%)"},
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
