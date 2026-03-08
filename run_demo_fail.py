#!/usr/bin/env python3
"""
ASL QC Toolbox — Failing QC Demo (ExploreASL pipeline)
=======================================================

Creates a fake ExploreASL directory with deliberately poor-quality
synthetic NIfTI data, runs the full QC pipeline through the ExploreASL
discovery → validation → metrics → thresholds → report path, and
produces an HTML report that shows FAIL verdicts.

Usage:
    python run_demo_fail.py
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

# ── Step 1: Build a fake ExploreASL directory tree ─────────────────
print("[ASL-QC] ASL QC Toolbox -- FAILING-QC Demo (ExploreASL)\n")
print("Step 1: Creating fake ExploreASL directory with bad data...")

rng = np.random.default_rng(99)
shape = (32, 32, 20)
affine = np.eye(4)

# Brain mask (ellipsoid)
mask = np.zeros(shape, dtype=bool)
cx, cy, cz = 16, 16, 10
for x in range(32):
    for y in range(32):
        for z in range(20):
            if ((x - cx) / 10) ** 2 + ((y - cy) / 10) ** 2 + ((z - cz) / 6) ** 2 <= 1:
                mask[x, y, z] = True

gm_mask = mask & ~binary_erosion(mask, iterations=2)
wm_mask = binary_erosion(mask, iterations=2)

# ── Deliberately BAD CBF map ──────────────────────────────────────
# • Very noisy GM with many negative voxels  → high neg_gm_cbf, low QEI
# • GM mean ≈ WM mean                        → GM/WM ratio ≈ 1 (FAIL, needs 2–3)
# • Huge spatial variance                    → sCoV >> 0.42 (FAIL)
cbf = np.zeros(shape, dtype=np.float64)
cbf[gm_mask] = 15 + rng.normal(0, 40, size=int(np.sum(gm_mask)))   # very noisy, lots of negatives
cbf[wm_mask] = 15 + rng.normal(0, 20, size=int(np.sum(wm_mask)))   # same mean as GM

# Write NIfTI files into a temporary ExploreASL-like directory
tmp_root = Path(tempfile.mkdtemp(prefix="xasl_fail_demo_"))
subject_id = "sub-bad01"
subject_dir = tmp_root / subject_id
asl_dir = subject_dir / "ASL_1"
t1w_dir = subject_dir / "T1w"
asl_dir.mkdir(parents=True)
t1w_dir.mkdir(parents=True)

nib.save(nib.Nifti1Image(cbf, affine),
         str(asl_dir / f"{subject_id}_ASL_1_CBF.nii.gz"))
nib.save(nib.Nifti1Image(gm_mask.astype(np.float32), affine),
         str(t1w_dir / f"{subject_id}_T1w_pGM.nii.gz"))
nib.save(nib.Nifti1Image(wm_mask.astype(np.float32), affine),
         str(t1w_dir / f"{subject_id}_T1w_pWM.nii.gz"))
nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),
         str(t1w_dir / f"{subject_id}_T1w_BrainMask.nii.gz"))

print(f"   Temp directory: {tmp_root}")
print(f"   Subject:        {subject_id}")
print(f"   GM voxels:      {int(np.sum(gm_mask))}")
print(f"   WM voxels:      {int(np.sum(wm_mask))}")

# ── Step 2: ExploreASL Discovery & Validation ─────────────────────
print("\nStep 2: Running ExploreASL discovery & validation...")

from asl_qc.exploreasl.discovery import discover_exploreasl_outputs
from asl_qc.exploreasl.validator import validate_exploreasl_outputs

paths = discover_exploreasl_outputs(subject_dir, subject_id=subject_id)
val = validate_exploreasl_outputs(paths)

print(f"   CBF map found:  {paths.cbf_map is not None}")
print(f"   GM mask found:  {paths.gm_mask is not None}")
print(f"   WM mask found:  {paths.wm_mask is not None}")
print(f"   Brain mask:     {paths.brain_mask is not None}")
print(f"   Validation OK:  {val.is_valid}")
if val.warnings:
    for w in val.warnings:
        print(f"   [WARN] {w}")

# ── Step 3: Load data back from the discovered paths ──────────────
print("\nStep 3: Loading NIfTI data from ExploreASL outputs...")

cbf_img = nib.load(str(paths.cbf_map))
cbf_data = np.asarray(cbf_img.dataobj, dtype=np.float64)

gm_img = nib.load(str(paths.gm_mask))
gm_data = np.asarray(gm_img.dataobj).astype(bool)

wm_img = nib.load(str(paths.wm_mask))
wm_data = np.asarray(wm_img.dataobj).astype(bool)

print(f"   CBF shape:  {cbf_data.shape}")
print(f"   GM voxels:  {int(np.sum(gm_data))}")
print(f"   WM voxels:  {int(np.sum(wm_data))}")

# ── Step 4: Compute QC Metrics ────────────────────────────────────
print("\nStep 4: Computing QC metrics on bad data...\n")

from asl_qc.metrics.qei import compute_qei
from asl_qc.metrics.scov import compute_scov
from asl_qc.metrics.histogram import analyze_histogram
from asl_qc.metrics.tissue_mask import compute_gm_wm_ratio

qei = compute_qei(cbf_data, gm_data, wm_data)
print(f"   QEI Score:              {qei.qei:.4f}       (threshold: >= 0.53)")
print(f"   Structural Similarity:  {qei.structural_similarity:.4f}")
print(f"   Negative GM Fraction:   {qei.negative_gm_fraction:.4f}  (threshold: <= 0.10)")
print(f"   Mean GM CBF:            {qei.mean_gm_cbf:.1f} ml/100g/min")

scov = compute_scov(cbf_data, gm_data)
print(f"\n   sCoV (GM):              {scov:.4f}       (threshold: <= 0.42)")

hist = analyze_histogram(cbf_data, gm_data)
print(f"\n   Histogram Skewness:     {hist.skewness:.4f}")
print(f"   Histogram Kurtosis:     {hist.kurtosis:.4f}")
print(f"   5th Percentile:         {hist.percentile_5:.1f}")
print(f"   95th Percentile:        {hist.percentile_95:.1f}")

ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf_data, gm_data, wm_data)
print(f"\n   GM/WM CBF Ratio:        {ratio:.2f}       (threshold: 2.0 – 3.0)")
print(f"   Mean GM CBF:            {mean_gm:.1f}")
print(f"   Mean WM CBF:            {mean_wm:.1f}")

# ── Step 5: Apply Empirical Thresholds ────────────────────────────
print("\nStep 5: Applying quality thresholds...\n")

from asl_qc.thresholds.empirical import apply_empirical_thresholds

# Simulated bad motion (FD > 0.5 mm) and bad registration (Dice < 0.70)
metrics = {
    "qei": qei.qei,
    "scov_gm": scov,
    "mean_fd": 0.85,      # high motion → FAIL
    "gm_wm_ratio": ratio,
    "neg_gm_cbf": qei.negative_gm_fraction,
    "dice": 0.55,          # poor registration → FAIL
}

result = apply_empirical_thresholds(metrics)

for v in result.verdicts:
    status = "[PASS]" if v.passed else "[FAIL]"
    print(f"   {v.metric_name:25s}  {v.value:8.4f}  {v.operator} {v.threshold:.3f}  {status}")

print(f"\n   {'[PASS] OVERALL PASS' if result.overall_pass else '[FAIL] OVERALL FAIL'}")
print(f"   {result.n_passed}/{result.n_passed + result.n_failed} metrics passed")

# ── Step 6: ML Outlier Detection ──────────────────────────────────
print("\nStep 6: Training ML outlier detector...\n")

from asl_qc.thresholds.ml_outlier import (
    build_feature_vector,
    build_feature_matrix,
    train_outlier_model,
    predict_outlier,
)

# Normative population (good data)
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

our_metrics = {
    **metrics,
    "skewness": hist.skewness,
    "kurtosis": hist.kurtosis,
    "tsnr": 15,              # terrible tSNR
    "rms_difference": 25,    # very high motion RMS
}
pred = predict_outlier(build_feature_vector(our_metrics), model)
print(f"   ML Verdict:  {'[ALERT] OUTLIER' if pred.is_outlier else '[PASS] INLIER'}")
print(f"   Anomaly Score: {pred.anomaly_score:.4f}")

# ── Step 7: Generate HTML Report ──────────────────────────────────
print("\nStep 7: Generating HTML report...")

from asl_qc.reporting.html_report import generate_html_report
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

print("   Rendering slice images...")

images = [
    {
        "title": "CBF Map",
        "base64": _render_slice(cbf_data, "CBF Map (ml/100g/min)",
                                cmap="RdBu_r", vmin=-60, vmax=80,
                                mask_overlay=gm_data.astype(float)),
        "caption": "CBF with GM contour (green). Note noisy values and many negatives.",
    },
    {
        "title": "GM Mask",
        "base64": _render_slice(gm_data.astype(float), "Grey Matter Mask",
                                cmap="Greens", vmin=0, vmax=1),
        "caption": "Binary GM tissue mask used for QC metrics.",
    },
    {
        "title": "WM Mask",
        "base64": _render_slice(wm_data.astype(float), "White Matter Mask",
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
gm_vals = cbf_data[gm_data]
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
               f"35% of GM voxels are negative (red dashed line).",
})

report_path = "demo_fail_report.html"

# Build data summary sections to show in the report
data_summary = [
    {
        "title": "Synthetic Brain Geometry",
        "rows": [
            {"name": "Volume shape", "value": f"{shape}"},
            {"name": "Total brain voxels", "value": f"{int(np.sum(mask))}"},
            {"name": "GM voxels", "value": f"{int(np.sum(gm_data))}"},
            {"name": "WM voxels", "value": f"{int(np.sum(wm_data))}"},
        ],
    },
    {
        "title": "CBF Map Statistics (Intentionally Bad)",
        "rows": [
            {"name": "GM CBF generation", "value": "15 ± 40 ml/100g/min (very noisy, many negatives)"},
            {"name": "WM CBF generation", "value": "15 ± 20 ml/100g/min (same mean as GM)"},
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
            {"name": "Mean FD (framewise displacement)", "value": "0.850 mm (high motion)"},
            {"name": "Dice coefficient (registration)", "value": "0.550 (poor alignment)"},
            {"name": "tSNR", "value": "15 (very low)"},
            {"name": "RMS difference", "value": "25 (very high)"},
        ],
    },
    {
        "title": "ExploreASL Discovery",
        "rows": [
            {"name": "CBF map", "value": str(paths.cbf_map)},
            {"name": "GM mask", "value": str(paths.gm_mask)},
            {"name": "WM mask", "value": str(paths.wm_mask)},
            {"name": "Brain mask", "value": str(paths.brain_mask)},
            {"name": "Validation passed", "value": str(val.is_valid)},
            {"name": "Missing optional", "value": ", ".join(paths.missing_optional) or "None"},
        ],
    },
]

generate_html_report(
    subject_id=subject_id,
    timestamp=datetime.now().isoformat(),
    overall_pass=result.overall_pass,
    verdicts=[{
        "metric_name": v.metric_name,
        "value": v.value,
        "threshold": v.threshold,
        "operator": v.operator,
        "passed": v.passed,
    } for v in result.verdicts],
    summary_stats=[
        {"value": f"{qei.qei:.3f}", "label": "QEI"},
        {"value": f"{scov:.3f}", "label": "sCoV (GM)"},
        {"value": f"{metrics['mean_fd']:.3f} mm", "label": "Mean FD"},
        {"value": f"{ratio:.2f}", "label": "GM/WM Ratio"},
    ],
    input_files={
        "cbf_map": str(paths.cbf_map),
        "gm_mask": str(paths.gm_mask),
        "wm_mask": str(paths.wm_mask),
        "brain_mask": str(paths.brain_mask),
        "source": "ExploreASL discovery",
    },
    ml_verdict={
        "method": "isolation_forest",
        "anomaly_score": pred.anomaly_score,
        "is_outlier": pred.is_outlier,
    },
    data_summary=data_summary,
    images=images,
    output_path=report_path,
)
print(f"   Report: file://{os.path.abspath(report_path)}")

# ── Cleanup ───────────────────────────────────────────────────────
shutil.rmtree(tmp_root)
print(f"   Cleaned up temp directory: {tmp_root}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[FAIL] Demo complete -- data INTENTIONALLY failed quality checks.")
print("=" * 60)
print(f"\n   Failed metrics: {result.n_failed}/{result.n_passed + result.n_failed}")
print(f"   ML outlier:     {'Yes' if pred.is_outlier else 'No'}")
print(f"   Report saved:   {report_path}")
print("\nThis demo shows how the toolbox flags low-quality ASL data.")
