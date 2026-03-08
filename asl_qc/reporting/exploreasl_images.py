"""
ExploreASL-style QC image renderer.

Produces high-quality (300 DPI) publication-grade brain slice images
matching the ExploreASL QC tutorial output format:

- T1w structural check (Tra_Src_rT1)
- WM segmentation overlay on T1w (Tra_Seg_rT1) — WM shown in red
- CBF map with GM-WM contrast check (Tra_qCBF)
- ASL-to-T1w registration check — WM contour on CBF (Tra_Reg_pWM_qCBF)
- Temporal SD map (Tra_SD)
- Motion parameters plot (rp_motion)
- M0 calibration image (M0_noSmooth)
- GM CBF histogram

Reference: ExploreASL v1.12.0 beta QC Tutorial
https://exploreasl.github.io/Documentation/1.12.0_beta/Tutorials-QC/
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Shared rendering configuration ──────────────────────────────
_DPI = 300
_FIG_BG = "#0f172a"
_PANEL_BG = "#0f172a"
_TEXT_CLR = "#e2e8f0"
_MUTED_CLR = "#94a3b8"
_BORDER_CLR = "#334155"


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render matplotlib figure to base64 PNG at _DPI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _three_plane_slices(volume: np.ndarray, mid=None):
    """Extract sagittal, coronal, axial mid-slices.

    For the sagittal view, use a parasagittal slice (25% from midline)
    to avoid the interhemispheric fissure and show brain parenchyma.
    """
    if mid is None:
        mid = [s // 2 for s in volume.shape]
    # Parasagittal: offset ~25% of the hemisphere width from midline
    sag_idx = mid[0] + max(1, volume.shape[0] // 8)
    return [
        volume[sag_idx, :, :],  # parasagittal (right hemisphere)
        volume[:, mid[1], :],   # coronal
        volume[:, :, mid[2]],   # axial
    ]


def render_t1w_check(t1w: np.ndarray) -> str:
    """Tra_Src_rT1 — T1w structural image in 3 planes.

    ExploreASL Step 2: Check overall T1w image quality.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    slices = _three_plane_slices(t1w)
    labels = ["Sagittal", "Coronal", "Axial"]
    for ax, sl, lbl in zip(axes, slices, labels):
        ax.imshow(sl.T, origin="lower", cmap="gray", interpolation="bicubic",
                  aspect="equal", vmin=0, vmax=np.percentile(t1w[t1w > 0], 99))
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")
    fig.suptitle("T1w Structural Image (MNI Space)", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, top=0.88)
    return _fig_to_base64(fig)


def render_wm_segmentation(t1w: np.ndarray, wm_prob: np.ndarray) -> str:
    """Tra_Seg_rT1 — WM segmentation shown in red overlay on T1w.

    ExploreASL Step 2: WM overlay should tightly follow white matter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    t1_slices = _three_plane_slices(t1w)
    wm_slices = _three_plane_slices(wm_prob)
    labels = ["Sagittal", "Coronal", "Axial"]
    vmax = np.percentile(t1w[t1w > 0], 99)

    for ax, t1sl, wmsl, lbl in zip(axes, t1_slices, wm_slices, labels):
        ax.imshow(t1sl.T, origin="lower", cmap="gray", interpolation="bicubic",
                  aspect="equal", vmin=0, vmax=vmax)
        # Red WM overlay with alpha
        wm_rgba = np.zeros((*wmsl.T.shape, 4))
        wm_rgba[:, :, 0] = 1.0   # red channel
        wm_rgba[:, :, 3] = wmsl.T * 0.45  # alpha tracks WM probability
        ax.imshow(wm_rgba, origin="lower", aspect="equal", interpolation="bicubic")
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("WM Segmentation on T1w (red = WM)", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, top=0.88)
    return _fig_to_base64(fig)


def render_cbf_map(cbf: np.ndarray, gm_prob: np.ndarray,
                   vmin: float = -10, vmax: float = 80) -> str:
    """Tra_qCBF — CBF map with GM contour.

    ExploreASL Step 3: Check GM-WM contrast; GM should be brighter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    cbf_slices = _three_plane_slices(cbf)
    gm_slices = _three_plane_slices(gm_prob)
    labels = ["Sagittal", "Coronal", "Axial"]

    for ax, csl, gmsl, lbl in zip(axes, cbf_slices, gm_slices, labels):
        im = ax.imshow(csl.T, origin="lower", cmap="hot", interpolation="bicubic",
                       aspect="equal", vmin=vmin, vmax=vmax)
        ax.contour(gmsl.T, levels=[0.5], colors="lime", linewidths=0.8, origin="lower")
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("CBF Map (ml/100g/min) — GM contour in green", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED_CLR, labelsize=8)
    cbar.set_label("CBF (ml/100g/min)", color=_MUTED_CLR, fontsize=9)
    fig.subplots_adjust(right=0.92, top=0.88)
    return _fig_to_base64(fig)


def render_registration_check(cbf: np.ndarray, wm_prob: np.ndarray,
                               vmin: float = -10, vmax: float = 80) -> str:
    """Tra_Reg_pWM_qCBF — ASL-to-T1w registration: WM contour on CBF.

    ExploreASL Step 3: WM contour should align with low-CBF regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    cbf_slices = _three_plane_slices(cbf)
    wm_slices = _three_plane_slices(wm_prob)
    labels = ["Sagittal", "Coronal", "Axial"]

    for ax, csl, wmsl, lbl in zip(axes, cbf_slices, wm_slices, labels):
        im = ax.imshow(csl.T, origin="lower", cmap="hot", interpolation="bicubic",
                       aspect="equal", vmin=vmin, vmax=vmax)
        ax.contour(wmsl.T, levels=[0.3], colors="cyan", linewidths=0.8, origin="lower")
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("ASL-T1w Registration — WM contour (cyan) on CBF", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED_CLR, labelsize=8)
    fig.subplots_adjust(right=0.92, top=0.88)
    return _fig_to_base64(fig)


def render_temporal_sd(sd_map: np.ndarray) -> str:
    """Tra_SD — Temporal standard deviation map.

    ExploreASL Step 3: Should be smooth and uniform — no GM/WM contrast.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    slices = _three_plane_slices(sd_map)
    labels = ["Sagittal", "Coronal", "Axial"]
    vmax = np.percentile(sd_map[sd_map > 0], 99) if np.any(sd_map > 0) else 1.0

    for ax, sl, lbl in zip(axes, slices, labels):
        im = ax.imshow(sl.T, origin="lower", cmap="inferno", interpolation="bicubic",
                       aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("Temporal Standard Deviation", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED_CLR, labelsize=8)
    fig.subplots_adjust(right=0.92, top=0.88)
    return _fig_to_base64(fig)


def render_motion_plot(motion_params: np.ndarray, threshold: float = 0.5) -> str:
    """rp_motion — Head motion plot (translations + rotations).

    ExploreASL Step 3: Flat curve = low motion = high quality.
    """
    n_vol = motion_params.shape[0]
    x_axis = np.arange(n_vol)

    # Compute framewise displacement (simplified)
    diff = np.diff(motion_params, axis=0)
    fd = np.sum(np.abs(diff), axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor=_FIG_BG)

    # Panel 1: translations
    trans_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    trans_colors = ["#ef4444", "#22c55e", "#3b82f6"]
    for i, (lbl, clr) in enumerate(zip(trans_labels, trans_colors)):
        axes[0].plot(x_axis, motion_params[:, i], color=clr, linewidth=1.2, label=lbl)
    axes[0].set_ylabel("Translation", color=_TEXT_CLR, fontsize=10)
    axes[0].legend(fontsize=8, facecolor=_FIG_BG, edgecolor=_BORDER_CLR, labelcolor=_TEXT_CLR)
    axes[0].set_title("Head Motion Parameters", color=_TEXT_CLR, fontsize=13, fontweight="bold")

    # Panel 2: rotations
    rot_labels = ["Pitch (deg)", "Roll (deg)", "Yaw (deg)"]
    rot_colors = ["#f59e0b", "#a855f7", "#06b6d4"]
    for i, (lbl, clr) in enumerate(zip(rot_labels, rot_colors)):
        axes[1].plot(x_axis, motion_params[:, i + 3], color=clr, linewidth=1.2, label=lbl)
    axes[1].set_ylabel("Rotation", color=_TEXT_CLR, fontsize=10)
    axes[1].legend(fontsize=8, facecolor=_FIG_BG, edgecolor=_BORDER_CLR, labelcolor=_TEXT_CLR)

    # Panel 3: FD
    axes[2].bar(x_axis[1:], fd, color="#6366f1", alpha=0.8, width=1.0)
    axes[2].axhline(threshold, color="#ef4444", linestyle="--", linewidth=1.5,
                    label=f"FD threshold = {threshold} mm")
    axes[2].set_ylabel("FD (mm)", color=_TEXT_CLR, fontsize=10)
    axes[2].set_xlabel("Volume", color=_TEXT_CLR, fontsize=10)
    axes[2].legend(fontsize=8, facecolor=_FIG_BG, edgecolor=_BORDER_CLR, labelcolor=_TEXT_CLR)

    for ax in axes:
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_MUTED_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_BORDER_CLR)

    fig.tight_layout()
    return _fig_to_base64(fig)


def render_m0_image(m0: np.ndarray) -> str:
    """M0 calibration image — should show smooth, uniform signal.

    ExploreASL Step 3: Brain mask must exclude CSF/extracranial.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    slices = _three_plane_slices(m0)
    labels = ["Sagittal", "Coronal", "Axial"]
    vmax = np.percentile(m0[m0 > 0], 99) if np.any(m0 > 0) else 1.0

    for ax, sl, lbl in zip(axes, slices, labels):
        im = ax.imshow(sl.T, origin="lower", cmap="gray", interpolation="bicubic",
                       aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("M0 Calibration Image", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED_CLR, labelsize=8)
    fig.subplots_adjust(right=0.92, top=0.88)
    return _fig_to_base64(fig)


def render_gm_probability(gm_prob: np.ndarray) -> str:
    """GM probability map — cortical ribbon visualisation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    slices = _three_plane_slices(gm_prob)
    labels = ["Sagittal", "Coronal", "Axial"]

    for ax, sl, lbl in zip(axes, slices, labels):
        im = ax.imshow(sl.T, origin="lower", cmap="hot", interpolation="bicubic",
                       aspect="equal", vmin=0, vmax=1)
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("Grey Matter Probability Map (pGM)", color=_TEXT_CLR,
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED_CLR, labelsize=8)
    fig.subplots_adjust(right=0.92, top=0.88)
    return _fig_to_base64(fig)


def render_brain_mask(brain_mask: np.ndarray) -> str:
    """Brain mask visualisation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=_FIG_BG)
    slices = _three_plane_slices(brain_mask.astype(float))
    labels = ["Sagittal", "Coronal", "Axial"]

    for ax, sl, lbl in zip(axes, slices, labels):
        ax.imshow(sl.T, origin="lower", cmap="bone", interpolation="bicubic",
                  aspect="equal", vmin=0, vmax=1)
        ax.set_title(lbl, color=_TEXT_CLR, fontsize=12)
        ax.axis("off")

    fig.suptitle("Brain Mask", color=_TEXT_CLR, fontsize=14, fontweight="bold")
    fig.subplots_adjust(wspace=0.05, top=0.88)
    return _fig_to_base64(fig)


def render_histogram(cbf_values: np.ndarray, percentile_5: float,
                     percentile_95: float, skewness: float,
                     kurtosis: float, caption_extra: str = "") -> str:
    """GM CBF histogram — distribution quality check."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_FIG_BG)

    ax.hist(cbf_values, bins=60, color="#6366f1", alpha=0.85, edgecolor="#1e293b",
            linewidth=0.5)
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.8, label="CBF = 0")
    ax.axvline(percentile_5, color="#f59e0b", linestyle=":", linewidth=1.3,
               label=f"P5 = {percentile_5:.1f}")
    ax.axvline(percentile_95, color="#22c55e", linestyle=":", linewidth=1.3,
               label=f"P95 = {percentile_95:.1f}")
    ax.axvline(np.mean(cbf_values), color="#06b6d4", linestyle="-", linewidth=1.3,
               label=f"Mean = {np.mean(cbf_values):.1f}")

    ax.set_xlabel("CBF (ml/100g/min)", color=_TEXT_CLR, fontsize=11)
    ax.set_ylabel("Voxel Count", color=_TEXT_CLR, fontsize=11)
    ax.set_title("GM CBF Histogram", color=_TEXT_CLR, fontsize=14, fontweight="bold")
    ax.tick_params(colors=_MUTED_CLR, labelsize=9)
    ax.legend(fontsize=9, facecolor=_FIG_BG, edgecolor=_BORDER_CLR, labelcolor=_TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(_BORDER_CLR)
    ax.set_facecolor(_PANEL_BG)

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_all_qc_images(
    *,
    t1w: np.ndarray,
    cbf: np.ndarray,
    gm_prob: np.ndarray,
    wm_prob: np.ndarray,
    brain_mask: np.ndarray,
    temporal_sd: np.ndarray | None = None,
    motion_params: np.ndarray | None = None,
    m0: np.ndarray | None = None,
    cbf_gm_values: np.ndarray | None = None,
    hist_stats: dict | None = None,
    cbf_vmin: float = -10,
    cbf_vmax: float = 80,
) -> list[dict]:
    """Generate all ExploreASL-style QC images for the HTML report.

    Returns list of dicts with ``title``, ``base64``, ``caption`` keys.
    """
    images = []

    # 1. T1w structural check
    images.append({
        "title": "T1w Structural (Tra_Src_rT1)",
        "base64": render_t1w_check(t1w),
        "caption": "T1w image quality and MNI normalization. Check for "
                   "structural anomalies and consistent orientation.",
    })

    # 2. WM segmentation on T1w
    images.append({
        "title": "WM Segmentation (Tra_Seg_rT1)",
        "base64": render_wm_segmentation(t1w, wm_prob),
        "caption": "WM segmentation in red. Should tightly follow white "
                   "matter boundaries. T1w shows GM (darker) vs WM (brighter).",
    })

    # 3. GM probability map
    images.append({
        "title": "GM Probability Map (pGM)",
        "base64": render_gm_probability(gm_prob),
        "caption": "Probabilistic GM segmentation — cortical ribbon should "
                   "be clearly visible with sulcal folding patterns.",
    })

    # 4. CBF map with GM contour
    images.append({
        "title": "CBF Map (Tra_qCBF)",
        "base64": render_cbf_map(cbf, gm_prob, vmin=cbf_vmin, vmax=cbf_vmax),
        "caption": "CBF with GM contour (green). GM should be visually "
                   "brighter than WM. No bright halo = no motion artifacts.",
    })

    # 5. ASL-T1w registration check
    images.append({
        "title": "Registration Check (Tra_Reg_pWM_qCBF)",
        "base64": render_registration_check(cbf, wm_prob, vmin=cbf_vmin, vmax=cbf_vmax),
        "caption": "WM contour (cyan) on CBF map. WM contour should align "
                   "with low-CBF (dark) regions in the CBF map.",
    })

    # 6. Brain mask
    images.append({
        "title": "Brain Mask",
        "base64": render_brain_mask(brain_mask),
        "caption": "Full brain mask with interhemispheric fissure and ventricles.",
    })

    # 7. Temporal SD (if available)
    if temporal_sd is not None:
        images.append({
            "title": "Temporal SD (Tra_SD)",
            "base64": render_temporal_sd(temporal_sd),
            "caption": "Temporal standard deviation. Should be smooth and "
                       "uniform — GM/WM contrast indicates motion contamination.",
        })

    # 8. Motion parameters (if available)
    if motion_params is not None:
        images.append({
            "title": "Motion Parameters (rp_motion)",
            "base64": render_motion_plot(motion_params),
            "caption": "Head motion: translations, rotations, and framewise "
                       "displacement. Flat curves = low motion = high quality.",
        })

    # 9. M0 calibration (if available)
    if m0 is not None:
        images.append({
            "title": "M0 Calibration (M0_noSmooth)",
            "base64": render_m0_image(m0),
            "caption": "M0 calibration image. Should show smooth, uniform "
                       "signal. Brain mask must exclude CSF/extracranial.",
        })

    # 10. Histogram (if values provided)
    if cbf_gm_values is not None and hist_stats is not None:
        caption = (f"Skewness={hist_stats['skewness']:.3f}, "
                   f"Kurtosis={hist_stats['kurtosis']:.3f}.")
        if hist_stats.get("caption_extra"):
            caption += f" {hist_stats['caption_extra']}"
        images.append({
            "title": "GM CBF Histogram",
            "base64": render_histogram(
                cbf_gm_values,
                hist_stats["percentile_5"],
                hist_stats["percentile_95"],
                hist_stats["skewness"],
                hist_stats["kurtosis"],
            ),
            "caption": caption,
        })

    return images
