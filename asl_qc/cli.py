"""
CLI entry point for the ASL QC Toolbox.

Orchestrates the full pipeline: BIDS parsing → preprocessing checks →
QC metric computation → threshold evaluation → report generation.

Usage::

    asl-qc --input /path/to/bids/sub-01 --output-dir ./qc_output
    asl-qc --config custom_config.yaml --input /path/to/bids/sub-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asl-qc",
        description="Advanced Quality Control Toolbox for ASL MRI",
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to BIDS-formatted ASL subject directory.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="./qc_output",
        help="Directory for QC reports (default: ./qc_output).",
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Path to custom YAML configuration file.",
    )
    p.add_argument(
        "--cbf-map",
        default=None,
        help="Path to pre-computed CBF map (NIfTI). If not provided, "
             "the pipeline will look for standard BIDS derivatives.",
    )
    p.add_argument(
        "--gm-mask",
        default=None,
        help="Path to GM probability/binary mask (NIfTI).",
    )
    p.add_argument(
        "--wm-mask",
        default=None,
        help="Path to WM probability/binary mask (NIfTI).",
    )
    p.add_argument(
        "--brain-mask",
        default=None,
        help="Path to brain mask (NIfTI). If absent, one is generated from M0.",
    )
    p.add_argument(
        "--motion-params",
        default=None,
        help="Path to motion parameters file (N×6 text).",
    )
    p.add_argument(
        "--subject-id",
        default=None,
        help="Subject identifier for the report.",
    )
    p.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML-based outlier detection.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output.",
    )
    return p


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[asl-qc] {msg}", flush=True)


def run_pipeline(
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    cbf_map_path: Optional[str] = None,
    gm_mask_path: Optional[str] = None,
    wm_mask_path: Optional[str] = None,
    brain_mask_path: Optional[str] = None,
    motion_params_path: Optional[str] = None,
    subject_id: Optional[str] = None,
    skip_ml: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full ASL QC pipeline.

    Parameters
    ----------
    input_dir : str
        BIDS subject directory.
    output_dir : str
        Output directory for reports.
    config_path : str, optional
        Custom YAML config.
    cbf_map_path, gm_mask_path, wm_mask_path, brain_mask_path : str, optional
        Explicit paths to NIfTI inputs.
    motion_params_path : str, optional
        Path to N×6 motion parameters text file.
    subject_id : str, optional
        Subject label for the report.
    skip_ml : bool
        Skip ML outlier detection.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Complete results dictionary.
    """
    from .config import load_config
    from .io_utils import load_nifti, generate_brain_mask
    from .metrics.qei import compute_qei
    from .metrics.scov import compute_scov
    from .metrics.histogram import analyze_histogram
    from .metrics.snr import compute_temporal_snr
    from .metrics.tissue_mask import compute_dice, compute_gm_wm_ratio
    from .thresholds.empirical import apply_empirical_thresholds
    from .reporting.json_report import generate_json_report
    from .reporting.html_report import generate_html_report

    # ---- Setup ----
    cfg = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp = Path(input_dir)

    if subject_id is None:
        subject_id = inp.name

    _log(f"ASL QC Toolbox v{cfg.version}", verbose)
    _log(f"Input:  {inp}", verbose)
    _log(f"Output: {out}", verbose)

    results: Dict[str, Any] = {}
    input_files: Dict[str, str] = {"input_dir": str(inp)}

    # ---- Load CBF map ----
    if cbf_map_path:
        cbf_data, cbf_aff, cbf_hdr = load_nifti(cbf_map_path)
        input_files["cbf_map"] = cbf_map_path
        _log(f"Loaded CBF map: {cbf_map_path} {cbf_data.shape}", verbose)
    else:
        _log("No CBF map provided — skipping CBF-dependent metrics.", verbose)
        cbf_data = None

    # ---- Load masks ----
    gm_data = wm_data = brain_data = None
    if gm_mask_path:
        gm_data, _, _ = load_nifti(gm_mask_path)
        gm_data = (gm_data >= cfg.metrics.tissue_mask.gm_threshold).astype(bool)
        input_files["gm_mask"] = gm_mask_path
    if wm_mask_path:
        wm_data, _, _ = load_nifti(wm_mask_path)
        wm_data = (wm_data >= cfg.metrics.tissue_mask.wm_threshold).astype(bool)
        input_files["wm_mask"] = wm_mask_path
    if brain_mask_path:
        brain_data, _, _ = load_nifti(brain_mask_path)
        brain_data = brain_data.astype(bool)
        input_files["brain_mask"] = brain_mask_path

    # ---- Compute metrics ----
    metrics_flat: Dict[str, float] = {}

    # QEI
    if cbf_data is not None and gm_data is not None and wm_data is not None:
        _log("Computing QEI...", verbose)
        qei_result = compute_qei(
            cbf_data, gm_data, wm_data,
            gm_weight=cfg.metrics.qei.gm_weight,
            wm_weight=cfg.metrics.qei.wm_weight,
            alpha=cfg.metrics.qei.alpha,
            beta=cfg.metrics.qei.beta,
            gamma=cfg.metrics.qei.gamma,
        )
        results["qei"] = {
            "qei": qei_result.qei,
            "structural_similarity": qei_result.structural_similarity,
            "index_of_dispersion": qei_result.index_of_dispersion,
            "negative_gm_fraction": qei_result.negative_gm_fraction,
            "mean_gm_cbf": qei_result.mean_gm_cbf,
        }
        metrics_flat["qei"] = qei_result.qei
        metrics_flat["neg_gm_cbf"] = qei_result.negative_gm_fraction
        _log(f"  QEI = {qei_result.qei:.4f}", verbose)

    # sCoV
    roi_mask = gm_data if gm_data is not None else brain_data
    if cbf_data is not None and roi_mask is not None:
        _log("Computing sCoV...", verbose)
        scov_val = compute_scov(cbf_data, roi_mask)
        results["scov"] = {"global_scov_gm": scov_val}
        metrics_flat["scov_gm"] = scov_val
        _log(f"  sCoV(GM) = {scov_val:.4f}", verbose)

    # Histogram
    if cbf_data is not None and roi_mask is not None:
        _log("Computing histogram metrics...", verbose)
        hist = analyze_histogram(cbf_data, roi_mask)
        results["histogram"] = {
            "mean": hist.mean,
            "median": hist.median,
            "std": hist.std,
            "skewness": hist.skewness,
            "kurtosis": hist.kurtosis,
            "percentile_5": hist.percentile_5,
            "percentile_95": hist.percentile_95,
            "iqr": hist.iqr,
            "n_voxels": hist.n_voxels,
        }
        metrics_flat["skewness"] = hist.skewness
        metrics_flat["kurtosis"] = hist.kurtosis

    # GM/WM ratio
    if cbf_data is not None and gm_data is not None and wm_data is not None:
        ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf_data, gm_data, wm_data)
        results["tissue_mask"] = {
            "gm_wm_cbf_ratio": ratio,
            "mean_gm_cbf": mean_gm,
            "mean_wm_cbf": mean_wm,
        }
        metrics_flat["gm_wm_ratio"] = ratio

    # Motion
    if motion_params_path:
        from .preprocessing.motion import compute_framewise_displacement, summarize_motion
        _log("Computing motion metrics...", verbose)
        mp = np.loadtxt(motion_params_path)
        fd = compute_framewise_displacement(mp, radius_mm=cfg.preprocessing.motion.fd_radius_mm)
        ms = summarize_motion(
            fd, np.zeros_like(fd),
            fd_spike_threshold=cfg.preprocessing.motion.fd_spike_threshold_mm,
        )
        results["motion"] = {
            "mean_fd": ms.mean_fd,
            "max_fd": ms.max_fd,
            "n_fd_spikes": ms.n_fd_spikes,
        }
        metrics_flat["mean_fd"] = ms.mean_fd
        _log(f"  Mean FD = {ms.mean_fd:.4f} mm", verbose)

    # ---- Threshold evaluation ----
    _log("Applying empirical thresholds...", verbose)
    emp_result = apply_empirical_thresholds(metrics_flat)
    results["empirical_thresholds"] = {
        "overall_pass": emp_result.overall_pass,
        "n_passed": emp_result.n_passed,
        "n_failed": emp_result.n_failed,
        "verdicts": [
            {
                "metric_name": v.metric_name,
                "value": v.value,
                "threshold": v.threshold,
                "operator": v.operator,
                "passed": v.passed,
            }
            for v in emp_result.verdicts
        ],
    }

    # ---- Reports ----
    timestamp = datetime.now(timezone.utc).isoformat()

    if cfg.reporting.json:
        json_path = generate_json_report(
            metrics=results,
            verdicts=results.get("empirical_thresholds", {}),
            input_files=input_files,
            config_path=config_path,
            output_path=out / f"{subject_id}_qc_report.json",
        )
        _log(f"JSON report: {json_path}", verbose)

    if cfg.reporting.html:
        summary_stats = []
        if "qei" in results:
            summary_stats.append({"value": f"{results['qei']['qei']:.3f}", "label": "QEI"})
        if "scov" in results:
            summary_stats.append({"value": f"{results['scov']['global_scov_gm']:.3f}", "label": "sCoV (GM)"})
        if "motion" in results:
            summary_stats.append({"value": f"{results['motion']['mean_fd']:.3f} mm", "label": "Mean FD"})
        if "tissue_mask" in results:
            summary_stats.append({"value": f"{results['tissue_mask']['gm_wm_cbf_ratio']:.2f}", "label": "GM/WM Ratio"})

        html_path = generate_html_report(
            subject_id=subject_id,
            timestamp=timestamp,
            overall_pass=emp_result.overall_pass,
            verdicts=results["empirical_thresholds"]["verdicts"],
            summary_stats=summary_stats,
            input_files=input_files,
            output_path=out / f"{subject_id}_qc_report.html",
        )
        _log(f"HTML report: {html_path}", verbose)

    _log("Pipeline complete.", verbose)
    return results


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            input_dir=args.input,
            output_dir=args.output_dir,
            config_path=args.config,
            cbf_map_path=args.cbf_map,
            gm_mask_path=args.gm_mask,
            wm_mask_path=args.wm_mask,
            brain_mask_path=args.brain_mask,
            motion_params_path=args.motion_params,
            subject_id=args.subject_id,
            skip_ml=args.skip_ml,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"[asl-qc] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
