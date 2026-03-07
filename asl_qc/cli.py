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
        default=None,
        help="Path to BIDS-formatted ASL subject directory.",
    )
    p.add_argument(
        "--exploreasl-dir", "-e",
        default=None,
        help=(
            "Path to ExploreASL derivatives directory for this subject "
            "(e.g. /derivatives/ExploreASL/sub-01). "
            "When provided, all input files are auto-discovered. "
            "Individual --cbf-map etc. flags override auto-discovered paths."
        ),
    )
    p.add_argument(
        "--field-strength",
        type=float,
        default=3.0,
        help="Scanner field strength in Tesla (default: 3.0). "
             "Used for adaptive threshold selection.",
    )
    p.add_argument(
        "--labeling-type",
        default="PCASL",
        choices=["PCASL", "PASL", "CASL"],
        help="ASL labeling type (default: PCASL). "
             "Auto-detected from BIDS sidecar if available.",
    )
    p.add_argument(
        "--session-id",
        default=None,
        help="Session identifier for longitudinal tracking.",
    )
    p.add_argument(
        "--wmh-mask",
        default=None,
        help="Path to white matter hyperintensity mask (NIfTI). "
             "Auto-discovered from ExploreASL if --exploreasl-dir used.",
    )
    p.add_argument(
        "--att-map",
        default=None,
        help="Path to arterial transit time map (NIfTI). "
             "Available from ExploreASL multi-PLD processing.",
    )
    p.add_argument(
        "--cbf-pvc-map",
        default=None,
        help="Path to PVC-corrected CBF map (NIfTI). "
             "ExploreASL produces this as CBF_PVC2.nii.gz.",
    )
    p.add_argument(
        "--normative-comparison",
        action="store_true",
        help="Compare metrics against ENIGMA-ASL normative database.",
    )
    p.add_argument(
        "--longitudinal-db",
        default=None,
        help="Path to longitudinal QC database JSON for session tracking.",
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
    input_dir: Optional[str] = None,
    output_dir: str = "./qc_output",
    config_path: Optional[str] = None,
    cbf_map_path: Optional[str] = None,
    gm_mask_path: Optional[str] = None,
    wm_mask_path: Optional[str] = None,
    brain_mask_path: Optional[str] = None,
    motion_params_path: Optional[str] = None,
    subject_id: Optional[str] = None,
    skip_ml: bool = False,
    verbose: bool = True,
    exploreasl_dir: Optional[str] = None,
    field_strength: float = 3.0,
    labeling_type: str = "PCASL",
    session_id: Optional[str] = None,
    wmh_mask_path: Optional[str] = None,
    att_map_path: Optional[str] = None,
    cbf_pvc_path: Optional[str] = None,
    normative_comparison: bool = False,
    longitudinal_db: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full ASL QC pipeline.

    Parameters
    ----------
    input_dir : str, optional
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
    exploreasl_dir : str, optional
        Path to ExploreASL derivatives subject directory.
    field_strength : float
        Scanner field strength in Tesla.
    labeling_type : str
        ASL labeling type (PCASL, PASL, CASL).
    session_id : str, optional
        Session identifier for longitudinal tracking.
    wmh_mask_path : str, optional
        Path to WMH mask (NIfTI).
    att_map_path : str, optional
        Path to ATT map (NIfTI).
    cbf_pvc_path : str, optional
        Path to PVC-corrected CBF map (NIfTI).
    normative_comparison : bool
        Compare against ENIGMA-ASL normative database.
    longitudinal_db : str, optional
        Path to longitudinal QC database JSON.

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

    inp = Path(input_dir) if input_dir else None

    if subject_id is None:
        if exploreasl_dir:
            subject_id = Path(exploreasl_dir).name
        elif inp:
            subject_id = inp.name
        else:
            subject_id = "unknown"

    _log(f"ASL QC Toolbox v{cfg.version}", verbose)
    if inp:
        _log(f"Input:  {inp}", verbose)
    _log(f"Output: {out}", verbose)

    results: Dict[str, Any] = {}
    input_files: Dict[str, str] = {}
    if inp:
        input_files["input_dir"] = str(inp)

    # ---- ExploreASL auto-discovery ----
    if exploreasl_dir is not None:
        from .exploreasl.discovery import discover_exploreasl_outputs, load_exploreasl_qc
        _log(f"Auto-discovering ExploreASL outputs from: {exploreasl_dir}", verbose)

        xasl_paths = discover_exploreasl_outputs(
            exploreasl_dir,
            subject_id=subject_id,
        )

        if xasl_paths.missing_required:
            _log(
                f"WARNING: Missing required files: {xasl_paths.missing_required}",
                verbose,
            )

        # Use auto-discovered paths unless manually overridden
        if cbf_map_path is None and xasl_paths.cbf_map:
            cbf_map_path = str(xasl_paths.cbf_map)
        if gm_mask_path is None and xasl_paths.gm_mask:
            gm_mask_path = str(xasl_paths.gm_mask)
        if wm_mask_path is None and xasl_paths.wm_mask:
            wm_mask_path = str(xasl_paths.wm_mask)
        if brain_mask_path is None and xasl_paths.brain_mask:
            brain_mask_path = str(xasl_paths.brain_mask)
        if motion_params_path is None and xasl_paths.motion_params:
            motion_params_path = str(xasl_paths.motion_params)
        if wmh_mask_path is None and xasl_paths.wmh_mask:
            wmh_mask_path = str(xasl_paths.wmh_mask)
        if att_map_path is None and xasl_paths.att_map:
            att_map_path = str(xasl_paths.att_map)
        if cbf_pvc_path is None and xasl_paths.cbf_pvc_map:
            cbf_pvc_path = str(xasl_paths.cbf_pvc_map)

        # Auto-detect labeling type from BIDS sidecar
        if xasl_paths.asl_json:
            from .bids import parse_asl_metadata
            try:
                meta = parse_asl_metadata(xasl_paths.asl_json)
                if meta.labeling_type and labeling_type == "PCASL":
                    labeling_type = meta.labeling_type
                    _log(f"Auto-detected labeling type: {labeling_type}", verbose)
                if meta.magnetic_field_strength and field_strength == 3.0:
                    field_strength = meta.magnetic_field_strength
                    _log(f"Auto-detected field strength: {field_strength}T", verbose)
            except Exception:
                pass

        # Load ExploreASL's own QC values for provenance
        if xasl_paths.xasl_qc_json:
            xasl_qc_values = load_exploreasl_qc(xasl_paths.xasl_qc_json)
            results["exploreasl_qc"] = xasl_qc_values
            _log("Loaded ExploreASL QC collection.", verbose)

        input_files["exploreasl_dir"] = str(exploreasl_dir)

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

    # GM/WM ratio (with optional WMH correction)
    wmh_data = None
    if wmh_mask_path:
        wmh_data, _, _ = load_nifti(wmh_mask_path)
        wmh_data = (wmh_data > 0.5).astype(bool)
        input_files["wmh_mask"] = wmh_mask_path

    if cbf_data is not None and gm_data is not None and wm_data is not None:
        ratio, mean_gm, mean_wm = compute_gm_wm_ratio(cbf_data, gm_data, wm_data)
        results["tissue_mask"] = {
            "gm_wm_cbf_ratio": ratio,
            "mean_gm_cbf": mean_gm,
            "mean_wm_cbf": mean_wm,
        }
        metrics_flat["gm_wm_ratio"] = ratio

        # WMH-corrected GM/WM ratio
        if wmh_data is not None:
            wm_no_wmh = wm_data & ~wmh_data
            if np.sum(wm_no_wmh) > 0:
                ratio_corr, mean_gm_c, mean_wm_c = compute_gm_wm_ratio(
                    cbf_data, gm_data, wm_no_wmh
                )
                results["tissue_mask"]["gm_wm_cbf_ratio_wmh_corrected"] = ratio_corr
                results["tissue_mask"]["mean_wm_cbf_wmh_corrected"] = mean_wm_c
                metrics_flat["gm_wm_ratio"] = ratio_corr  # use corrected as primary
                _log(f"  GM/WM ratio (WMH-corrected) = {ratio_corr:.4f}", verbose)

    # ATT QC (if ATT map available)
    if att_map_path:
        from .metrics.att import compute_att_qc
        _log("Computing ATT QC...", verbose)
        att_data, _, _ = load_nifti(att_map_path)
        input_files["att_map"] = att_map_path
        att_gm = gm_data if gm_data is not None else (brain_data if brain_data is not None else None)
        if att_gm is not None:
            att_result = compute_att_qc(att_data, att_gm)
            results["att"] = {
                "mean_att_gm": att_result.mean_att_gm,
                "att_std_gm": att_result.att_std_gm,
                "transit_artifact_fraction": att_result.transit_artifact_fraction,
                "long_att_fraction": att_result.long_att_fraction,
                "att_scov": att_result.att_scov,
                "is_att_problematic": att_result.is_att_problematic,
                "message": att_result.message,
            }
            metrics_flat["mean_att_gm"] = att_result.mean_att_gm
            metrics_flat["transit_artifact_fraction"] = att_result.transit_artifact_fraction
            _log(f"  Mean GM ATT = {att_result.mean_att_gm:.3f}s", verbose)

    # PVC impact assessment (if PVC map available)
    if cbf_pvc_path and cbf_data is not None:
        from .pvc import compute_pvc_qc
        _log("Computing PVC QC...", verbose)
        cbf_pvc_data, _, _ = load_nifti(cbf_pvc_path)
        input_files["cbf_pvc_map"] = cbf_pvc_path
        if gm_mask_path and wm_mask_path:
            gm_prob, _, _ = load_nifti(gm_mask_path)
            wm_prob, _, _ = load_nifti(wm_mask_path)
            pvc_result = compute_pvc_qc(cbf_data, cbf_pvc_data, gm_prob, wm_prob)
            results["pvc"] = {
                "gm_cbf_uncorrected": pvc_result.gm_cbf_uncorrected,
                "gm_cbf_pvc": pvc_result.gm_cbf_pvc,
                "wm_cbf_uncorrected": pvc_result.wm_cbf_uncorrected,
                "wm_cbf_pvc": pvc_result.wm_cbf_pvc,
                "ratio_uncorrected": pvc_result.ratio_uncorrected,
                "ratio_pvc": pvc_result.ratio_pvc,
                "pvc_impact_gm": pvc_result.pvc_impact_gm,
                "pvc_impact_ratio": pvc_result.pvc_impact_ratio,
                "pvc_needed": pvc_result.pvc_needed,
                "message": pvc_result.message,
            }
            _log(f"  PVC impact on GM CBF: {pvc_result.pvc_impact_gm:+.1f} ml/100g/min", verbose)

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

    # ---- Threshold evaluation (adaptive) ----
    from .thresholds.normative import get_adaptive_thresholds
    adaptive_thresholds = get_adaptive_thresholds(field_strength, labeling_type)
    _log(
        f"Using adaptive thresholds for {field_strength}T {labeling_type}",
        verbose,
    )
    emp_result = apply_empirical_thresholds(metrics_flat, thresholds=adaptive_thresholds)
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

    # ---- Normative z-score comparison ----
    if normative_comparison:
        from .thresholds.normative import compute_normative_zscores
        norm_result = compute_normative_zscores(
            metrics_flat, field_strength, labeling_type
        )
        results["normative"] = {
            "acquisition_key": norm_result.acquisition_key,
            "z_scores": norm_result.z_scores,
            "percentiles": norm_result.percentiles,
            "n_outlier_metrics": norm_result.n_outlier_metrics,
            "verdict": norm_result.normative_verdict,
        }
        _log(
            f"Normative verdict: {norm_result.normative_verdict} "
            f"({norm_result.n_outlier_metrics} outlier metrics)",
            verbose,
        )

    # ---- Longitudinal tracking ----
    if longitudinal_db and session_id:
        from .reporting.longitudinal import LongitudinalTracker
        _log("Updating longitudinal QC database...", verbose)
        tracker = LongitudinalTracker(longitudinal_db)
        from .thresholds.normative import get_acquisition_key
        acq_key = get_acquisition_key(field_strength, labeling_type)
        long_summary = tracker.add_session(
            subject_id=subject_id,
            session_id=session_id,
            metrics=metrics_flat,
            overall_pass=emp_result.overall_pass,
            acquisition_key=acq_key,
        )
        tracker.save()
        results["longitudinal"] = {
            "n_sessions": long_summary.n_sessions,
            "metric_trends": long_summary.metric_trends,
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "metric": a.metric,
                    "message": a.message,
                    "severity": a.severity,
                }
                for a in long_summary.alerts
            ],
        }
        for alert in long_summary.alerts:
            _log(f"  [{alert.severity}] {alert.message}", verbose)

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
            exploreasl_dir=args.exploreasl_dir,
            field_strength=args.field_strength,
            labeling_type=args.labeling_type,
            session_id=args.session_id,
            wmh_mask_path=args.wmh_mask,
            att_map_path=args.att_map,
            cbf_pvc_path=args.cbf_pvc_map,
            normative_comparison=args.normative_comparison,
            longitudinal_db=args.longitudinal_db,
        )
    except Exception as exc:
        print(f"[asl-qc] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
