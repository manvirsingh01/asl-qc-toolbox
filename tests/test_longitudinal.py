"""Tests for longitudinal QC tracking."""

import json
from pathlib import Path

import numpy as np

from asl_qc.reporting.longitudinal import LongitudinalTracker


def test_tracker_first_session(tmp_path):
    """First session returns INFO alert (need >= 3 sessions)."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)
    summary = tracker.add_session(
        subject_id="sub-01",
        session_id="ses-01",
        metrics={"qei": 0.75, "scov_gm": 0.28},
        overall_pass=True,
    )
    assert summary.n_sessions == 1
    assert len(summary.alerts) == 1
    assert summary.alerts[0].severity == "INFO"


def test_tracker_save_load(tmp_path):
    """Tracker persists and loads data correctly."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)
    tracker.add_session("sub-01", "ses-01", {"qei": 0.75}, True)
    tracker.save()

    # Reload
    tracker2 = LongitudinalTracker(db_path)
    assert "sub-01" in tracker2._records
    assert len(tracker2._records["sub-01"]) == 1


def test_tracker_trend_analysis(tmp_path):
    """3+ sessions triggers trend analysis."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)

    for i in range(5):
        tracker.add_session(
            "sub-01", f"ses-{i:02d}",
            {"qei": 0.75, "scov_gm": 0.28, "mean_fd": 0.2},
            True,
        )

    summary = tracker.add_session(
        "sub-01", "ses-05",
        {"qei": 0.75, "scov_gm": 0.28, "mean_fd": 0.2},
        True,
    )
    assert summary.n_sessions == 6
    # Stable metrics should not generate WARNING or CRITICAL
    warning_critical = [
        a for a in summary.alerts
        if a.severity in ("WARNING", "CRITICAL")
    ]
    assert len(warning_critical) == 0


def test_tracker_declining_metric(tmp_path):
    """Declining QEI over sessions triggers alert."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)

    # QEI declining rapidly
    for i in range(5):
        tracker.add_session(
            "sub-01", f"ses-{i:02d}",
            {"qei": 0.80 - i * 0.08},
            True,
        )

    summary = tracker._analyse("sub-01")
    qei_alerts = [a for a in summary.alerts if a.metric == "qei"]
    assert len(qei_alerts) > 0
    assert qei_alerts[0].severity in ("WARNING", "CRITICAL")


def test_tracker_increasing_metric(tmp_path):
    """Increasing sCoV over sessions triggers alert."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)

    for i in range(5):
        tracker.add_session(
            "sub-01", f"ses-{i:02d}",
            {"scov_gm": 0.25 + i * 0.06},
            True,
        )

    summary = tracker._analyse("sub-01")
    scov_alerts = [a for a in summary.alerts if a.metric == "scov_gm"]
    assert len(scov_alerts) > 0


def test_tracker_multiple_subjects(tmp_path):
    """Tracker handles multiple subjects independently."""
    db_path = tmp_path / "qc_db.json"
    tracker = LongitudinalTracker(db_path)

    tracker.add_session("sub-01", "ses-01", {"qei": 0.75}, True)
    tracker.add_session("sub-02", "ses-01", {"qei": 0.60}, True)
    tracker.save()

    tracker2 = LongitudinalTracker(db_path)
    assert len(tracker2._records["sub-01"]) == 1
    assert len(tracker2._records["sub-02"]) == 1
