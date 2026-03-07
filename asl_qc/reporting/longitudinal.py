"""
Longitudinal QC tracking across sessions.

Detects subject-level CBF decline, scanner drift, and inter-session
motion increase. Based on: Clement et al. 2023, MRM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SessionRecord:
    """QC record for a single session."""
    subject_id: str
    session_id: str
    timestamp: str
    metrics: Dict[str, float]
    overall_pass: bool
    acquisition_key: str = "3T_PCASL"


@dataclass
class LongitudinalAlert:
    """Alert raised by longitudinal analysis."""
    alert_type: str
    metric: str
    message: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    trend_per_session: Optional[float] = None


@dataclass
class LongitudinalSummary:
    """Summary of longitudinal QC analysis."""
    subject_id: str
    n_sessions: int
    alerts: List[LongitudinalAlert] = field(default_factory=list)
    metric_trends: Dict[str, float] = field(default_factory=dict)


class LongitudinalTracker:
    """
    Track QC metrics across sessions and detect drift.

    Usage
    -----
    tracker = LongitudinalTracker("/path/to/qc_database.json")
    summary = tracker.add_session(subject_id, session_id, metrics, overall_pass)
    tracker.save()
    """

    DECLINING_METRICS = {"qei", "tsnr", "dice"}
    INCREASING_METRICS = {"scov_gm", "mean_fd", "neg_gm_cbf"}
    TREND_WARNING_THRESHOLD = 0.02   # per session
    TREND_CRITICAL_THRESHOLD = 0.05  # per session

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._records: Dict[str, List[SessionRecord]] = {}
        if self.db_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.db_path, "r") as fh:
            raw = json.load(fh)
        for subj, sessions in raw.items():
            self._records[subj] = [
                SessionRecord(**s) for s in sessions
            ]

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {
            subj: [asdict(s) for s in sessions]
            for subj, sessions in self._records.items()
        }
        with open(self.db_path, "w") as fh:
            json.dump(serialised, fh, indent=2)

    def add_session(
        self,
        subject_id: str,
        session_id: str,
        metrics: Dict[str, float],
        overall_pass: bool,
        acquisition_key: str = "3T_PCASL",
    ) -> LongitudinalSummary:
        """
        Add a new session and compute longitudinal alerts.

        Returns
        -------
        LongitudinalSummary
            Trends and alerts for this subject.
        """
        record = SessionRecord(
            subject_id=subject_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            overall_pass=overall_pass,
            acquisition_key=acquisition_key,
        )

        if subject_id not in self._records:
            self._records[subject_id] = []
        self._records[subject_id].append(record)

        return self._analyse(subject_id)

    def _analyse(self, subject_id: str) -> LongitudinalSummary:
        """Compute trends and generate alerts for a subject."""
        sessions = self._records[subject_id]
        alerts: List[LongitudinalAlert] = []
        trends: Dict[str, float] = {}

        if len(sessions) < 3:
            return LongitudinalSummary(
                subject_id=subject_id,
                n_sessions=len(sessions),
                alerts=[LongitudinalAlert(
                    alert_type="INFO",
                    metric="all",
                    message=f"Only {len(sessions)} sessions — need >=3 for trend analysis.",
                    severity="INFO",
                )],
            )

        # Compute per-metric trends (slope via linear regression)
        all_metrics = set()
        for s in sessions:
            all_metrics.update(s.metrics.keys())

        for metric in all_metrics:
            values = [
                s.metrics.get(metric, np.nan) for s in sessions
            ]
            valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
            if len(valid) < 3:
                continue

            x = np.array([i for i, _ in valid], dtype=float)
            y = np.array([v for _, v in valid], dtype=float)

            # Linear regression slope
            slope = float(np.polyfit(x, y, 1)[0])
            trends[metric] = slope

            abs_slope = abs(slope)

            # Declining metrics (should stay high)
            if metric in self.DECLINING_METRICS and slope < 0:
                if abs_slope > self.TREND_CRITICAL_THRESHOLD:
                    alerts.append(LongitudinalAlert(
                        alert_type="METRIC_DECLINE",
                        metric=metric,
                        message=(
                            f"{metric} declining rapidly: "
                            f"{slope:+.4f} per session. "
                            "Possible scanner degradation or disease progression."
                        ),
                        severity="CRITICAL",
                        trend_per_session=slope,
                    ))
                elif abs_slope > self.TREND_WARNING_THRESHOLD:
                    alerts.append(LongitudinalAlert(
                        alert_type="METRIC_DECLINE",
                        metric=metric,
                        message=f"{metric} declining: {slope:+.4f} per session.",
                        severity="WARNING",
                        trend_per_session=slope,
                    ))

            # Increasing metrics (should stay low)
            elif metric in self.INCREASING_METRICS and slope > 0:
                if abs_slope > self.TREND_CRITICAL_THRESHOLD:
                    alerts.append(LongitudinalAlert(
                        alert_type="METRIC_INCREASE",
                        metric=metric,
                        message=(
                            f"{metric} increasing rapidly: "
                            f"{slope:+.4f} per session. "
                            "Check scanner stability."
                        ),
                        severity="CRITICAL",
                        trend_per_session=slope,
                    ))
                elif abs_slope > self.TREND_WARNING_THRESHOLD:
                    alerts.append(LongitudinalAlert(
                        alert_type="METRIC_INCREASE",
                        metric=metric,
                        message=f"{metric} increasing: {slope:+.4f} per session.",
                        severity="WARNING",
                        trend_per_session=slope,
                    ))

        return LongitudinalSummary(
            subject_id=subject_id,
            n_sessions=len(sessions),
            alerts=alerts,
            metric_trends=trends,
        )
