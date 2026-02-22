"""
HTML report generator for ASL QC results.

Produces a self-contained HTML document with a traffic-light summary
table and optional diagnostic plots (FD/DVARS timecourses, CBF histogram).
Uses Jinja2 for templating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ASL QC Report — {{ subject_id }}</title>
  <style>
    :root {
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --text: #e2e8f0; --text-muted: #94a3b8;
      --pass: #22c55e; --warn: #f59e0b; --fail: #ef4444;
      --accent: #6366f1;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); padding: 2rem;
      line-height: 1.6;
    }
    h1 {
      font-size: 1.75rem; font-weight: 700; margin-bottom: .25rem;
      background: linear-gradient(135deg, var(--accent), #a855f7);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: var(--text-muted); font-size: .875rem; margin-bottom: 2rem; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    }
    .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: var(--text); }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: .6rem 1rem; text-align: left; border-bottom: 1px solid var(--border);
      font-size: .875rem;
    }
    th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; font-size: .75rem; }
    .badge {
      display: inline-block; padding: .15rem .6rem; border-radius: 9999px;
      font-size: .75rem; font-weight: 600; text-transform: uppercase;
    }
    .badge-pass { background: rgba(34,197,94,.15); color: var(--pass); }
    .badge-fail { background: rgba(239,68,68,.15); color: var(--fail); }
    .badge-warn { background: rgba(245,158,11,.15); color: var(--warn); }
    .overall {
      font-size: 1.25rem; font-weight: 700; text-align: center; padding: 1rem;
      border-radius: 8px; margin-bottom: 1.5rem;
    }
    .overall-pass { background: rgba(34,197,94,.1); color: var(--pass); border: 1px solid var(--pass); }
    .overall-fail { background: rgba(239,68,68,.1); color: var(--fail); border: 1px solid var(--fail); }
    .meta { color: var(--text-muted); font-size: .75rem; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
    .stat-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem; text-align: center;
    }
    .stat-val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .stat-label { font-size: .75rem; color: var(--text-muted); margin-top: .25rem; }
  </style>
</head>
<body>
  <h1>🧠 ASL Quality Control Report</h1>
  <p class="subtitle">Subject: {{ subject_id }} &middot; Generated: {{ timestamp }}</p>

  <div class="overall {{ 'overall-pass' if overall_pass else 'overall-fail' }}">
    {{ '✅ OVERALL PASS' if overall_pass else '❌ OVERALL FAIL' }}
    — {{ n_passed }}/{{ n_total }} metrics passed
  </div>

  <div class="summary-grid">
    {% for stat in summary_stats %}
    <div class="stat-card">
      <div class="stat-val">{{ stat.value }}</div>
      <div class="stat-label">{{ stat.label }}</div>
    </div>
    {% endfor %}
  </div>

  <div class="card">
    <h2>Threshold Verdicts</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
      </thead>
      <tbody>
        {% for v in verdicts %}
        <tr>
          <td>{{ v.metric_name }}</td>
          <td>{{ "%.4f"|format(v.value) }}</td>
          <td>{{ v.operator }} {{ "%.3f"|format(v.threshold) }}</td>
          <td><span class="badge {{ 'badge-pass' if v.passed else 'badge-fail' }}">
            {{ 'PASS' if v.passed else 'FAIL' }}
          </span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {% if ml_verdict %}
  <div class="card">
    <h2>ML Outlier Detection</h2>
    <table>
      <thead><tr><th>Method</th><th>Anomaly Score</th><th>Status</th></tr></thead>
      <tbody>
        <tr>
          <td>{{ ml_verdict.method }}</td>
          <td>{{ "%.4f"|format(ml_verdict.anomaly_score) }}</td>
          <td><span class="badge {{ 'badge-fail' if ml_verdict.is_outlier else 'badge-pass' }}">
            {{ 'OUTLIER' if ml_verdict.is_outlier else 'INLIER' }}
          </span></td>
        </tr>
      </tbody>
    </table>
  </div>
  {% endif %}

  <div class="card">
    <h2>Provenance</h2>
    <p class="meta">Toolbox: asl-qc-toolbox v1.0.0</p>
    {% for key, path in input_files.items() %}
    <p class="meta">{{ key }}: {{ path }}</p>
    {% endfor %}
  </div>
</body>
</html>
""")


def generate_html_report(
    subject_id: str,
    timestamp: str,
    overall_pass: bool,
    verdicts: List[Dict[str, Any]],
    summary_stats: List[Dict[str, str]],
    input_files: Dict[str, str],
    ml_verdict: Optional[Dict[str, Any]] = None,
    output_path: str | Path = "qc_report.html",
) -> Path:
    """Generate a self-contained HTML QC report.

    Parameters
    ----------
    subject_id : str
        BIDS subject identifier.
    timestamp : str
        ISO timestamp string.
    overall_pass : bool
        Whether the scan passed all thresholds.
    verdicts : list of dict
        Each dict has ``metric_name``, ``value``, ``threshold``,
        ``operator``, ``passed``.
    summary_stats : list of dict
        Each dict has ``value`` (display string) and ``label``.
    input_files : dict
        Input file provenance.
    ml_verdict : dict, optional
        ML outlier detection result.
    output_path : path-like
        Where to write the HTML.

    Returns
    -------
    Path
        Absolute path to the generated report.
    """
    output_path = Path(output_path)
    n_passed = sum(1 for v in verdicts if v.get("passed"))

    html = _HTML_TEMPLATE.render(
        subject_id=subject_id,
        timestamp=timestamp,
        overall_pass=overall_pass,
        n_passed=n_passed,
        n_total=len(verdicts),
        verdicts=verdicts,
        summary_stats=summary_stats,
        input_files=input_files,
        ml_verdict=ml_verdict,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(html)

    return output_path.resolve()
