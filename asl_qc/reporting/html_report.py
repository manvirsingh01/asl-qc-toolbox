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
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f8f9fa; color: #212529; padding: 2rem;
      line-height: 1.6; max-width: 960px; margin: 0 auto;
    }
    h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: .25rem; color: #212529; }
    .subtitle { color: #6c757d; font-size: .85rem; margin-bottom: 1.5rem; }
    .card {
      background: #fff; border: 1px solid #dee2e6;
      border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;
    }
    .card h2 { font-size: 1rem; margin-bottom: .75rem; color: #212529; }
    .card h3 { font-size: .9rem; margin: 1rem 0 .5rem; color: #495057; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: .5rem .75rem; text-align: left; border-bottom: 1px solid #e9ecef; font-size: .85rem; }
    th { color: #6c757d; font-weight: 600; font-size: .75rem; text-transform: uppercase; }
    .badge {
      display: inline-block; padding: .2rem .5rem; border-radius: 4px;
      font-size: .75rem; font-weight: 600; text-transform: uppercase;
    }
    .badge-pass { background: #d1e7dd; color: #0f5132; }
    .badge-fail { background: #f8d7da; color: #842029; }
    .badge-warn { background: #fff3cd; color: #664d03; }
    .overall {
      font-size: 1.1rem; font-weight: 700; text-align: center; padding: .75rem;
      border-radius: 6px; margin-bottom: 1.25rem;
    }
    .overall-pass { background: #d1e7dd; color: #0f5132; }
    .overall-fail { background: #f8d7da; color: #842029; }
    .meta { color: #6c757d; font-size: .8rem; margin-bottom: .25rem; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .75rem; margin-bottom: 1.25rem; }
    .stat-card {
      background: #fff; border: 1px solid #dee2e6;
      border-radius: 6px; padding: .75rem; text-align: center;
    }
    .stat-val { font-size: 1.25rem; font-weight: 700; color: #0d6efd; }
    .stat-label { font-size: .75rem; color: #6c757d; margin-top: .15rem; }
    .image-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; }
    .image-panel { text-align: center; }
    .image-panel img { width: 100%; max-width: 900px; border-radius: 6px; border: 1px solid #dee2e6; }
    .image-panel .img-title { font-size: .85rem; font-weight: 600; margin-bottom: .4rem; }
    .image-panel .img-caption { font-size: .75rem; color: #6c757d; margin-top: .3rem; max-width: 700px; margin-left: auto; margin-right: auto; }
  </style>
</head>
<body>
  <h1>ASL Quality Control Report</h1>
  <p class="subtitle">Subject: {{ subject_id }} &middot; Generated: {{ timestamp }}</p>

  <div class="overall {{ 'overall-pass' if overall_pass else 'overall-fail' }}">
    {{ 'OVERALL PASS' if overall_pass else 'OVERALL FAIL' }}
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

  {% if images %}
  <div class="card">
    <h2>ExploreASL QC Image Panels</h2>
    <div class="image-grid">
      {% for img in images %}
      <div class="image-panel">
        <div class="img-title">{{ img.title }}</div>
        <img src="data:image/png;base64,{{ img.base64 }}" alt="{{ img.title }}">
        {% if img.caption %}<div class="img-caption">{{ img.caption }}</div>{% endif %}
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if data_summary %}
  <div class="card">
    <h2>Input Data Summary</h2>
    {% for section in data_summary %}
    <h3>{{ section.title }}</h3>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>
        {% for row in section.rows %}
        <tr><td>{{ row.name }}</td><td>{{ row.value }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
    {% endfor %}
  </div>
  {% endif %}

  <div class="card">
    <h2>Provenance</h2>
    <p class="meta">Toolbox: asl-qc-toolbox v2.0.0</p>
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
    data_summary: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Dict[str, str]]] = None,
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
    data_summary : list of dict, optional
        Sections of input data detail. Each dict has ``title`` (str)
        and ``rows`` (list of dicts with ``name`` and ``value``).
    images : list of dict, optional
        Embedded images. Each dict has ``title`` (str),
        ``base64`` (base64-encoded PNG), and optional ``caption`` (str).
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
        data_summary=data_summary,
        images=images,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(html)

    return output_path.resolve()
