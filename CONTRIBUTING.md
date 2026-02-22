# Contributing to ASL QC Toolbox

Welcome! This guide will help you get started quickly.

## Setup (2 minutes)

```bash
# 1. Clone the repo
git clone <repo-url> && cd osipi

# 2. Install in development mode
pip install -e ".[dev]"

# 3. Verify everything works
python -m pytest tests/ -v

# 4. Run the demo
python run_demo.py
```

## Project Layout

```
asl_qc/                     ← Main package (edit here)
├── config.py                   Config loader
├── bids.py                     BIDS metadata parser
├── io_utils.py                 NIfTI I/O helpers
├── preprocessing/              Pre-QC checks
│   ├── control_label.py           Pattern verification
│   ├── m0_calibration.py          M0 assessment
│   ├── motion.py                  FD / DVARS
│   └── outlier_rejection.py       SCORE / ENABLE
├── metrics/                    QC metric computations
│   ├── qei.py                     Quality Evaluation Index
│   ├── scov.py                    Spatial CoV
│   ├── histogram.py               Distribution analysis
│   ├── snr.py                     tSNR / RMS difference
│   └── tissue_mask.py             Dice / Jaccard / GM-WM ratio
├── thresholds/                 Classification
│   ├── empirical.py               Linear threshold gates
│   └── ml_outlier.py              ML anomaly detection
├── reporting/                  Output generation
│   ├── json_report.py             JSON reports
│   └── html_report.py             HTML visual reports
└── cli.py                      CLI entry point

tests/                       ← Tests (add yours here)
examples/                    ← Usage examples
config/                      ← Default YAML config
```

## How to Add a New QC Metric

1. **Create** `asl_qc/metrics/your_metric.py`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class YourResult:
    value: float
    message: str = ""

def compute_your_metric(cbf_map, mask):
    vals = cbf_map[mask.astype(bool)]
    return YourResult(value=float(np.mean(vals)))
```

2. **Add a test** in `tests/test_your_metric.py`:

```python
from asl_qc.metrics.your_metric import compute_your_metric

def test_basic(brain_mask):
    import numpy as np
    cbf = np.ones(brain_mask.shape) * 50
    result = compute_your_metric(cbf, brain_mask)
    assert result.value > 0
```

3. **Run tests**: `python -m pytest tests/test_your_metric.py -v`

4. **Wire it into the CLI** (optional): add your metric call in `asl_qc/cli.py` → `run_pipeline()`

## How to Change Thresholds

Edit `config/default_config.yaml`:

```yaml
thresholds:
  empirical:
    qei_min: 0.53        # ← Change these
    scov_max: 0.42
    mean_fd_max_mm: 0.5
```

Or pass a custom YAML at runtime:

```bash
asl-qc --config my_thresholds.yaml --input /path/to/data
```

## Running Tests

```bash
python -m pytest tests/ -v           # All tests
python -m pytest tests/test_qei.py   # Single module
python -m pytest tests/ -k "motion"  # Pattern match
```

## Code Style

- Use **type hints** on all function signatures
- Return **dataclasses** (not dicts) from metric functions
- Write **docstrings** with Parameters/Returns sections
- Keep functions **pure** — no side effects, no global state
