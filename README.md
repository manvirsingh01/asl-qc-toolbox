# ASL QC Toolbox

**Automated Quality Control for Arterial Spin Labeling MRI**

---

## Install and Use on Another Machine (from GitHub)

```bash
# Install directly from GitHub
pip install "git+https://github.com/manvirsingh01/asl-qc-toolbox.git@main"
```

After install:

```bash
# Check CLI is available
asl-qc --help
```

Use in Python:

```python
from asl_qc.metrics.qei import compute_qei
from asl_qc.metrics.scov import compute_scov
```

---

## Get Started in 30 Seconds (local development)

```bash
# Install from source checkout (editable)
pip install -e ".[dev]"

# Run the demo (no MRI data needed!)
python run_demo.py
```

That's it! The demo generates synthetic brain data and runs all QC metrics.

---

## Alternative: Build Once, Install Anywhere

```bash
# In this repository, build a wheel
python3 -m pip wheel . -w dist

# Copy dist/asl_qc_toolbox-*.whl to the target machine and install there
pip install /path/to/asl_qc_toolbox-*.whl
```

---

## Use in Your Python Scripts

```python
import numpy as np
from asl_qc.metrics.qei import compute_qei
from asl_qc.metrics.scov import compute_scov
from asl_qc.metrics.histogram import analyze_histogram
from asl_qc.thresholds.empirical import apply_empirical_thresholds

# Load your CBF map and tissue masks (nibabel, numpy, etc.)
# cbf = nib.load("cbf.nii.gz").get_fdata()
# gm_mask = nib.load("gm.nii.gz").get_fdata() > 0.5
# wm_mask = nib.load("wm.nii.gz").get_fdata() > 0.5

# Compute metrics
qei = compute_qei(cbf, gm_mask, wm_mask)
scov = compute_scov(cbf, gm_mask)
hist = analyze_histogram(cbf, gm_mask)

# Check pass/fail against empirical thresholds
result = apply_empirical_thresholds({
    "qei": qei.qei,
    "scov_gm": scov,
    "mean_fd": 0.3,
    "gm_wm_ratio": 2.5,
    "neg_gm_cbf": qei.negative_gm_fraction,
    "dice": 0.85,
})
print(f"Overall: {'PASS' if result.overall_pass else 'FAIL'}")
```

See `examples/` for more complete scripts.

---

## Use the CLI

```bash
asl-qc \
  --input /path/to/bids/sub-01 \
  --cbf-map /path/to/cbf.nii.gz \
  --gm-mask /path/to/gm.nii.gz \
  --wm-mask /path/to/wm.nii.gz \
  --output-dir ./qc_output \
  --verbose
```

Generates JSON + HTML reports with traffic-light pass/fail summaries.

---

## Available QC Metrics

| Metric | What it detects | Module |
|---|---|---|
| **QEI** | Overall CBF map quality | `asl_qc.metrics.qei` |
| **sCoV** | Arterial transit time artifacts | `asl_qc.metrics.scov` |
| **Histogram** | Vascular artifacts, noise | `asl_qc.metrics.histogram` |
| **tSNR / RMS** | Acquisition instability | `asl_qc.metrics.snr` |
| **Dice / Jaccard** | Registration failures | `asl_qc.metrics.tissue_mask` |
| **GM/WM Ratio** | Segmentation errors | `asl_qc.metrics.tissue_mask` |
| **FD / DVARS** | Head motion | `asl_qc.preprocessing.motion` |
| **Control-Label** | Pattern desynchronization | `asl_qc.preprocessing.control_label` |
| **M0 Assessment** | Calibration artifacts | `asl_qc.preprocessing.m0_calibration` |
| **SCORE / ENABLE** | Outlier volume rejection | `asl_qc.preprocessing.outlier_rejection` |
| **ML Outlier** | Multi-dimensional anomalies | `asl_qc.thresholds.ml_outlier` |

---

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_qei.py
```

---

## Want to Contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) — add a new metric in 3 simple steps.

## License

MIT
