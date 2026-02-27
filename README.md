#  ASL QC Toolbox

**Automated Quality Control for Arterial Spin Labeling MRI**
  
---

## Get Started in 30 Seconds

```bash
# Install
pip install -e ".[dev]"

# Run the demo (no MRI data needed!)
python run_demo.py  
```
   
That's it! The demo generates synthetic brain data and runs all QC metrics.

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
qei = compute_qei(cbf, gm_mask, wm_mask)      # → QEI score
scov = compute_scov(cbf, gm_mask)              # → sCoV value
hist = analyze_histogram(cbf, gm_mask)          # → skewness, kurtosis, etc.

# Check pass/fail against clinical thresholds
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
python -m pytest tests/ -v          # All 49 tests
python -m pytest tests/test_qei.py  # Single module
```

---

## Want to Contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) — add a new metric in 3 simple steps.

## License

MIT
