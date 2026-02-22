"""
NIfTI I/O utilities and brain-mask generation.

Wraps **nibabel** for consistent loading / saving of NIfTI-1/2 images and
provides a simple intensity-based brain mask generator for M0 images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI file and return ``(data, affine, header)``.

    Parameters
    ----------
    path : path-like
        Path to ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    data : np.ndarray
        Image data array (3-D or 4-D).
    affine : np.ndarray
        4×4 affine matrix mapping voxel → world coordinates.
    header : nib.Nifti1Header
        NIfTI header object.
    """
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float64), img.affine, img.header


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header | None,
    path: str | Path,
) -> None:
    """Save a numpy array as a NIfTI file."""
    img = nib.Nifti1Image(data.astype(np.float64), affine, header)
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a 1-D array of voxel values inside *mask*.

    Parameters
    ----------
    data : np.ndarray
        3-D volume (or 4-D timeseries; mask is applied per-volume).
    mask : np.ndarray
        Boolean 3-D mask of the same spatial shape as *data*.
    """
    mask = mask.astype(bool)
    if data.ndim == 3:
        return data[mask]
    elif data.ndim == 4:
        # Apply mask to each volume → (n_masked_voxels, n_vols)
        return np.array([data[..., t][mask] for t in range(data.shape[-1])]).T
    else:
        raise ValueError(f"Expected 3-D or 4-D data, got {data.ndim}-D")


def generate_brain_mask(
    volume: np.ndarray,
    threshold_fraction: float = 0.15,
    smoothing_fwhm_mm: float = 0.0,
    voxel_size_mm: float = 1.0,
) -> np.ndarray:
    """Generate a binary brain mask from a 3-D volume via intensity thresholding.

    Parameters
    ----------
    volume : np.ndarray
        3-D image (e.g. M0 or mean control image).
    threshold_fraction : float
        Fraction of the robust maximum (98th percentile) used as the
        binarisation threshold.
    smoothing_fwhm_mm : float
        Optional Gaussian smoothing FWHM applied before thresholding
        (set to 0 to skip).
    voxel_size_mm : float
        Isotropic voxel size used to convert FWHM to sigma.

    Returns
    -------
    mask : np.ndarray
        Boolean 3-D mask.
    """
    vol = volume.copy()

    if smoothing_fwhm_mm > 0:
        sigma = smoothing_fwhm_mm / (2.355 * voxel_size_mm)
        vol = gaussian_filter(vol, sigma=sigma)

    robust_max = np.percentile(vol[vol > 0], 98) if np.any(vol > 0) else 1.0
    threshold = threshold_fraction * robust_max
    mask = vol > threshold
    return mask.astype(bool)
