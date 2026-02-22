"""
Motion tracking metrics for ASL timeseries.

Computes **Framewise Displacement (FD)** from rigid-body motion parameters
and **DVARS** from the 4-D timeseries to quantify inter-volume head motion
and intensity instability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotionSummary:
    """Aggregate motion statistics for an ASL run."""

    mean_fd: float
    max_fd: float
    fd_timecourse: np.ndarray = field(repr=False)
    n_fd_spikes: int
    mean_dvars: float
    max_dvars: float
    dvars_timecourse: np.ndarray = field(repr=False)
    n_dvars_spikes: int


def compute_framewise_displacement(
    motion_params: np.ndarray,
    radius_mm: float = 50.0,
) -> np.ndarray:
    """Compute Framewise Displacement from rigid-body motion parameters.

    Parameters
    ----------
    motion_params : np.ndarray
        (N, 6) array where columns are
        ``[trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]``.
        Translations in mm, rotations in radians.
    radius_mm : float
        Assumed head radius for converting angular displacements to mm.

    Returns
    -------
    fd : np.ndarray
        (N-1,) Framewise Displacement in mm.
    """
    if motion_params.ndim != 2 or motion_params.shape[1] != 6:
        raise ValueError("motion_params must be (N, 6)")

    # Differentiate
    diff = np.diff(motion_params, axis=0)  # (N-1, 6)

    # Convert rotations (columns 3-5) from radians to mm arc-length
    diff[:, 3:] *= radius_mm

    # FD = sum of absolute displacements
    fd = np.sum(np.abs(diff), axis=1)
    return fd


def compute_dvars(
    data_4d: np.ndarray,
    brain_mask: np.ndarray,
) -> np.ndarray:
    """Compute DVARS (Derivative of RMS Variance) from a 4-D timeseries.

    DVARS measures the root-mean-square change in voxel intensity from
    one volume to the next within the brain mask.

    Parameters
    ----------
    data_4d : np.ndarray
        (X, Y, Z, T) timeseries.
    brain_mask : np.ndarray
        Boolean 3-D mask.

    Returns
    -------
    dvars : np.ndarray
        (T-1,) DVARS values.
    """
    mask = brain_mask.astype(bool)
    n_vols = data_4d.shape[-1]

    if n_vols < 2:
        return np.array([0.0])

    dvars = np.empty(n_vols - 1)
    for t in range(n_vols - 1):
        diff = data_4d[..., t + 1][mask] - data_4d[..., t][mask]
        dvars[t] = np.sqrt(np.mean(diff ** 2))

    return dvars


def summarize_motion(
    fd: np.ndarray,
    dvars: np.ndarray,
    fd_spike_threshold: float = 0.5,
    dvars_spike_threshold: float = 1.5,
) -> MotionSummary:
    """Aggregate FD and DVARS into summary statistics.

    Parameters
    ----------
    fd : np.ndarray
        Framewise Displacement timecourse.
    dvars : np.ndarray
        DVARS timecourse.
    fd_spike_threshold : float
        FD threshold (mm) above which a volume is counted as a spike.
    dvars_spike_threshold : float
        DVARS z-score threshold above which a volume is counted as a spike.

    Returns
    -------
    MotionSummary
    """
    # DVARS z-score spikes
    dvars_mean = np.mean(dvars) if len(dvars) > 0 else 0.0
    dvars_std = np.std(dvars) if len(dvars) > 1 else 1.0
    dvars_z = (dvars - dvars_mean) / (dvars_std + 1e-12)

    return MotionSummary(
        mean_fd=float(np.mean(fd)) if len(fd) > 0 else 0.0,
        max_fd=float(np.max(fd)) if len(fd) > 0 else 0.0,
        fd_timecourse=fd,
        n_fd_spikes=int(np.sum(fd > fd_spike_threshold)),
        mean_dvars=float(dvars_mean),
        max_dvars=float(np.max(dvars)) if len(dvars) > 0 else 0.0,
        dvars_timecourse=dvars,
        n_dvars_spikes=int(np.sum(dvars_z > dvars_spike_threshold)),
    )
