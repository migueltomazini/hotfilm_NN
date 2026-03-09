"""Physics-based utility functions for spectral analysis and turbulence metrics.

This module contains functions for calculating spectral properties and turbulence
characteristics from velocity signals, used in physics-informed neural network training.
"""

import numpy as np
from scipy.signal import periodogram
from scipy.stats import linregress
from scipy.integrate import trapz


def calculate_spectral_slope(velocity_signal: np.ndarray, fs: float) -> float:
    """Calculate PSD slope in the inertial subrange (Target -5/3).

    Args:
        velocity_signal: 2D array of velocity components (shape: [n_samples, 3]).
        fs: Sampling frequency in Hz.

    Returns:
        Spectral slope as float. Returns -1.0 if insufficient data or invalid.
    """
    if len(velocity_signal) < 512:
        return -1.0

    # Compute PSD from longitudinal component (u)
    f, psd = periodogram(velocity_signal[:, 0], fs=fs)

    # Inertial range: 5-50 Hz provides overlap for real and synthetic data
    mask = (f >= 5.0) & (f <= 50.0)
    if not np.any(mask):
        return -1.0

    # Protection against log10(0) that generates NaNs
    log_f = np.log10(f[mask] + 1e-12)
    log_psd = np.log10(psd[mask] + 1e-12)

    slope, _, _, _, _ = linregress(log_f, log_psd)

    if np.isnan(slope):
        return -1.0
    return float(slope)


def calculate_isotropy_ratio(velocity_signal: np.ndarray, fs: float) -> float:
    """Calculate the ratio between transverse and longitudinal energy (Target: 4/3).

    Args:
        velocity_signal: 2D array of velocity components (shape: [n_samples, 3]).
        fs: Sampling frequency in Hz.

    Returns:
        Isotropy ratio as float. Returns 1.0 if insufficient data.
    """
    if len(velocity_signal) < 512:
        return 1.0

    # Extract PSD for each component
    f, psd_u = periodogram(velocity_signal[:, 0], fs=fs)
    _, psd_v = periodogram(velocity_signal[:, 1], fs=fs)
    _, psd_w = periodogram(velocity_signal[:, 2], fs=fs)

    mask = (f >= 5.0) & (f <= 50.0)
    if not np.any(mask):
        return 1.0

    # Energy in the inertial range using trapezoidal integration
    energy_u = trapz(psd_u[mask], f[mask])
    energy_trans = trapz((psd_v[mask] + psd_w[mask]) / 2, f[mask])

    # Protection against division by zero
    return float(energy_trans / (energy_u + 1e-12))