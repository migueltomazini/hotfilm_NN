"""Spectral analysis utilities for hot-film velocity data.

This module provides functions to calculate periodograms, apply log-bin smoothing,
and generate plots for wind velocity spectra, ensuring consistency across
training, prediction, and validation scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram


def compute_spectrum(signal: np.ndarray, fs: float):
    """Calculate the Power Spectral Density (PSD) of a signal.

    Args:
        signal: 1D array of velocity.
        fs: Sampling frequency.

    Returns:
        freqs, power_density: Arrays from scipy.signal.periodogram.
    """
    # Fluctuation u' = u - mean(u)
    fluc = signal - np.mean(signal)
    return periodogram(fluc, fs=fs)


def plot_spectral_density(
    df: pd.DataFrame, columns: list, fs: float, title: str, output_path: str
):
    """Generate and save a log-log spectral density plot.

    Args:
        df: Dataframe containing the velocity columns.
        columns: List of column names to plot (e.g., ['u_x', 'u_y', 'u_z']).
        fs: Sampling frequency.
        title: Plot title.
        output_path: Full path to save the .png file.
    """
    plt.figure(figsize=(10, 6))
    colors = ["red", "green", "blue"]
    labels = ["X Component", "Y Component", "Z Component"]

    # Reference -5/3 Kolmogorov line
    f_ref = np.logspace(0, np.log10(fs / 2), 100)
    y_ref = f_ref ** (-5 / 3) * (df[columns[0]].var() if not df.empty else 1e-2)
    plt.loglog(f_ref, y_ref, "k--", label="Reference -5/3", alpha=0.5)

    for col, color, lbl in zip(columns, colors, labels):
        if col in df.columns:
            freqs, psd = compute_spectrum(df[col].values, fs)
            plt.loglog(freqs, psd, color=color, label=lbl, alpha=0.8)

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral Density")
    plt.ylim(1e-18, 1e4)  # Match spectrum.py standard
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
