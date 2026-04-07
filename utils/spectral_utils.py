"""Spectral analysis utilities for hot-film velocity data.

This module provides optimized functions to calculate power spectral density
and generate multi-component comparison plots while managing memory usage
for large time-series datasets.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib

# Use Agg backend to avoid GUI overhead and memory leaks in loops
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import periodogram


def compute_spectrum(signal: np.ndarray, fs: float):
    """Calculate the Power Spectral Density (PSD) of a signal.

    Args:
        signal: 1D array of velocity values.
        fs: Sampling frequency in Hz.

    Returns:
        tuple: (frequencies, power_spectral_density)
    """
    signal = signal[~np.isnan(signal)]
    fluc = signal - np.mean(signal)
    return periodogram(fluc, fs=fs, window="boxcar")


def estimate_sampling_frequency(
    df: pd.DataFrame, time_col: str = "time"
) -> float | None:
    """Estimate sampling frequency from a time column.

    Args:
        df: Data frame containing a monotonically increasing time column.
        time_col: Name of the time column.

    Returns:
        Estimated sampling frequency in Hz, or None if it cannot be estimated.
    """
    if time_col not in df.columns:
        return None

    time_values = pd.to_numeric(df[time_col], errors="coerce").dropna().values
    if len(time_values) < 2:
        return None

    dt = np.diff(time_values)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None

    return float(1.0 / np.median(dt))


def plot_combined_spectrum(
    df: pd.DataFrame,
    cols: list,
    fs_pred: float,
    title: str,
    output_path: str,
    sonic_df: pd.DataFrame = None,
    fs_sonic: float = 20.0,
):
    """Generate a single image with 3 subplots (X, Y, Z) comparing Pred vs Sonic.

    Args:
        df: Dataframe containing the predicted velocity columns.
        cols: List of column names to plot [x, y, z].
        fs_pred: Sampling frequency of the predicted data (e.g., 2000Hz).
        title: Main figure title.
        output_path: Full path to save the resulting PNG.
        sonic_df: Optional Dataframe containing ground truth Sonic velocities.
        fs_sonic: Sampling frequency of the Sonic data (e.g., 20Hz).
    """
    # Use a layout that shares the Y axis for better physical comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    components = ["x", "y", "z"]

    # Kolmogorov -5/3 Reference line calculation
    # Range starts from 1Hz to Nyquist frequency
    f_ref = np.logspace(0, np.log10(fs_pred / 2), 100)
    y_ref = f_ref ** (-5 / 3)

    for i, (col, ax) in enumerate(zip(cols, axes)):
        # 1. Plot Predicted Data (Blue)
        if col in df.columns:
            f_p, p_p = compute_spectrum(df[col].values, fs_pred)
            ax.loglog(
                f_p,
                p_p,
                label=f"Predicted (velocity_{components[i]})",
                alpha=0.7,
                linewidth=0.8,
                color="tab:blue",
            )

        # 2. Plot Sonic Data (Orange)
        if sonic_df is not None:
            sonic_col = f"velocity_{components[i]}"
            if sonic_col in sonic_df.columns:
                f_s, p_s = compute_spectrum(sonic_df[sonic_col].values, fs_sonic)
                ax.loglog(
                    f_s,
                    p_s,
                    label=f"Sonic (velocity_{components[i]})",
                    alpha=0.7,
                    linewidth=0.8,
                    color="tab:orange",
                )

        # 3. Plot Reference Slope (Green)
        ax.loglog(
            f_ref, y_ref, label="Reference Slope (-5/3)", linewidth=2, color="tab:green"
        )

        # Subplot Styling
        ax.set_title(f"Component {components[i].upper()}")
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("Spectral Density")

        ax.set_xlim(1e-3, 1e3)
        ax.set_ylim(1e-18, 1e4)  # Physical standard for the project
        ax.legend(loc="lower left", fontsize="x-small")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save with optimized DPI to prevent OOM errors
    plt.savefig(output_path, dpi=120)

    # CRITICAL: Memory cleanup to prevent RAM overflow in loops
    for ax in axes:
        ax.cla()  # Clear axis
    fig.clf()  # Clear figure
    plt.close(fig)
    plt.close("all")
    gc.collect()  # Force Python Garbage Collector
