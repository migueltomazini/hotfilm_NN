"""Utility functions for calculating error metrics in wind velocity predictions.

This module provides standardized functions for computing common error metrics
such as RMSE, MAE, and MSE. These are used across training, validation, and
inference to ensure consistency in performance evaluation. It also provides
specialized fluid dynamics metrics such as the Delta parameter used for
time-series validation in turbulence modeling.
"""

from scipy.signal import butter, filtfilt
import numpy as np
import torch


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).

    Args:
        predictions: Predicted values as numpy array.
        targets: Ground truth values as numpy array.

    Returns:
        RMSE value as float.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    Args:
        predictions: Predicted values as numpy array.
        targets: Ground truth values as numpy array.

    Returns:
        MAE value as float.
    """
    return np.mean(np.abs(predictions - targets))


def calculate_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Square Error (MSE).

    Args:
        predictions: Predicted values as numpy array.
        targets: Ground truth values as numpy array.

    Returns:
        MSE value as float.
    """
    return np.mean((predictions - targets) ** 2)


def calculate_rmse_torch(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate RMSE for PyTorch tensors.

    Args:
        predictions: Predicted values as torch tensor.
        targets: Ground truth values as torch tensor.

    Returns:
        RMSE value as float.
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def calculate_delta_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fs: float, cutoff: float = 2.0
) -> list:
    """Calculate the Delta parameter (Freire et al., 2023).

    The Delta parameter evaluates the similarity between two signals
    after low-pass filtering and Z-score normalization.
    A value closer to 0 indicates higher similarity in low-frequency
    structures.

    Args:
        y_true: Ground truth velocities (N, 3).
        y_pred: Predicted velocities (N, 3).
        fs: Sampling frequency in Hz.
        cutoff: Low-pass filter cutoff frequency in Hz (default 2Hz).

    Returns:
        List of 3 floats representing Delta for [u1, u2, u3].
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # 4th order Butterworth filter
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    deltas = []
    for i in range(3):
        # Apply zero-phase filtering
        s_true = filtfilt(b, a, y_true[:, i])
        s_pred = filtfilt(b, a, y_pred[:, i])

        # Z-score normalization
        std_true = np.std(s_true)
        std_pred = np.std(s_pred)

        # Avoid division by zero
        s_true_norm = (s_true - np.mean(s_true)) / std_true if std_true > 0 else s_true
        s_pred_norm = (s_pred - np.mean(s_pred)) / std_pred if std_pred > 0 else s_pred

        # RMS of the difference
        delta = np.sqrt(np.mean((s_pred_norm - s_true_norm) ** 2))
        deltas.append(delta)

    return deltas
