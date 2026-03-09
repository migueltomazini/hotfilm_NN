"""Utility functions for calculating error metrics in wind velocity predictions.

This module provides standardized functions for computing common error metrics
such as RMSE, MAE, and MSE. These are used across training, validation, and
inference to ensure consistency in performance evaluation.
"""

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