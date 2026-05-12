"""Hyperparameter optimization utilities using Optuna.

This module provides functions for optimizing neural network hyperparameters
using Optuna, with physics-informed objectives for hot-film velocity prediction.
"""

import os
import json
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import config, physics


def objective(trial, X_train, Y_train, X_val, Y_val, fs, device):
    """Optuna objective function for physics-informed hyperparameter optimization.

    Args:
        trial: Optuna trial object.
        X_train: Training input data.
        Y_train: Training target data.
        X_val: Validation input data.
        Y_val: Validation target data.
        fs: Sampling frequency.
        device: PyTorch device.

    Returns:
        Optimization score combining MSE and spectral slope error.
    """
    # Import locally to avoid circular dependency
    from train_mlp import MLP, VoltageVelocityDataset
    
    n_layers = trial.suggest_int("hidden_layers", 1, 4)
    n_size = trial.suggest_int("hidden_size", 16, 128)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    b_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_loader = DataLoader(
        VoltageVelocityDataset(X_train, Y_train, device),
        batch_size=b_size,
        shuffle=True,
    )
    model = MLP(config.INPUT_SIZE, config.OUTPUT_SIZE, n_size, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Fast training for optimization
    for _ in range(50):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()

    # Validation with physics metrics
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).float().to(device))
        mse = criterion(preds, torch.tensor(Y_val).float().to(device)).item()
        slope = physics.calculate_spectral_slope(preds.cpu().numpy(), fs)
        slope_err = abs(slope - (-5 / 3)) / (5 / 3)

        # Combined score with safety for NaN/inf
        score = (0.7 * mse) + (0.3 * slope_err)
        if np.isnan(score) or np.isinf(score):
            score = 1e6  # Penalize invalid results

    return score


def optimize_hyperparameters(X_train, Y_train, X_val, Y_val, fs, serie_identifier, device, suffix=""):
    """Optimize hyperparameters using Optuna.

    Args:
        X_train: Training input data.
        Y_train: Training target data.
        X_val: Validation input data.
        Y_val: Validation target data.
        fs: Sampling frequency.
        serie_identifier: Identifier for the series (used for caching).
        device: PyTorch device.
        suffix: Optional suffix for the parameters file (e.g., "_incremental").

    Returns:
        Dictionary with optimized hyperparameters.
    """
    best_params_dir = os.path.join(config.DATA_DIR, "train", "best_params")
    os.makedirs(best_params_dir, exist_ok=True)
    best_params_path = os.path.join(
        best_params_dir, f"best_params_{serie_identifier}{suffix}.json"
    )

    if os.path.exists(best_params_path):
        print(f"[Optuna] Loading cached hyperparameters from {best_params_path}")
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
    else:
        print(f"[Optuna] Starting hyperparameter optimization for {serie_identifier}...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, Y_train, X_val, Y_val, fs, device),
            n_trials=50,
            timeout=600,  # 10 minutes timeout
        )

        best_params = study.best_params
        best_params["epochs"] = 64
        best_params["epochs_finetune"] = 20

        # Save optimized parameters
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"[Optuna] Optimization completed. Best params saved to {best_params_path}")

    return best_params