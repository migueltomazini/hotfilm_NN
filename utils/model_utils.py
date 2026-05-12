"""Model utilities for training, evaluating and predicting on data blocks.

This module encapsulates PyTorch logic, handling model loads, memory cleanup,
and block-wise training/inference steps.
"""

import os
import json
import gc
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from utils import config, metrics
from train_mlp import MLP, VoltageVelocityDataset


def cleanup_memory():
    """Forces garbage collection and clears PyTorch cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_block_model_and_scaler(
    serie: str, block_idx: int, device: torch.device
) -> Tuple[MLP, StandardScaler]:
    """Load the specific model and scaler for a given series and block index."""
    model_name = f"model_{serie}_block{block_idx}.pth"
    scaler_name = f"scaler_{serie}_block{block_idx}.joblib"

    model_path = os.path.join(config.MODEL_DIR, "incremental", model_name)
    scaler_path = os.path.join(config.MODEL_DIR, "incremental", scaler_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Block model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Block scaler not found: {scaler_path}")

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load optimized hyperparameters from best_params file
    best_params_path = os.path.join(
        config.DATA_DIR, "train", "best_params", f"best_params_{serie}_incremental.json"
    )
    
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        hidden_size = best_params.get("hidden_size", 64)
        num_hidden_layers = best_params.get("hidden_layers", 2)
    else:
        # Fallback to defaults if params file not found
        hidden_size = 64
        num_hidden_layers = 2

    # Create model with correct architecture
    model = MLP(config.INPUT_SIZE, config.OUTPUT_SIZE, hidden_size, num_hidden_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, scaler


def predict_on_block(
    model: MLP, scaler: StandardScaler, df: pd.DataFrame, device: torch.device
) -> np.ndarray:
    """Generate predictions for a specific dataframe block."""
    # Convert to float32 to save 50% RAM
    X_raw = df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled).float().to(device))
        
    # Free memory immediately
    del X_raw
    del X_scaled
    
    return preds.cpu().numpy()


def train_on_block(
    model: torch.nn.Module,
    scaler: StandardScaler,
    block: pd.DataFrame,
    epochs: int,
    device: torch.device,
    freeze: bool = False,
    lr: float = 1e-4,
    batch_size: int = 32,
) -> torch.nn.Module:
    """Fit or fine-tune ``model`` on a single block of data."""
    X_raw = block[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = block[["velocity_x", "velocity_y", "velocity_z"]].values

    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] == X_raw.shape[1]:
        X_scaled = scaler.transform(X_raw)
    else:
        scaler.fit(X_raw)
        X_scaled = scaler.transform(X_raw)
        
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_raw[:split], Y_raw[split:]

    train_loader = DataLoader(
        VoltageVelocityDataset(X_train, Y_train, device), batch_size=batch_size, shuffle=True
    )

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.output_layer.parameters():
            param.requires_grad = True

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    for epoch in range(epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} completed")
    return model


def evaluate_block(
    model: torch.nn.Module, scaler: StandardScaler, block: pd.DataFrame, fs: float, device: torch.device
) -> dict:
    """Compute error metrics for a single block and return a dict."""
    X_raw = block[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = block[["velocity_x", "velocity_y", "velocity_z"]].values
    
    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] == X_raw.shape[1]:
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = scaler.fit_transform(X_raw)  
        
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled).float().to(device))
    preds_np = preds.cpu().numpy()
    rmse = metrics.calculate_rmse(preds_np, Y_raw)
    return {"rmse": rmse}