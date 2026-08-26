"""Physics-informed MLP training with spectral analysis and Fine-tuning support.

This module trains Multi-Layer Perceptron models to predict wind velocity
from hot-film voltage measurements. It supports both training from scratch (using Optuna)
and fine-tuning an existing model to adapt to new King's Law constants.

Usage:
    Train with one or more series: python3 train_mlp.py <series1> [series2 ...]
    Fine-tuning (optional):        python3 train_mlp.py <series1> [series2 ...] <base_model_name.pth>
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import joblib
import optuna

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Import utility modules
from utils import config, metrics, physics, data_loader, hyperparameter_optimization, spectral_utils

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Suppress Optuna info-level logging to prevent console spam, keeping only warnings/errors
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global Configurations
INPUT_SIZE = config.INPUT_SIZE
OUTPUT_SIZE = config.OUTPUT_SIZE
EPOCHS = config.EPOCHS
EPOCHS_FINETUNE = config.EPOCHS_FINETUNE
INPUT_DF_NAME = "voltage"
OUTPUT_DF_NAME = "velocity"

# Device configuration (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL AND DATASET
# ==============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for voltage-to-velocity prediction.

    Configurable fully-connected network with LeakyReLU activations to allow 
    safe high-frequency extrapolation without harmonic distortion (clipping).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Initialize LeakyReLU to avoid dead gradients and hard clipping
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
            
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            
        return self.output_layer(x)


class VoltageVelocityDataset(Dataset):
    """PyTorch Dataset for voltage-to-velocity pairs."""

    def __init__(self, X_data: np.ndarray, Y_data: np.ndarray, device_obj: torch.device = None):
        self.device = device_obj if device_obj is not None else DEVICE
        self.X = torch.tensor(X_data).float()
        self.Y = torch.tensor(Y_data).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx].to(self.device), self.Y[idx].to(self.device)


# ==============================================================================
# METADATA LOADING (For Fine-tuning)
# ==============================================================================

def get_base_model_params(model_name: str) -> tuple:
    """Retrieve hyperparameters of a previous model from its metadata file.
    
    Args:
        model_name: The filename of the base model (e.g., model_mlp_1234.pth).
        
    Returns:
        tuple: (num_hidden_layers, hidden_size) for accurate architecture reconstruction.
        
    Raises:
        FileNotFoundError: If the metadata CSV cannot be found.
    """
    model_id = model_name.replace("model_mlp_", "").replace(".pth", "")
    meta_path = os.path.join(
        config.DATA_DIR,
        "train",
        "results",
        f"results_{model_id}",
        f"hyperparameters_{model_id}.csv",
    )

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Base model metadata not found at: {meta_path}")

    df = pd.read_csv(meta_path)
    return int(df["Layers"].iloc[0]), int(df["Size"].iloc[0])


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def show_graphs(data: pd.DataFrame, predictions: torch.Tensor, train_loss_hist: list, val_loss_hist: list, serie_identifier: str):
    """Generate and save validation plots including time series comparison and loss evolution.
    
    Args:
        data: Original dataset containing reference variables.
        predictions: Model predictions tensor.
        train_loss_hist: History of training loss per epoch.
        val_loss_hist: History of validation loss per epoch.
        serie_identifier: The combined series string used for saving paths.
    """
    shown = predictions
    if torch.is_tensor(shown):
        shown_np = shown.cpu().detach().numpy()
        if shown_np.ndim == 3:
            shown_np = shown_np.squeeze(1)
        shown = pd.DataFrame(shown_np, columns=["axis_x", "axis_y", "axis_z"])

    graph_dir = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie_identifier}", "graphics"
    )
    os.makedirs(graph_dir, exist_ok=True)

    # Plotting for individual velocity components
    axes = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        plt.figure(i)
        plt.plot(data.time, data[f"velocity_{ax}"], color="r", label="Original")
        plt.plot(data.time, shown.iloc[:, i], color="g", label="Predicted")
        plt.title(f"Comparison Axis {ax.upper()}")
        plt.legend()
        plt.savefig(os.path.join(graph_dir, f"Velocity_Comparison_{ax}.png"))
        plt.close()

    # Plotting Training & Validation Loss
    plt.figure(3)
    t_hist = pd.DataFrame(train_loss_hist)
    v_hist = pd.DataFrame(val_loss_hist)
    
    if not t_hist.empty:
        plt.plot(t_hist.iloc[:, 0], t_hist.iloc[:, 1], label="Train")
    if not v_hist.empty:
        plt.plot(v_hist.iloc[:, 0], v_hist.iloc[:, 1], label="Validation")
        
    plt.title("Loss Evolution")
    plt.legend()
    plt.savefig(os.path.join(graph_dir, "Loss_Evolution.png"))
    plt.close()


def format_time(seconds: float) -> str:
    """Convert seconds to a readable HH:MM:SS format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physics-informed MLP training with fine-tuning support.",
        epilog="Note: Fine-tuning uses the OPTIMIZED HYPERPARAMETERS of the base model."
    )
    parser.add_argument(
        "args", 
        nargs="+", 
        help="One or more series identifiers, optionally followed by a base_model.pth for fine-tuning."
    )
    args_parsed = parser.parse_args()

    # Determine if the last argument is a model for fine-tuning
    args_list = args_parsed.args
    base_model_name = None
    if len(args_list) >= 2 and args_list[-1].endswith(".pth"):
        base_model_name = args_list.pop(-1)
        
    series_list = args_list
    serie_identifier = "_".join(series_list)

    script_start_time = time.time()
    logging.info(f"Starting training pipeline for series: {serie_identifier}")
    logging.info(f"Device configured to: {DEVICE}")

    # Load and combine all datasets
    dfs = []
    for s in series_list:
        df_path = os.path.join(config.DATA_DIR, "train", f"train_df_{s}.csv")
        if not os.path.exists(df_path):
            logging.error(f"Training data not found: {df_path}")
            sys.exit(1)
        dfs.append(pd.read_csv(df_path))

    df_total = pd.concat(dfs, ignore_index=True)
    df_total = df_total.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # DATA REDUCTION (20% for faster optimization runs)
    df_total = df_total.iloc[: int(len(df_total) * 0.2)].reset_index(drop=True)
    logging.info("Training with the initial 20% of data to accelerate execution.")

    # Calculate dynamic sampling frequency
    fs = spectral_utils.estimate_sampling_frequency(df_total, "time")
    if fs is None:
        fs = config.FS_HOTFILM_DEFAULT
        logging.warning(f"Could not estimate sampling frequency from 'time' column. Defaulting to {fs} Hz.")
    else:
        logging.info(f"Estimated sampling frequency from data: {fs:.2f} Hz")

    # Extract features and targets
    X_raw = df_total[[f"{INPUT_DF_NAME}_x", f"{INPUT_DF_NAME}_y", f"{INPUT_DF_NAME}_z"]].values
    Y_raw = df_total[[f"{OUTPUT_DF_NAME}_x", f"{OUTPUT_DF_NAME}_y", f"{OUTPUT_DF_NAME}_z"]].values

    # Handle Feature Scaling
    scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{serie_identifier}.joblib")
    
    if base_model_name:
        base_id = base_model_name.replace("model_mlp_", "").replace(".pth", "")
        base_scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{base_id}.joblib")

        if os.path.exists(base_scaler_path):
            logging.info(f"Fine-tuning mode: Loading original scaler from {base_scaler_path}")
            scaler = joblib.load(base_scaler_path)
            X_scaled = scaler.transform(X_raw)
        else:
            logging.warning("Original base scaler not found! Fine-tuning might produce anomalous results.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
    else:
        logging.info("Standard mode: Fitting new StandardScaler.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Train/validation split (90% / 10%)
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_raw[:split], Y_raw[split:]

    optuna_duration = None
    optuna_start_time = None

    # --- ARCHITECTURE & HYPERPARAMETERS SETUP ---
    if base_model_name:
        logging.info(f"Loading base model weights: {base_model_name}")
        h_layers, h_size = get_base_model_params(base_model_name)

        # Conservative hyperparameters for stable fine-tuning
        best_p = {
            "hidden_layers": h_layers,
            "hidden_size": h_size,
            "learning_rate": 1e-4,
            "batch_size": 32,
        }

        model = MLP(INPUT_SIZE, OUTPUT_SIZE, h_size, h_layers).to(DEVICE)
        model.load_state_dict(
            torch.load(os.path.join(config.MODEL_DIR, base_model_name), map_location=DEVICE)
        )

        # Freeze feature extraction layers to preserve previous knowledge
        for param in model.input_layer.parameters():
            param.requires_grad = False
        for layer in model.hidden_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        current_epochs = EPOCHS_FINETUNE
    else:
        logging.info("Starting hyperparameter optimization from scratch via Optuna...")
        optuna_start_time = time.time()

        best_p = hyperparameter_optimization.optimize_hyperparameters(
            X_train, Y_train, X_val, Y_val, fs, serie_identifier, DEVICE
        )

        optuna_duration = time.time() - optuna_start_time
        logging.info(f"Optimization completed in {format_time(optuna_duration)}")

        model = MLP(
            INPUT_SIZE, OUTPUT_SIZE, best_p["hidden_size"], best_p["hidden_layers"]
        ).to(DEVICE)
        current_epochs = EPOCHS

    # --- FINAL TRAINING LOOP ---
    training_start_time = time.time()
    logging.info(f"Initiating final training run for {current_epochs} epochs...")

    # Increase weight decay to prevent overly steep gradients on smoothed inputs
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_p["learning_rate"],
        weight_decay=1e-4, 
    )
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        VoltageVelocityDataset(X_train, Y_train, DEVICE),
        batch_size=best_p["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        VoltageVelocityDataset(X_val, Y_val, DEVICE), 
        batch_size=best_p["batch_size"]
    )

    train_loss_hist, val_loss_hist = [], []
    idx_t, idx_v = 0, 0

    for epoch in range(current_epochs):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
            train_loss_hist.append([idx_t, loss.item()])
            idx_t += 1

        model.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                val_loss = criterion(model(X), Y)
                val_loss_hist.append([idx_v, val_loss.item()])
                idx_v += 1

        if epoch % max(1, int(current_epochs / 8)) == 0 or epoch == current_epochs - 1:
            t_loss_val = train_loss_hist[-1][1] if train_loss_hist else 0
            v_loss_val = val_loss_hist[-1][1] if val_loss_hist else 0
            logging.info(f"Epoch {epoch:4d} | Train Loss: {t_loss_val:.4f} | Val Loss: {v_loss_val:.4f}")

    training_duration = time.time() - training_start_time

    # --- EVALUATION ---
    logging.info("Evaluating predictions on the full dataset...")
    model.eval()
    with torch.no_grad():
        all_X_tensor = torch.tensor(X_scaled).float().to(DEVICE)
        predictions = model(all_X_tensor)

        # Extract predicted data for physical validation
        pred_np = predictions.cpu().numpy()
        final_slope = physics.calculate_spectral_slope(pred_np, fs)
        final_iso = physics.calculate_isotropy_ratio(pred_np, fs)

        # Calculate final global RMSE
        target_tensor = torch.tensor(Y_raw).float().to(DEVICE)
        rmse = metrics.calculate_rmse_torch(predictions, target_tensor)

    # --- SAVE AND EXPORT ---
    dest = os.path.join(config.DATA_DIR, "train", "results", f"results_{serie_identifier}")
    os.makedirs(dest, exist_ok=True)
    
    new_model_path = os.path.join(config.MODEL_DIR, f"model_mlp_{serie_identifier}.pth")
    torch.save(model.state_dict(), new_model_path)
    
    logging.info(f"Training complete. Model saved to: {new_model_path}")

    # Execution Timing Summary
    total_duration = time.time() - script_start_time
    logging.info("--- TIMING SUMMARY ---")
    if optuna_duration is not None:
        logging.info(f"Optuna Optimization : {format_time(optuna_duration)}")
    logging.info(f"Final Training      : {format_time(training_duration)}")
    logging.info(f"Total Execution Time: {format_time(total_duration)}")
    logging.info("----------------------")

    # Save hyperparameters and metrics
    pd.DataFrame(
        {
            "Layers": [best_p["hidden_layers"]],
            "Size": [best_p["hidden_size"]],
            "RMSE": [rmse],
            "Final_Slope": [final_slope],
            "Final_Isotropy": [final_iso],
        }
    ).to_csv(os.path.join(dest, f"hyperparameters_{serie_identifier}.csv"), index=False)

    logging.info(f"Final Metrics -> RMSE: {rmse:.6f}, Slope: {final_slope:.4f}, Isotropy: {final_iso:.4f}")
    
    # Generate Plots
    show_graphs(df_total, predictions, train_loss_hist, val_loss_hist, serie_identifier)


if __name__ == "__main__":
    main()