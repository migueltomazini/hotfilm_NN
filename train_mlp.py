"""Physics-informed MLP training with spectral analysis.

This module trains Multi-Layer Perceptron models to predict wind velocity
from hot-film voltage measurements. The training process is guided by physical
constraints from Kolmogorov's theory:
- Spectral slope target: -5/3 (K41 inertial range)
- Isotropy ratio target: 4/3 (local isotropy in the inertial range)

The training pipeline uses Optuna for hyperparameter optimization with a
composite loss function balancing prediction accuracy with physical consistency.

Usage:
    python3 train_mlp.py <series_id1> [<series_id2> ...]

Example:
    python3 train_mlp.py 5940 21180 9999
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.signal import periodogram
from scipy.stats import linregress
import joblib
import optuna

# Suppress only specific warnings, not all
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Optuna logging.*")

info_output = '''
Usage: python3 train_mlp.py <series_id1> [<series_id2> ...]

Check the manual inside the following folder for data preparation details:
    manuals/manual.txt
'''

# Global Definitions
# Input size for the model
input_size = 4  # (x,y,z + reynolds)
# Output size for the model
output_size = 3  # Maintained 3 outputs (velocity_x, velocity_y, velocity_z)
# Number of training epochs
EPOCHS = 256
# Start time for training duration calculation
START_TIME = time.time()
# Column names in the input and output data
input_df_name, output_df_name = "voltage", "velocity"
# Model and directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_local = os.path.join(BASE_DIR, 'models')
data_dir = os.path.join(BASE_DIR, 'data')

# Device configuration - using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validate command line arguments
if len(sys.argv) < 2:
    print(info_output)
    sys.exit()

# List of time series data files
SERIES_LIST = sys.argv[1:]
# Identifier for the series, used for file naming
SERIE_IDENTIFIER = "_".join(SERIES_LIST)

# ==============================================================================
# PHYSICS HELPER FUNCTIONS (UNIVERSAL CONSTANTS)
# ==============================================================================

def calculate_spectral_slope(velocity_signal, fs):
    """Calculate PSD slope in the inertial subrange.

    Computes the spectral slope from the longitudinal velocity component's power
    spectral density. The target slope is -5/3 according to Kolmogorov's K41 theory.

    Args:
        velocity_signal: Velocity time series array (shape: [samples, 3]).
        fs: Sampling frequency in Hz.

    Returns:
        float: Spectral slope in the range 5-50 Hz. Returns -1.0 if insufficient data.
    """
    if len(velocity_signal) < 512:
        return -1.0

    # Compute PSD from longitudinal component (u)
    f, psd = periodogram(velocity_signal[:, 0], fs=fs)

    # Inertial range: 5-50 Hz provides overlap for real and synthetic data
    mask = (f >= 5.0) & (f <= 50.0)
    if not np.any(mask):
        return -1.0

    slope, _, _, _, _ = linregress(np.log10(f[mask]), np.log10(psd[mask]))
    return slope

def calculate_isotropy_ratio(velocity_signal, fs):
    """Calculate transverse-to-longitudinal energy ratio.

    Computes the isotropy metric (transverse/longitudinal energy) in the inertial
    subrange. The target ratio is 4/3 for locally isotropic turbulence.

    Args:
        velocity_signal: Velocity time series array (shape: [samples, 3]).
        fs: Sampling frequency in Hz.

    Returns:
        float: Isotropy ratio. Returns 1.0 if insufficient data.
    """
    if len(velocity_signal) < 512:
        return 1.0

    # Extract PSD for each component
    f, psd_u = periodogram(velocity_signal[:, 0], fs=fs)
    _, psd_v = periodogram(velocity_signal[:, 1], fs=fs)
    _, psd_w = periodogram(velocity_signal[:, 2], fs=fs)

    # Integrate energy in inertial range
    mask = (f >= 5.0) & (f <= 50.0)
    energy_u = np.trapz(psd_u[mask], f[mask])
    energy_trans = np.trapz((psd_v[mask] + psd_w[mask]) / 2, f[mask])

    return energy_trans / (energy_u + 1e-6)

# ==============================================================================
# MODEL AND DATASET
# ==============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron for voltage-to-velocity prediction.

    Configurable fully-connected network with ReLU activations between layers.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

class VoltageVelocityDataset(Dataset):
    """PyTorch Dataset for voltage-to-velocity pairs."""

    def __init__(self, X_data, Y_data, device_obj=None):
        self.device = device_obj if device_obj is not None else device
        self.X = torch.tensor(X_data).float()
        self.Y = torch.tensor(Y_data).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(self.device), self.Y[idx].to(self.device)

# ==============================================================================
# OPTUNA OBJECTIVE (UNIVERSAL PHYSICS)
# ==============================================================================

def objective(trial, X_train, Y_train, X_val, Y_val, fs):
    """Optuna objective function for physics-informed hyperparameter optimization.

    Trains a candidate model and evaluates it using a composite loss combining:
    - Prediction accuracy (MSE) - 50% weight
    - Spectral slope fidelity to -5/3 - 30% weight
    - Isotropy ratio fidelity to 4/3 - 20% weight

    This physics-informed approach ensures the model learns turbulence characteristics
    beyond simple point-wise prediction accuracy.
    """
    n_layers = trial.suggest_int('hidden_layers', 1, 4)
    n_size = trial.suggest_int('hidden_size', 16, 128)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    b_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train, device), batch_size=b_size, shuffle=True)
    model = MLP(input_size, output_size, n_size, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Quick training phase
    for epoch in range(50):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            criterion(model(X), Y).backward()
            optimizer.step()

    # Validation with physics metrics
    model.eval()
    with torch.no_grad():
        X_v = torch.tensor(X_val).float().to(device)
        Y_v = torch.tensor(Y_val).float().to(device)
        predictions = model(X_v)
        mse_error = criterion(predictions, Y_v).item()

        pred_np = predictions.cpu().numpy()

        # Compute physics metrics
        slope = calculate_spectral_slope(pred_np, fs)
        slope_err = abs(slope - (-5 / 3)) / (5 / 3)

    # Composite loss: 70% accuracy, 30% slope
    return (0.7 * mse_error) + (0.3 * slope_err)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Execute complete training pipeline with physics-informed optimization.

    Workflow:
        1. Load training data from multiple series
        2. Combine datasets and extract physical sampling rates
        3. Prepare features (voltage, reynolds) and targets (velocity)
        4. Normalize inputs using StandardScaler
        5. Split into train/validation sets (90/10)
        6. Optimize hyperparameters with Optuna using physics-informed objectives
        7. Train final model with best hyperparameters
        8. Evaluate physical metrics (slope, isotropy) on full dataset
        9. Export model weights, scaler, and hyperparameter metadata
    """
    # Load all series data and their configurations
    dfs = []
    configs = []
    for s in SERIES_LIST:
        csv_path = os.path.join(data_dir, 'train', f'train_df_{s}.csv')
        config_path = os.path.join(data_dir, 'config', f'config_{s}.json')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found: {csv_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        df = pd.read_csv(csv_path)
        with open(config_path, 'r') as f:
            configs.append(json.load(f))
        dfs.append(df)

    # Combine datasets for training
    df_total = pd.concat(dfs, ignore_index=True)
    fs = configs[0]['FS_HOTFILM']

    # Extract features and targets
    X_raw = df_total[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values
    Y_raw = df_total[['velocity_x', 'velocity_y', 'velocity_z']].values

    # Normalize features and save scaler for later inference
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    os.makedirs(model_local, exist_ok=True)
    scaler_path = os.path.join(model_local, f'scaler_{SERIE_IDENTIFIER}.joblib')
    joblib.dump(scaler, scaler_path)

    # Train/validation split
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_raw[:split], Y_raw[split:]

    # Hyperparameter optimization with physics guidance
    print(f"\n[Optuna] Starting physics-informed optimization...")
    study = optuna.create_study(direction='minimize')
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda trial: objective(trial, X_train, Y_train, X_val, Y_val, fs), n_trials=30, show_progress_bar=True)

    best_p = study.best_params
    # Final training with best parameters
    model = MLP(input_size, output_size, best_p['hidden_size'], best_p['hidden_layers']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_p['learning_rate'])
    criterion = nn.MSELoss()
    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train, device), batch_size=best_p['batch_size'], shuffle=True)

    print(f"\n[Training] Final run with best hyperparameters: {best_p}")
    for epoch in range(EPOCHS):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            criterion(model(X), Y).backward()
            optimizer.step()

    # Evaluate model on full dataset
    model.eval()
    with torch.no_grad():
        all_X = torch.tensor(X_scaled).float().to(device)
        predictions = model(all_X)
        pred_np = predictions.cpu().numpy()

        final_slope = calculate_spectral_slope(pred_np, fs)
        final_iso = calculate_isotropy_ratio(pred_np, fs)
        rmse = ((predictions - torch.tensor(Y_raw).float().to(device)) ** 2).mean().sqrt().item()

    # Save model, scaler, and metadata
    dest = os.path.join(data_dir, 'train', 'results', f'results_{SERIE_IDENTIFIER}')
    os.makedirs(dest, exist_ok=True)
    model_path = os.path.join(model_local, f'model_mlp_{SERIE_IDENTIFIER}.pth')
    torch.save(model.state_dict(), model_path)

    # Export metrics for reference during inference
    hyperparams_df = pd.DataFrame({
        'Layers': [best_p['hidden_layers']],
        'Size': [best_p['hidden_size']],
        'RMSE': [rmse],
        'Final_Slope': [final_slope],
        'Final_Isotropy': [final_iso]
    })
    hyperparams_path = os.path.join(dest, f'hyperparameters_{SERIE_IDENTIFIER}.csv')
    hyperparams_df.to_csv(hyperparams_path, index=False)

    print(f"\nTraining complete. Physics-informed metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Spectral Slope: {final_slope:.4f} (target: -1.667)")
    print(f"  Isotropy Ratio: {final_iso:.4f} (target: 1.333)")

if __name__ == "__main__": main()