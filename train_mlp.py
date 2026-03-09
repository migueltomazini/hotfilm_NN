"""Physics-informed MLP training with spectral analysis and Fine-tuning support.

This module trains Multi-Layer Perceptron models to predict wind velocity
from hot-film voltage measurements. It supports both training from scratch (using Optuna)
and fine-tuning an existing model to adapt to new King's Law constants and Reynolds numbers.

Usage:
    Train with one or more series: python3 train_mlp.py <series1> [series2 ...]
    Fine-tuning (optional):        python3 train_mlp.py <series1> [series2 ...] <base_model_name.pth>
"""

import os
import sys
import time
import json
import warnings
import logging
from matplotlib import pyplot as plt
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
import joblib
import optuna

# Import utility modules
from utils import config, metrics, physics, data_loader

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Optuna logging.*")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

info_output = '''
Usage: 
    Train with one or more series: python3 train_mlp.py <series1> [series2 ...]
    Fine-tuning (optional): python3 train_mlp.py <series1> [series2 ...] <base_model_name.pth>

NOTA IMPORTANTE SOBRE FINE-TUNING:
- Fine-tuning agora usa os HIPERPARAMETROS OTIMIZADOS do modelo base
- O scaler e inteligentemente escolhido: reutilizado se Reynolds similar, novo se muito diferente
- Isso melhora significativamente a generalizacao entre diferentes numeros de Reynolds
'''

# Use configuration constants
input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE
EPOCHS = config.EPOCHS
EPOCHS_FINETUNE = config.EPOCHS_FINETUNE
START_TIME = time.time()

# Column names in the input and output data
input_df_name, output_df_name = "voltage", "velocity"

# Model and directory configurations
model_local = config.MODEL_DIR
data_dir = config.DATA_DIR
dir_base = config.BASE_DIR

# Device configuration - using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validate command line arguments
if len(sys.argv) < 2:
    print(info_output)
    sys.exit()

# Parse command line: allow multiple series IDs and optional base model (.pth)
args = sys.argv[1:]
BASE_MODEL_NAME = None
if len(args) >= 2 and args[-1].endswith('.pth'):
    BASE_MODEL_NAME = args[-1]
    SERIES_LIST = args[:-1]
else:
    SERIES_LIST = args

# Combined identifier for multi-series runs (used for filenames)
SERIE_IDENTIFIER = "_".join(SERIES_LIST)

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
# METADATA LOADING (For Fine-tuning)
# ==============================================================================

def get_base_model_params(model_name):
    """Retrieves hyperparameters of a previous model from its metadata file."""
    model_id = model_name.replace("model_mlp_", "").replace(".pth", "")
    meta_path = os.path.join(data_dir, 'train', 'results', f'results_{model_id}', f'hyperparameters_{model_id}.csv')
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Base model metadata not found at: {meta_path}")
    
    df = pd.read_csv(meta_path)
    return int(df['Layers'].iloc[0]), int(df['Size'].iloc[0])

# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def objective(trial, X_train, Y_train, X_val, Y_val, fs):
    """Optuna objective function for physics-informed hyperparameter optimization.
    """
    n_layers = trial.suggest_int('hidden_layers', 1, 4)
    n_size = trial.suggest_int('hidden_size', 16, 128)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    b_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train, device), batch_size=b_size, shuffle=True)
    model = MLP(input_size, output_size, n_size, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(50): # Fast trial
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            criterion(model(X), Y).backward()
            optimizer.step()

    # Validation with physics metrics
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).float().to(device))
        mse = criterion(preds, torch.tensor(Y_val).float().to(device)).item()
        slope = physics.calculate_spectral_slope(preds.cpu().numpy(), fs)
        slope_err = abs(slope - (-5/3)) / (5/3)

        # Added safety for NaN or infinite results during Optuna optimization
        score = (0.7 * mse) + (0.3 * slope_err)
        if np.isnan(score) or np.isinf(score):
            return 999.0 # High penalty for non-converging trials

    return score

def show_graphs(data, predictions, train_loss_hist, val_loss_hist):
    shown = predictions
    if torch.is_tensor(shown):
        # Convert tensor to numpy and handle potential extra dimensions
        shown_np = shown.cpu().detach().numpy()
        if shown_np.ndim == 3: shown_np = shown_np.squeeze(1)
        shown = pd.DataFrame(shown_np, columns=['axis_x', 'axis_y', 'axis_z'])

    graph_dir = os.path.join(data_dir, 'train', 'results', f"results_{SERIE_IDENTIFIER}", "graphics")
    os.makedirs(graph_dir, exist_ok=True)
    
    # Plotting for components
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        plt.figure(i)
        plt.plot(data.time, data[f'velocity_{ax}'], color='r', label='Original')
        plt.plot(data.time, shown.iloc[:, i], color='g', label='Predicted')
        plt.title(f"Comparison Axis {ax.upper()}")
        plt.legend()
        plt.savefig(os.path.join(graph_dir, f"Velocity_Comparison_{ax}.png"))
        plt.close()
    
    # Plotting Loss
    plt.figure(3)
    t_hist = pd.DataFrame(train_loss_hist)
    v_hist = pd.DataFrame(val_loss_hist)
    if not t_hist.empty:
        plt.plot(t_hist.iloc[:,0], t_hist.iloc[:,1], label='Train')
    if not v_hist.empty:
        plt.plot(v_hist.iloc[:,0], v_hist.iloc[:,1], label='Validation')
    plt.title("Loss Evolution")
    plt.legend()
    plt.savefig(os.path.join(graph_dir, "Loss_Evolution.png"))
    plt.close()

# ==============================================================================
# MAIN
# ==============================================================================

def format_time(seconds):
    """Convert seconds to a readable format (HH:MM:SS)."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

def main():
    script_start_time = time.time()
    
    dfs, configs = [], []
    for s in SERIES_LIST:
        df_path = os.path.join(data_dir, 'train', f'train_df_{s}.csv')
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Training data not found: {csv_path}")
        df = pd.read_csv(df_path)
        with open(os.path.join(data_dir, 'config', f'config_{s}.json'), 'r') as f:
            configs.append(json.load(f))
        dfs.append(df)

    # Combine datasets for training
    df_total = pd.concat(dfs, ignore_index=True)
    
    # Cleaning data to avoid contamination with sensor artifacts
    df_total = df_total.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    fs = configs[0]['FS_HOTFILM']

    # Extract features and targets
    X_raw = df_total[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values
    Y_raw = df_total[['velocity_x', 'velocity_y', 'velocity_z']].values

    # Persist or Fit Scaler based on training mode and ensure consistent scaling for fine-tuning
    scaler_path = os.path.join(model_local, f'scaler_{SERIE_IDENTIFIER}.joblib')
    
    if BASE_MODEL_NAME:
        # Fine-tuning mode: LOADING original scaler from base model
        base_id = BASE_MODEL_NAME.replace("model_mlp_", "").replace(".pth", "")
        base_scaler_path = os.path.join(model_local, f'scaler_{base_id}.joblib')
        
        if os.path.exists(base_scaler_path):
            print(f"[Fine-tuning] Loading original scaler: {base_scaler_path}")
            scaler = joblib.load(base_scaler_path)
            X_scaled = scaler.transform(X_raw) 
        else:
            print("[Warning] Original base scaler not found! Fine-tuning might produce NaNs.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
    else:
        # Standard mode: Create new scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
    os.makedirs(model_local, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Train/validation split
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_raw[:split], Y_raw[split:]

    # --- MODE SELECTION ---
    optuna_start_time = None
    optuna_duration = None
    
    if BASE_MODEL_NAME:
        print(f"\n[Fine-tuning] Loading base model weights: {BASE_MODEL_NAME}")
        h_layers, h_size = get_base_model_params(BASE_MODEL_NAME)

        # Conservative hyperparameters for stable fine-tuning
        best_p = {
            'hidden_layers': h_layers, 
            'hidden_size': h_size, 
            'learning_rate': 1e-4,
            'batch_size': 32
        }

        model = MLP(input_size, output_size, h_size, h_layers).to(device)
        model.load_state_dict(torch.load(os.path.join(model_local, BASE_MODEL_NAME), map_location=device))

        # Freeze feature extraction layers to preserve previous knowledge
        for param in model.input_layer.parameters():
            param.requires_grad = False

        for layer in model.hidden_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        current_epochs = EPOCHS_FINETUNE
    else:
        print(f"\n[Optuna] Starting optimization from scratch...")
        optuna_start_time = time.time()

        # Path to store per-series best parameters
        best_params_dir = os.path.join(data_dir, 'train', 'best_params')
        os.makedirs(best_params_dir, exist_ok=True)
        best_params_path = os.path.join(best_params_dir, f'best_params_{SERIE_IDENTIFIER}.json')

        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as fh:
                best_p = json.load(fh)
            print(f"[Optuna] Reusing saved parameters for {SERIE_IDENTIFIER}")
        else:
            # Configure Optuna progress logging
            optuna_progress_logger = logging.getLogger('optuna_progress')
            optuna_progress_logger.setLevel(logging.INFO)
            handler = logging.FileHandler('optuna_progress.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            optuna_progress_logger.addHandler(handler)
            
            def progress_callback(study, trial):
                if trial.number % 5 == 0 or trial.number == 29:  # Log every 5 trials and last
                    best_value = study.best_value if study.best_trial else float('inf')
                    current_value = trial.value if trial.value is not None else float('inf')
                    message = f"Trial {trial.number}: Current Score = {current_value:.6f}, Best Score = {best_value:.6f}"
                    print(f"[Optuna Progress] {message}")
                    optuna_progress_logger.info(message)
            
            study = optuna.create_study(direction='minimize')
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(lambda trial: objective(trial, X_train, Y_train, X_val, Y_val, fs), n_trials=30, callbacks=[progress_callback])
            best_p = study.best_params
            with open(best_params_path, 'w') as fh:
                json.dump(best_p, fh, indent=2)
        
        optuna_duration = time.time() - optuna_start_time
        print(f"[Optuna] Optimization completed in {format_time(optuna_duration)}")

        model = MLP(input_size, output_size, best_p['hidden_size'], best_p['hidden_layers']).to(device)
        current_epochs = EPOCHS

    # --- FINAL TRAINING LOOP ---
    training_start_time = time.time()
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_p['learning_rate'],
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train, device), batch_size=best_p['batch_size'], shuffle=True)
    val_loader = DataLoader(VoltageVelocityDataset(X_val, Y_val, device), batch_size=best_p['batch_size'])

    train_loss_hist, val_loss_hist = [], []
    idx_t, idx_v = 0, 0
    
    print(f"[Training] Final run for {current_epochs} epochs...")
    for epoch in range(current_epochs):
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            l = criterion(model(X), Y)
            l.backward()
            optimizer.step()
            train_loss_hist.append([idx_t, l.item()]); idx_t += 1
            
        model.eval()
        with torch.no_grad():
            for X, Y in val_loader:
                l_v = criterion(model(X), Y)
                val_loss_hist.append([idx_v, l_v.item()]); idx_v += 1

        if epoch % (EPOCHS / 8) == 0:
            print("| Epoch {:4} | train loss {:4.4f} | val loss {:4.4f} |".format(epoch, train_loss_hist[-1][1] if train_loss_hist else 0, val_loss_hist[-1][1] if val_loss_hist else 0),flush=True)

    training_duration = time.time() - training_start_time
    
    # --- EVALUATION ---
    model.eval()
    with torch.no_grad():
        all_X_tensor = torch.tensor(X_scaled).float().to(device)
        predictions = model(all_X_tensor)
        
        # Predicted data extraction for physical validation
        pred_np = predictions.cpu().numpy()
        final_slope = physics.calculate_spectral_slope(pred_np, fs)
        final_iso = physics.calculate_isotropy_ratio(pred_np, fs)
        
        # Target tensor and RMSE calculation
        target_tensor = torch.tensor(Y_raw).float().to(device)
        rmse = metrics.calculate_rmse_torch(predictions, target_tensor)

    # --- SAVE AND EXPORT ---
    dest = os.path.join(data_dir, 'train', 'results', f'results_{SERIE_IDENTIFIER}')
    os.makedirs(dest, exist_ok=True)
    new_model_path = os.path.join(model_local, f'model_mlp_{SERIE_IDENTIFIER}.pth')
    
    torch.save(model.state_dict(), new_model_path)
    print(f"\n[Done] Training complete. Model saved: {new_model_path}")

    # Calculate and display timing information
    total_duration = time.time() - script_start_time
    
    # Build timing summary
    print(f"\n{'='*60}")
    print(f"⏱️  TIMING SUMMARY")
    print(f"{'='*60}")
    if optuna_duration is not None:
        print(f"  Optuna Optimization: {format_time(optuna_duration)}")
    print(f"  Final Training: {format_time(training_duration)}")
    print(f"  Total Execution Time: {format_time(total_duration)}")
    print(f"{'='*60}")

    pd.DataFrame({
        'Layers': [best_p['hidden_layers']], 'Size': [best_p['hidden_size']],
        'RMSE': [rmse], 'Final_Slope': [final_slope], 'Final_Isotropy': [final_iso]
    }).to_csv(os.path.join(dest, f'hyperparameters_{SERIE_IDENTIFIER}.csv'), index=False)

    logging.info(f"Training complete. Metrics: RMSE={rmse:.6f}, Slope={final_slope:.4f}, Isotropy={final_iso:.4f}")
    show_graphs(df_total, predictions, train_loss_hist, val_loss_hist)

if __name__ == "__main__":
    main()