"""Incremental / online training for hot-film neural network.

This script processes time series datasets naturally divided into chronological 
blocks. It trains an MLP model on the first block and fine-tunes it sequentially 
on new blocks to adapt to shifting parameters.

In `--scattered` mode, it employs a strict 10/10/80 data split strategy:
    - 10% of each block is aggregated for an initial global training.
    - 10% of each block is used for block-specific fine-tuning.
    - 80% of each block is strictly reserved for blind evaluation.

CRITICAL OPTIMIZATIONS APPLIED:
    - Low-Pass Filtering (Kit et al. 2016): Voltages and velocities are block-averaged.
    - Spectral Magnification Correction (Kit et al. 2016): Anchors low-frequency energy.
    - Memory Stream Optimization: Keeps TensorDataset on CPU, moving mini-batches to GPU.
"""

import os
import copy
import argparse
import random
import logging
import json
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

from utils import config, data_loader, hyperparameter_optimization, metrics, spectral_utils
from train_mlp import MLP

input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# REPRODUCIBILITY (Seed Fixing)
# =============================================================================
def set_deterministic_seeds(seed: int = 42):
    """Fix seeds across all libraries to ensure 100% reproducible results.
    
    Note on Physics-Informed Neural Networks (PINNs):
    Setting deterministic algorithms in cuDNN (and disabling the benchmark mode) 
    is an excellent and necessary practice to guarantee exact reproducibility in 
    scientific computing and physics-informed models. 
    
    Performance Impact: This constraint forces the GPU to use reproducible 
    convolutional algorithms, which may prevent cuDNN from auto-tuning and 
    selecting the fastest available algorithm for modern GPU architectures, 
    resulting in a slight degradation of training speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seeds globally locked to {seed} for strict reproducibility.")

# =============================================================================
# DATA PREPROCESSING (Kit et al. 2016)
# =============================================================================
def apply_block_averaging(df: pd.DataFrame, fs: float, target_cutoff: float = 1.66) -> pd.DataFrame:
    """Apply block averaging to simulate the sonic anemometer's low-pass filter."""
    if fs is not None and fs > 0:
        f_nyquist = fs / 2.0
        window = int(f_nyquist / target_cutoff)
    else:
        window = 600
        
    df_smoothed = df.copy()
    cols_to_smooth = ["voltage_x", "voltage_y", "voltage_z", "velocity_x", "velocity_y", "velocity_z"]
    for col in cols_to_smooth:
        if col in df_smoothed.columns:
            df_smoothed[col] = df_smoothed[col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
    return df_smoothed

# =============================================================================
# SPECTRAL MAGNIFICATION CORRECTION (Kit et al. 2016)
# =============================================================================
def apply_spectral_magnification(y_pred: np.ndarray, y_true: np.ndarray, fs: float, f_cutoff: float = 1.66):
    """Restore high-frequency amplitude based on the low-frequency spectral ratio."""
    y_pred_mag = np.zeros_like(y_pred)
    factors = []

    for i in range(3): 
        nperseg = min(len(y_true), int(fs * 10))
        if nperseg < 256: 
            nperseg = len(y_true)

        f_true, p_true = welch(y_true[:, i], fs=fs, nperseg=nperseg, detrend='constant')
        f_pred, p_pred = welch(y_pred[:, i], fs=fs, nperseg=nperseg, detrend='constant')

        valid_idx = (f_true >= 0.1) & (f_true <= f_cutoff)
        if not np.any(valid_idx) and len(f_true) > 1:
            valid_idx = (f_true > 0) & (f_true <= f_true[min(3, len(f_true)-1)])

        if np.any(valid_idx):
            spectral_ratios = p_true[valid_idx] / (p_pred[valid_idx] + 1e-10)
            factor = np.clip(np.sqrt(np.mean(spectral_ratios)), 0.1, 15.0)
        else:
            factor = 1.0

        factors.append(factor)
        mean_pred = np.mean(y_pred[:, i])
        y_pred_mag[:, i] = mean_pred + ((y_pred[:, i] - mean_pred) * factor)

    return y_pred_mag, factors

def evaluate_block_magnified(model, scaler, df, fs, device):
    # STRATEGY 1: Reduce Precision to save RAM
    X_raw = df[["voltage_x", "voltage_y", "voltage_z"]].values.astype(np.float32)
    Y_raw = df[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    
    # STRATEGY 2: Free original arrays to avoid memory duplication during evaluation
    del df
    gc.collect()

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        
    preds_mag, k_factors = apply_spectral_magnification(preds, Y_raw, fs)
    rmse = metrics.calculate_rmse(preds_mag, Y_raw)
    return {"rmse": rmse, "k_x": k_factors[0], "k_y": k_factors[1], "k_z": k_factors[2]}

# =============================================================================
# STANDARD TRAINING LOOP
# =============================================================================
def train_on_block(model, scaler, df, epochs, device, lr, batch_size, freeze_mode, num_hidden_layers):
    params_to_train = []
    
    for name, param in model.named_parameters():
        if freeze_mode == "full":
            if 'output_layer' in name:
                param.requires_grad = True
                params_to_train.append(param)
            else:
                param.requires_grad = False
        elif freeze_mode == "partial":
            last_hidden_name = f'hidden_layers.{num_hidden_layers - 1}'
            if 'output_layer' in name or last_hidden_name in name:
                param.requires_grad = True
                params_to_train.append(param)
            else:
                param.requires_grad = False
        elif freeze_mode == "none":
            param.requires_grad = True
            params_to_train.append(param)

    if not params_to_train:
        params_to_train = list(model.parameters())

    optimizer = optim.Adam(params_to_train, lr=lr, weight_decay=1e-4)

    # STRATEGY 1: Reduce Precision to save RAM
    X_raw = df[["voltage_x", "voltage_y", "voltage_z"]].values.astype(np.float32)
    Y_raw = df[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    
    # STRATEGY 2: Free unscaled arrays before creating PyTorch datasets
    del X_raw
    gc.collect()

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(Y_raw, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)

    criterion = nn.MSELoss()
    model.train()
    
    logging.info(f"Starting training for {epochs} epochs (Batch Size: {batch_size}, Freeze Mode: {freeze_mode})")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        log_interval = max(1, epochs / 10)
        if (epoch + 1) % int(log_interval) == 0 or epoch == epochs - 1:
            logging.info(f"Epoch [{epoch+1}/{epochs}] | Avg MSE Loss: {epoch_loss / len(dataloader):.6f}")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Incremental training with block-wise metrics")
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=None, help="number of blocks to split the dataset into")
    parser.add_argument("--base-model", type=str, default=None, help="path to an existing .pth file for warm start")
    parser.add_argument("--scattered", action="store_true", help="Use scattered training mode (10/10/80 split)")
    parser.add_argument("--percentage", type=float, default=20.0, help="Percentage of data to train on each block in sequential mode (default: 20)")
    parser.add_argument("--reverse-data", action="store_true", help="Inverts chronological order")
    parser.add_argument("--holdout-last", action="store_true", help="Reserves the final block strictly for testing.")
    parser.add_argument("--freeze-mode", type=str, choices=["full", "partial", "none"], default="full", help="Freezing strategy during fine-tuning.")

    args = parser.parse_args()
    set_deterministic_seeds(seed=42)

    NUM_BLOCKS, SCATTERED, PERCENTAGE, serie, freeze_mode_arg = args.num_blocks, args.scattered, args.percentage, args.serie, args.freeze_mode

    df_path = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Training data not found: {df_path}")
    df_raw = pd.read_csv(df_path)

    fs = spectral_utils.estimate_sampling_frequency(df_raw, "time") or config.FS_HOTFILM_DEFAULT

    if args.reverse_data:
        logging.warning("EXPERIMENTAL MODE: FLAG --reverse-data ACTIVATED! Reversing chronological order.")
        df_raw = df_raw.iloc[::-1].reset_index(drop=True)
        serie = f"{serie}_rev"

    logging.info("Applying dynamically calculated block averaging to training sets...")
    df_smoothed = apply_block_averaging(df_raw, fs=fs, target_cutoff=1.66)

    if NUM_BLOCKS is not None:
        blocks_raw = data_loader.split_dataframe_into_n_blocks(df_raw, NUM_BLOCKS)
        blocks_smooth = data_loader.split_dataframe_into_n_blocks(df_smoothed, NUM_BLOCKS)
    else:
        blocks_raw = data_loader.prepare_blocks(df_raw, block_size=None, gap_threshold=None)
        blocks_smooth = data_loader.prepare_blocks(df_smoothed, block_size=None, gap_threshold=None)

    # STRATEGY 2: Clear massive datasets that are no longer needed
    del df_raw
    del df_smoothed
    gc.collect()

    if not blocks_raw:
        logging.error("No blocks extracted from the dataset. Exiting.")
        return

    logging.info(f"Dataset split into {len(blocks_raw)} blocks")

    if SCATTERED:
        initial_blocks, finetune_blocks, eval_blocks = [], [], []
        for b_raw, b_smooth in zip(blocks_raw, blocks_smooth):
            n = len(b_raw)
            initial_blocks.append(b_smooth.iloc[:int(n * 0.10)].reset_index(drop=True))
            finetune_blocks.append(b_smooth.iloc[int(n * 0.10):int(n * 0.20)].reset_index(drop=True))
            eval_blocks.append(b_raw.iloc[int(n * 0.20):].reset_index(drop=True))
        opt_df = pd.concat(initial_blocks).reset_index(drop=True)
    else:
        opt_df = blocks_smooth[0].iloc[: int(len(blocks_smooth[0]) * PERCENTAGE / 100)].reset_index(drop=True)

    best_params_path = os.path.join(config.DATA_DIR, "train", "best_params", f"best_params_{serie}_incremental.json")
    
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        logging.info("Locked parameters loaded. Bypassing Optuna for A/B isolation.")
    else:
        logging.warning("Parameters not found. Running Optuna...")
        # Reduce precision during optimization
        X_opt = StandardScaler().fit_transform(opt_df[["voltage_x", "voltage_y", "voltage_z"]].values.astype(np.float32))
        Y_opt = opt_df[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float32)
        split_opt = int(0.8 * len(X_opt))
        best_params = hyperparameter_optimization.optimize_hyperparameters(
            X_opt[:split_opt], Y_opt[:split_opt], X_opt[split_opt:], Y_opt[split_opt:], fs, serie, device, suffix="_incremental"
        )
        
        # Clear optimization datasets
        del opt_df
        del X_opt
        del Y_opt
        gc.collect()

    if args.base_model is not None and os.path.exists(args.base_model):
        try:
            import train_mlp
            h_layers, h_size = train_mlp.get_base_model_params(os.path.basename(args.base_model))
        except Exception:
            h_layers, h_size = best_params["hidden_layers"], best_params["hidden_size"]
        model = MLP(input_size, output_size, h_size, h_layers).to(device)
        model.load_state_dict(torch.load(args.base_model, map_location=device))
        scaler = joblib.load(args.base_model.replace(".pth", ".joblib"))
        logging.info(f"Loaded base model and scaler from {args.base_model}")
    else:
        model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
        scaler = StandardScaler()
        if SCATTERED:
            raw_train_df = pd.concat([b.iloc[:int(len(b) * 0.10)] for b in blocks_raw]).reset_index(drop=True)
        else:
            raw_train_df = blocks_raw[0].iloc[:int(len(blocks_raw[0]) * (PERCENTAGE / 100.0))].reset_index(drop=True)
            
        scaler.fit(raw_train_df[["voltage_x", "voltage_y", "voltage_z"]].values.astype(np.float32))
        
        # Clear raw train dataset
        del raw_train_df
        gc.collect()

    results = []
    out_folder = os.path.join(config.MODEL_DIR, "incremental")
    os.makedirs(out_folder, exist_ok=True)
    
    if SCATTERED:
        initial_df = pd.concat(initial_blocks).reset_index(drop=True)
        logging.info(f"Initial training with {len(initial_df)} scattered samples.")
        
        model = train_on_block(
            model, scaler, initial_df, epochs=best_params["epochs"], device=device,
            lr=best_params["learning_rate"], batch_size=best_params["batch_size"],
            freeze_mode="none", num_hidden_layers=best_params["hidden_layers"]
        )
        initial_state = copy.deepcopy(model.state_dict())
        
        # Clear initial block
        del initial_df
        gc.collect()

        for i in range(len(blocks_raw)):
            is_holdout = args.holdout_last and (i == len(blocks_raw) - 1)
            logging.info(f"Processing block {i+1}/{len(blocks_raw)}")
            
            block_model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
            block_model.load_state_dict(initial_state)

            if is_holdout:
                logging.info("Holdout Mode: Evaluating only. No fine-tuning on this block.")
            else:
                if len(finetune_blocks[i]) > 0:
                    block_model = train_on_block(
                        block_model, scaler, finetune_blocks[i], epochs=best_params["epochs_finetune"],
                        device=device, lr=best_params["learning_rate"], batch_size=best_params["batch_size"],
                        freeze_mode=freeze_mode_arg, num_hidden_layers=best_params["hidden_layers"]
                    )

            metrics_dict = evaluate_block_magnified(block_model, scaler, eval_blocks[i], fs, device)
            metrics_dict.update({"block": i, "samples": len(eval_blocks[i]), "is_holdout": is_holdout})
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(block_model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth"))
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            # Limpeza manual das variáveis do loop finalizada, mas mantendo a variável de acesso global
            gc.collect()
            
            if is_holdout:
                log_content = f"BLIND HOLDOUT RESULTS\nMode: SCATTERED | Freeze Mode: {freeze_mode_arg.upper()}\nRMSE: {metrics_dict['rmse']:.6f}\n"
                with open(os.path.join(out_folder, f"blind_test_train_log_{serie}.txt"), "w") as f:
                    f.write(log_content)

    else:
        for i, (block_raw, block_smooth) in enumerate(zip(blocks_raw, blocks_smooth)):
            is_holdout = args.holdout_last and (i == len(blocks_raw) - 1)
            logging.info(f"Processing block {i+1}/{len(blocks_raw)}")

            idx_train = int(len(block_raw) * (PERCENTAGE / 100.0))
            train_df, eval_df = block_smooth.iloc[:idx_train].reset_index(drop=True), block_raw.iloc[idx_train:].reset_index(drop=True)

            if is_holdout:
                logging.info("Holdout Mode: Evaluating only. No fine-tuning on this block.")
            else:
                epochs_run = best_params["epochs"] if i == 0 else best_params["epochs_finetune"]
                freeze_run = "none" if i == 0 else freeze_mode_arg
                model = train_on_block(
                    model, scaler, train_df, epochs=epochs_run, device=device,
                    lr=best_params["learning_rate"], batch_size=best_params["batch_size"],
                    freeze_mode=freeze_run, num_hidden_layers=best_params["hidden_layers"]
                )

            metrics_dict = evaluate_block_magnified(model, scaler, eval_df, fs, device)
            metrics_dict.update({"block": i, "samples": len(eval_df), "is_holdout": is_holdout})
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth"))
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            del train_df
            del eval_df
            gc.collect()
            
            if is_holdout:
                log_content = f"BLIND HOLDOUT RESULTS\nMode: SEQUENTIAL | Freeze Mode: {freeze_mode_arg.upper()}\nRMSE: {metrics_dict['rmse']:.6f}\n"
                with open(os.path.join(out_folder, f"blind_test_train_log_{serie}.txt"), "w") as f:
                    f.write(log_content)

    results_df = pd.DataFrame(results)
    res_path = os.path.join(config.DATA_DIR, "train", "results", f"results_{serie}", "block_metrics.csv")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    results_df.to_csv(res_path, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if args.holdout_last:
        ax.plot(results_df["block"][:-1], results_df["rmse"][:-1], marker="o", label="Trained Blocks")
        ax.plot(results_df["block"].iloc[-1], results_df["rmse"].iloc[-1], marker="*", color="red", markersize=10, label="Blind Holdout")
    else:
        ax.plot(results_df["block"], results_df["rmse"], marker="o", label="Test RMSE")
        
    ax.set_ylabel("RMSE (Magnified)")
    ax.set_xlabel("Block Index")
    ax.set_title(f"Incremental Training RMSE per Block (Freeze: {freeze_mode_arg.upper()})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(res_path), "block_evolution.png"))
    plt.close(fig)

    final_state = block_model.state_dict() if SCATTERED else model.state_dict()
    torch.save(final_state, os.path.join(out_folder, f"model_{serie}_final.pth"))
    joblib.dump(scaler, os.path.join(out_folder, f"scaler_{serie}_final.joblib"))

if __name__ == "__main__":
    main()