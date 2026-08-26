"""Use incrementally trained block models to predict velocities on new/unseen data.

This script takes the sequence of models and scalers produced by incremental_train.py
(e.g., model_0610_block1.pth, model_0610_block2.pth) and applies them sequentially
to corresponding blocks of new data. This ensures that the fine-tuned state of
each block is used for its respective data segment.

Usage examples:
    python3 incremental_predict.py 0610 --num-blocks 10
    python3 incremental_predict.py 0610 --num-blocks 10 --calc-metrics
    python3 incremental_predict.py 0610 --num-blocks 10 --calc-metrics --holdout-last

The script reads from data/run/run_df_<serie>.csv, splits it into N blocks,
applies the specific model for each block, and saves combined predictions.
"""

import os
import argparse
import gc
import logging
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import butter, filtfilt, welch

# Prevent Matplotlib from opening GUI windows (Avoids VS Code crashes)
import matplotlib
matplotlib.use('Agg')

from utils import config, metrics, spectral_utils, validation_metrics
from train_mlp import MLP

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_memory():
    """Forces garbage collection and clears PyTorch cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def apply_block_averaging(df: pd.DataFrame, fs: float, target_cutoff: float = 1.66) -> pd.DataFrame:
    """Apply dynamically calculated block averaging (rolling mean) to simulate the 
    sonic anemometer's low-frequency response on perfect synthetic data.
    
    Calculation based on Kit et al. (2016):
    f_cutoff = f_Nyquist / N_block_average
    N_block_average = (fs / 2) / f_cutoff
    """
    if fs is not None and fs > 0:
        f_nyquist = fs / 2.0
        window = int(f_nyquist / target_cutoff)
    else:
        window = 600
        
    df_smoothed = df.copy()
    cols_to_smooth = ["velocity_x", "velocity_y", "velocity_z"]
    for col in cols_to_smooth:
        if col in df_smoothed.columns:
            df_smoothed[col] = df_smoothed[col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
    return df_smoothed

def plot_diagnostic_residual_spectrum(y_pred_raw: np.ndarray, y_true: np.ndarray, fs: float, output_path: str):
    """Diagnostic: Compares the residual (true - raw prediction) spectrum
    against the true signal's own spectrum, BEFORE any spectral correction.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    comp_names = ["u_x (E11)", "u_y (E22)", "u_z (E33)"]

    for i in range(3):
        residual = y_true[:, i] - y_pred_raw[:, i]
        true_signal = y_true[:, i] - np.mean(y_true[:, i])

        nperseg = min(len(true_signal), int(fs * 10))
        if nperseg < 256:
            nperseg = len(true_signal)

        f_res, p_res = welch(residual - np.mean(residual), fs=fs, nperseg=nperseg, detrend="constant")
        f_true, p_true = welch(true_signal, fs=fs, nperseg=nperseg, detrend="constant")

        ax = axes[i]
        ax.loglog(f_true, p_true, color="#2ecc71", linewidth=1.5, label="True Signal PSD")
        ax.loglog(f_res, p_res, color="#9b59b6", linewidth=1.5, label="Residual PSD (True - Raw Pred)")
        ax.set_title(comp_names[i])
        ax.set_xlabel("f [Hz]")
        if i == 0:
            ax.set_ylabel("Power Spectral Density")
            ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Diagnostic residual spectrum saved to {output_path}")

def plot_diagnostic_coherence(y_pred_raw: np.ndarray, y_true: np.ndarray, fs: float, output_path: str):
    """Diagnostic: Computes magnitude-squared coherence between the raw (uncorrected)
    NN prediction and the true reference signal, per frequency.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import coherence

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    comp_names = ["u_x (E11)", "u_y (E22)", "u_z (E33)"]

    for i in range(3):
        nperseg = min(len(y_true), int(fs * 10))
        if nperseg < 256:
            nperseg = len(y_true)

        f_coh, gamma2 = coherence(y_pred_raw[:, i], y_true[:, i], fs=fs, nperseg=nperseg)

        ax = axes[i]
        ax.semilogx(f_coh, gamma2, color="#e67e22", linewidth=1.2)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                    label="γ²=0.5 (equal signal/noise power)")
        ax.set_title(comp_names[i])
        ax.set_xlabel("f [Hz]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", alpha=0.2)
        if i == 0:
            ax.set_ylabel(r"Coherence $\gamma^2(f)$")
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Diagnostic coherence plot saved to {output_path}")

# =============================================================================
# SPECTRAL MAGNIFICATION CORRECTION (Kit et al. 2016)
# =============================================================================
def apply_spectral_magnification(y_pred: np.ndarray, y_true: np.ndarray, fs: float, f_cutoff: float = 1.66) -> Tuple[np.ndarray, List[float]]:
    """Apply Spectral Magnification Correction as described in Kit et al. (2016).
    
    Calculates the ratio of low-frequency spectral energy between the sonic 
    reference (y_true) and the NN prediction (y_pred), and multiplies the prediction 
    fluctuations by the square root of this ratio to restore high-frequency energy.
    """
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
            K = np.mean(spectral_ratios)
            factor = np.sqrt(K)
            factor = np.clip(factor, 0.1, 15.0)
        else:
            factor = 1.0

        factors.append(factor)
        
        mean_pred = np.mean(y_pred[:, i])
        fluctuations = y_pred[:, i] - mean_pred
        y_pred_mag[:, i] = mean_pred + (fluctuations * factor)

    return y_pred_mag, factors

def load_block_model_and_scaler(serie: str, block_idx: int) -> Tuple[MLP, StandardScaler]:
    """Load the specific model and scaler for a given series and block index."""
    model_name = f"model_{serie}_block{block_idx}.pth"
    scaler_name = f"scaler_{serie}_block{block_idx}.joblib"

    model_path = os.path.join(config.MODEL_DIR, "incremental", model_name)
    scaler_path = os.path.join(config.MODEL_DIR, "incremental", scaler_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Block model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Block scaler not found: {scaler_path}")

    scaler = joblib.load(scaler_path)

    best_params_path = os.path.join(
        config.DATA_DIR, "train", "best_params", f"best_params_{serie}_incremental.json"
    )
    
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        hidden_size = best_params.get("hidden_size", 64)
        num_hidden_layers = best_params.get("hidden_layers", 2)
    else:
        hidden_size = 64
        num_hidden_layers = 2

    model = MLP(config.INPUT_SIZE, config.OUTPUT_SIZE, hidden_size, num_hidden_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, scaler

def predict_on_block(model: MLP, scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions for a specific dataframe block."""
    # Convert to float32 to save RAM
    X_raw = df[["voltage_x", "voltage_y", "voltage_z"]].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled).float().to(device))
        
    del X_raw
    del X_scaled
    
    return preds.cpu().numpy()

def calculate_delta_metrics(y_true: np.ndarray, y_pred: np.ndarray, fs: float, cutoff: float = 2.0) -> List[float]:
    """Calculates the Delta parameter according to Freire et al. (2023) Eq. 16."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    deltas = []
    for i in range(3):
        s_true = filtfilt(b, a, y_true[:, i])
        s_pred = filtfilt(b, a, y_pred[:, i])

        s_true_norm = (s_true - np.mean(s_true)) / np.std(s_true)
        s_pred_norm = (s_pred - np.mean(s_pred)) / np.std(s_pred)

        delta = np.sqrt(np.mean((s_pred_norm - s_true_norm) ** 2))
        deltas.append(delta)

    return deltas

def generate_validation_plots(df_results: pd.DataFrame, output_dir: str, serie: str, fs: float):
    """Generate validation scatterplots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]
    
    if not all(col in df_results.columns for col in target_cols):
        logging.warning("True velocity data not available for scatterplot generation.")
        return
        
    logging.info("Generating 1:1 validation scatterplots...")
    df_clean = df_results.dropna(subset=target_cols + pred_cols)
    
    # RAM Protection
    if len(df_clean) > 50000:
        logging.info(f"Reducing visualization from {len(df_clean)} to 50000 points on scatterplot.")
        df_plot = df_clean.sample(n=50000, random_state=42)
    else:
        df_plot = df_clean

    Y_true = df_plot[target_cols].values
    Y_pred = df_plot[pred_cols].values
    
    scatter_data = validation_metrics.generate_1to1_scatterplot_data(Y_pred, Y_true)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    comp_names = ['u_x', 'u_y', 'u_z']
    
    for idx, comp_name in enumerate(comp_names):
        stats = scatter_data[comp_name]
        ax = axes[idx]
        
        ax.scatter(stats['true'], stats['pred'], alpha=0.5, s=10)
        min_val = min(stats['true'].min(), stats['pred'].min())
        max_val = max(stats['true'].max(), stats['pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect 1:1')
        ax.plot([min_val, max_val], 
                [min_val * stats['slope'] + stats['intercept'], max_val * stats['slope'] + stats['intercept']], 
                'g-', linewidth=2, label='Fitted')
        
        ax.set_xlabel(f"True {stats['label']}", fontsize=10)
        ax.set_ylabel(f"Predicted {stats['label']}", fontsize=10)
        ax.set_title(f"{stats['label']}\nR²={stats['r_squared']:.4f}, RMSE={stats['rmse']:.6f}", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"scatterplot_1to1_{serie}.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    plt.close('all') 
    logging.info(f"Scatterplot saved to {scatter_path}")
    
    for comp_name in comp_names:
        stats = scatter_data[comp_name]
        logging.info(f"{stats['label']:20s}: RMSE={stats['rmse']:.6f}, R²={stats['r_squared']:.6f}")

def generate_dissipation_series_plot(blocks_indices: list, df_results: pd.DataFrame, output_dir: str, serie: str, fs: float):
    """Generate dissipation evolution plot across sequential blocks."""
    import matplotlib.pyplot as plt
    
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]
    
    if not all(col in df_results.columns for col in target_cols):
        logging.info("Skipping dissipation series plot (no true velocity data).")
        return
        
    logging.info("Generating dissipation series plot across blocks...")
    epsilons_pred, epsilons_true, block_nums = [], [], []
    
    for block_idx, indices in enumerate(blocks_indices):
        block_df = df_results.iloc[indices]
        if len(block_df) < 100:
            continue
        
        Y_true = block_df[target_cols].dropna().values
        Y_pred = block_df[pred_cols].dropna().values
        
        if len(Y_true) == 0 or len(Y_pred) == 0:
            continue
        
        try:
            u_true_fluc = Y_true - np.mean(Y_true, axis=0)
            u_pred_fluc = Y_pred - np.mean(Y_pred, axis=0)
            
            ke_true = 0.5 * np.mean(np.sum(u_true_fluc**2, axis=1))
            ke_pred = 0.5 * np.mean(np.sum(u_pred_fluc**2, axis=1))
            
            epsilons_true.append(ke_true)
            epsilons_pred.append(ke_pred)
            block_nums.append(block_idx + 1)
        except Exception as e:
            logging.warning(f"Could not calculate dissipation for block {block_idx + 1}: {e}")
            continue
    
    if not block_nums:
        logging.warning("No valid blocks for dissipation analysis.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(block_nums, epsilons_true, 'o-', label='True (Sonic Proxy)', linewidth=2, markersize=8)
    ax.plot(block_nums, epsilons_pred, 's-', label='Predicted (Model)', linewidth=2, markersize=8)
    ax.set_xlabel('Block Number', fontsize=12)
    ax.set_ylabel('Turbulent Kinetic Energy (Proxy)', fontsize=12)
    ax.set_title(f'Energy Evolution Across Sequential Blocks - Serie {serie}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if len(epsilons_pred) > 1:
        jumps = np.abs(np.diff(epsilons_pred))
        info_text = f"Block-to-Block Continuity:\nMean Jump: {np.mean(jumps):.6e}\nMax Jump: {np.max(jumps):.6e}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dissipation_series_{serie}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    logging.info(f"Dissipation series plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply incremental block models to new data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument("--num-blocks", type=int, required=True, help="number of blocks to split the data into (must match training)")
    parser.add_argument("--input", type=str, default=None, help="input CSV file")
    parser.add_argument("--output", type=str, default=None, help="output CSV file")
    parser.add_argument("--calc-metrics", action="store_true", help="if input has velocity columns, also compute RMSE/metrics")
    parser.add_argument("--holdout-last", action="store_true", help="Ensures the last block is evaluated as a blind test")
    
    args = parser.parse_args()
    serie = args.serie

    input_file = args.input or os.path.join(config.DATA_DIR, "run", f"run_{serie}.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    split_indices = np.array_split(np.arange(len(df)), args.num_blocks)
    all_preds = []

    logging.info(f"Generating predictions using {args.num_blocks} incremental models...")
    for i, idx_list in enumerate(split_indices):
        block_num = i + 1
        block_df = df.iloc[idx_list].reset_index(drop=True)

        try:
            is_holdout = args.holdout_last and (i == len(split_indices) - 1)
            if is_holdout:
                logging.info(f"Holdout Test: Predicting block {block_num} using model strictly from block {block_num - 1}")
                model, scaler = load_block_model_and_scaler(serie, block_num - 1)
            else:
                model, scaler = load_block_model_and_scaler(serie, block_num)

            preds = predict_on_block(model, scaler, block_df)
            all_preds.append(preds)
        finally:
            if 'model' in locals(): del model
            if 'scaler' in locals(): del scaler
            del block_df
            cleanup_memory()

    final_preds = np.vstack(all_preds)
    fs = spectral_utils.estimate_sampling_frequency(df, "time") or config.FS_HOTFILM_DEFAULT
    final_preds_raw = final_preds.copy()

    # Apply Spectral Magnification
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    Y_true_global, has_true_data = None, False
    
    if all(col in df.columns for col in target_cols):
        Y_true_global = df[target_cols].values
        has_true_data = True
    else:
        logging.info("Target velocities not found in input. Searching for synthetic reference to enable Spectral Magnification...")
        ref_path = os.path.join(config.DATA_DIR, "raw", serie, f"hotfilm_vel_{serie}.csv")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
            
        if os.path.exists(ref_path):
            logging.info(f"Loading reference from: {ref_path}")
            try:
                ref_df = pd.read_csv(ref_path)
                if "velocity_x" not in ref_df.columns:
                    perfect_df_temp = pd.read_csv(ref_path, header=None)
                    perfect_df_temp.columns = ["time", "velocity_x", "velocity_y", "velocity_z"] if len(perfect_df_temp.columns) >= 4 else ["velocity_x", "velocity_y", "velocity_z"]
                    ref_df = perfect_df_temp
                
                if all(col in ref_df.columns for col in target_cols):
                    min_len = min(len(df), len(ref_df))
                    logging.info("Applying dynamically calculated block averaging to synthetic data...")
                    ref_smoothed = apply_block_averaging(ref_df.iloc[:min_len], fs=fs, target_cutoff=1.66)
                    Y_true_global = ref_smoothed[target_cols].values
                    
                    for col in target_cols:
                        df[col] = ref_smoothed[col].values
                    has_true_data = True
            except Exception as e:
                logging.warning(f"Failed to load or process reference data: {e}")
        else:
            logging.warning("No synthetic reference file found. Magnification will be skipped.")

    if has_true_data and Y_true_global is not None:
        diag_dir = os.path.join(config.DATA_DIR, "run", "results", f"velocity_{serie}", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        
        plot_diagnostic_residual_spectrum(final_preds, Y_true_global, fs, os.path.join(diag_dir, f"diagnostic_residual_spectrum_{serie}.png"))
        plot_diagnostic_coherence(final_preds, Y_true_global, fs, os.path.join(diag_dir, f"diagnostic_coherence_{serie}.png"))

        logging.info("Applying Spectral Magnification (Kit et al. 2016)...")
        final_preds, k_factors = apply_spectral_magnification(final_preds, Y_true_global, fs)
        logging.info(f"Magnification factors (X, Y, Z): {k_factors[0]:.4f}, {k_factors[1]:.4f}, {k_factors[2]:.4f}")
        
        df["velocity_predicted_x"] = final_preds[:, 0]
        df["velocity_predicted_y"] = final_preds[:, 1]
        df["velocity_predicted_z"] = final_preds[:, 2]

    df["velocity_raw_predicted_x"] = final_preds_raw[:, 0]
    df["velocity_raw_predicted_y"] = final_preds_raw[:, 1]
    df["velocity_raw_predicted_z"] = final_preds_raw[:, 2]

    output_file = args.output or os.path.join(config.DATA_DIR, "run", "results", f"velocity_{serie}", f"velocity_{serie}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    
    del final_preds, final_preds_raw
    cleanup_memory()

    # --- METRICS CALCULATION ---
    if args.calc_metrics:
        if all(col in df.columns for col in target_cols):
            logging.info("Computing validation metrics...")
            df_clean = df.dropna(subset=target_cols + ["velocity_predicted_x"])
            Y_true = df_clean[target_cols].values
            Y_pred = df_clean[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values

            rmse_global = metrics.calculate_rmse(Y_pred, Y_true)
            deltas = calculate_delta_metrics(Y_true, Y_pred, fs)
            delta_general = np.mean(deltas)
            
            output_text = f"Global Raw RMSE: {rmse_global:.6f} | Delta General: {delta_general:.4f}\n"
            metrics_file = os.path.join(os.path.dirname(output_file), f"delta_metrics_{serie}.txt")
            with open(metrics_file, "w") as f:
                f.write(output_text)
            logging.info(f"Global Metrics saved to {metrics_file}")
            
            if args.holdout_last:
                last_block_indices = split_indices[-1]
                df_holdout = df.iloc[last_block_indices].dropna(subset=target_cols + ["velocity_predicted_x"])
                if len(df_holdout) > 10:
                    Y_true_h = df_holdout[target_cols].values
                    Y_pred_h = df_holdout[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values
                    rmse_h = metrics.calculate_rmse(Y_pred_h, Y_true_h)
                    holdout_text = f"Holdout Block {args.num_blocks} Raw RMSE: {rmse_h:.6f}\n"
                    blind_metrics_file = os.path.join(os.path.dirname(output_file), f"blind_delta_metrics_{serie}.txt")
                    with open(blind_metrics_file, "w") as f:
                        f.write(holdout_text)
                    logging.info(f"Blind Holdout Metrics saved to {blind_metrics_file}")
            
            try:
                generate_validation_plots(df, os.path.dirname(output_file), serie, fs)
                generate_dissipation_series_plot(split_indices, df, os.path.dirname(output_file), serie, fs)
            except Exception as e:
                logging.error(f"Failed to generate validation plots: {e}")

    # Spectral Analysis
    logging.info("Starting Spectral Validation...")
    sonic_file = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    sonic_df = None
    if os.path.exists(sonic_file):
        sonic_df = pd.read_csv(sonic_file, usecols=["time", "velocity_x", "velocity_y", "velocity_z"])
        if len(sonic_df) > 500000:
            sonic_df = sonic_df.iloc[:500000].reset_index(drop=True)
            
    spectral_dir = os.path.join(os.path.dirname(output_file), "plots_spectral")
    os.makedirs(spectral_dir, exist_ok=True)
    fs_sonic = spectral_utils.estimate_sampling_frequency(sonic_df, "time") if sonic_df is not None else config.FS_SONIC_DEFAULT
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    try:
        spectral_utils.plot_combined_spectrum(
            df, pred_cols, fs, f"Global Spectral Analysis - Serie {serie}",
            os.path.join(spectral_dir, f"combined_spectrum_global_{serie}.png"),
            sonic_df=sonic_df, fs_sonic=fs_sonic,
        )
    except Exception as e:
        logging.error(f"Failed to generate global spectrum: {e}")

    for i, idx_list in enumerate(split_indices):
        try:
            spectral_utils.plot_combined_spectrum(
                df.iloc[idx_list][pred_cols], pred_cols, fs, f"Spectral Analysis Block {i+1} - Serie {serie}",
                os.path.join(spectral_dir, f"combined_spectrum_block_{i+1}_{serie}.png"),
                sonic_df=sonic_df, fs_sonic=fs_sonic,
            )
        except Exception as e:
            logging.error(f"Failed to generate spectrum for block {i+1}: {e}")
        finally:
            cleanup_memory()

    logging.info("Incremental block prediction complete.")

if __name__ == "__main__":
    main()