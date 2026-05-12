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
import json
import argparse
import gc  # Garbarge collector for RAM protection
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import butter, filtfilt

# Prevent Matplotlib from opening GUI windows (Avoids VS Code crashes)
import matplotlib
matplotlib.use('Agg')

from utils import config, metrics, physics, spectral_utils, validation_metrics
from train_mlp import MLP

device = torch.device("cpu")

def cleanup_memory():
    """Forces garbage collection and clears PyTorch cache to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_block_model_and_scaler(
    serie: str, block_idx: int
) -> Tuple[MLP, StandardScaler]:
    """Load the specific model and scaler for a given series and block index.

    Args:
        serie: series identifier (e.g. 0610).
        block_idx: The 1-based index of the block.

    Returns:
        Tuple of (Loaded MLP model, Loaded StandardScaler).
    """
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
    model: MLP, scaler: StandardScaler, df: pd.DataFrame
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


def calculate_delta_metrics(y_true, y_pred, fs, cutoff=2.0):
    """
    Calculates the Delta parameter according to Freire et al. (2023) Eq. 16.
    y_true/y_pred: arrays (N, 3)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Butterworth filter of 4th order
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    deltas = []
    for i in range(3):
        # Zero-phase filtfilt filter (does not delay the signal)
        s_true = filtfilt(b, a, y_true[:, i])
        s_pred = filtfilt(b, a, y_pred[:, i])

        # Normalization (Z-score)
        s_true_norm = (s_true - np.mean(s_true)) / np.std(s_true)
        s_pred_norm = (s_pred - np.mean(s_pred)) / np.std(s_pred)

        # RMS of the difference
        delta = np.sqrt(np.mean((s_pred_norm - s_true_norm) ** 2))
        deltas.append(delta)

    return deltas


def generate_validation_plots(
    df_results: pd.DataFrame,
    output_dir: str,
    serie: str,
    fs: float
):
    """Generate validation plots: scatterplots and dissipation series."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if true velocity columns exist
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]
    
    has_true_data = all(col in df_results.columns for col in target_cols)
    
    if has_true_data:
        print(f"\n{'='*60}")
        print("📊 GENERATING VALIDATION PLOTS")
        print(f"{'='*60}")
        
        # Clean data for visualization
        df_clean = df_results.dropna(subset=target_cols + pred_cols)
        
        # RAM Protection - Subsample if data is too large to avoid Matplotlib OOM
        if len(df_clean) > 50000:
            print(f"  [RAM Protection] Reducing visualization from {len(df_clean)} to 50000 points on scatterplot.")
            df_plot = df_clean.sample(n=50000, random_state=42)
        else:
            df_plot = df_clean

        Y_true = df_plot[target_cols].values
        Y_pred = df_plot[pred_cols].values
        
        # --- 1. Scatterplot 1:1 Comparison ---
        print("Generating 1:1 velocity scatterplots...")
        scatter_data = validation_metrics.generate_1to1_scatterplot_data(Y_pred, Y_true)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        comp_names = ['u_x', 'u_y', 'u_z']
        for idx, comp_name in enumerate(comp_names):
            stats = scatter_data[comp_name]
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(stats['true'], stats['pred'], alpha=0.5, s=10)
            
            # Perfect 1:1 reference line
            min_val = min(stats['true'].min(), stats['pred'].min())
            max_val = max(stats['true'].max(), stats['pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect 1:1')
            
            # Fitted regression line
            ax.plot([min_val, max_val], 
                   [min_val * stats['slope'] + stats['intercept'], 
                    max_val * stats['slope'] + stats['intercept']], 
                   'g-', linewidth=2, label='Fitted')
            
            ax.set_xlabel(f"True {stats['label']}", fontsize=10)
            ax.set_ylabel(f"Predicted {stats['label']}", fontsize=10)
            ax.set_title(f"{stats['label']}\nR²={stats['r_squared']:.4f}, RMSE={stats['rmse']:.6f}", 
                        fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, f"scatterplot_1to1_{serie}.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        plt.close('all') 
        print(f"  ✓ Saved to {scatter_path}")
        
        # Print scatterplot statistics
        print("\n  Scatterplot Statistics:")
        for comp_name in comp_names:
            stats = scatter_data[comp_name]
            print(f"    {stats['label']:20s}: RMSE={stats['rmse']:.6f}, R²={stats['r_squared']:.6f}")
            
        # Free memory
        del df_clean, df_plot, Y_true, Y_pred
        cleanup_memory()
    else:
        print(f"\n[Warning] True velocity data not available for scatterplot generation.")
    
    print(f"{'='*60}\n")


def generate_dissipation_series_plot(
    blocks_indices: list,
    df_results: pd.DataFrame,
    output_dir: str,
    serie: str,
    fs: float
):
    """Generate dissipation evolution plot across blocks."""
    import matplotlib.pyplot as plt
    from scipy.signal import periodogram
    
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]
    
    has_true_data = all(col in df_results.columns for col in target_cols)
    
    if not has_true_data:
        print("[Info] Skipping dissipation series plot (no true velocity data).")
        return
    
    print("Generating dissipation series plot across blocks...")
    
    epsilons_pred = []
    epsilons_true = []
    block_nums = []
    
    for block_idx, indices in enumerate(blocks_indices):
        block_df = df_results.iloc[indices]
        
        # Skip blocks with insufficient data
        if len(block_df) < 100:
            continue
        
        Y_true = block_df[target_cols].dropna().values
        Y_pred = block_df[pred_cols].dropna().values
        
        if len(Y_true) == 0 or len(Y_pred) == 0:
            continue
        
        try:
            u_true_fluc = Y_true - np.mean(Y_true, axis=0)
            u_pred_fluc = Y_pred - np.mean(Y_pred, axis=0)
            
            # Total kinetic energy (proxy for dissipation rate)
            ke_true = 0.5 * np.mean(np.sum(u_true_fluc**2, axis=1))
            ke_pred = 0.5 * np.mean(np.sum(u_pred_fluc**2, axis=1))
            
            epsilons_true.append(ke_true)
            epsilons_pred.append(ke_pred)
            block_nums.append(block_idx + 1)
        except Exception as e:
            print(f"  Warning: Could not calculate dissipation for block {block_idx + 1}: {e}")
            continue
    
    if len(block_nums) == 0:
        print("[Warning] No valid blocks for dissipation analysis.")
        return
    
    # Plot dissipation evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(block_nums, epsilons_true, 'o-', label='True (Sonic)', linewidth=2, markersize=8)
    ax.plot(block_nums, epsilons_pred, 's-', label='Predicted (Model)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Block Number', fontsize=12)
    ax.set_ylabel('Turbulent Kinetic Energy (Proxy)', fontsize=12)
    ax.set_title(f'Energy Evolution Across Sequential Blocks - Serie {serie}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add continuity analysis
    if len(epsilons_pred) > 1:
        jumps = np.abs(np.diff(epsilons_pred))
        mean_jump = np.mean(jumps)
        max_jump = np.max(jumps)
        
        info_text = f"Block-to-Block Continuity:\nMean Jump: {mean_jump:.6e}\nMax Jump: {max_jump:.6e}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dissipation_series_{serie}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    plt.close('all')
    print(f"  ✓ Saved to {plot_path}")
    cleanup_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Apply incremental block models to new data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument(
        "--num-blocks",
        type=int,
        required=True,
        help="number of blocks to split the data into (must match training)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="input CSV file (default: data/run/run_<serie>.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output CSV file (default: data/run/results/<serie>/predictions.csv)",
    )
    parser.add_argument(
        "--calc-metrics",
        action="store_true",
        help="if input has velocity columns, also compute RMSE/metrics",
    )
    # INJECTION: Holdout Validation Flag
    parser.add_argument(
        "--holdout-last",
        action="store_true",
        help="Ensures the last block is evaluated as a completely blind test using the (N-1) model.",
    )
    
    args = parser.parse_args()

    serie = args.serie

    # Determine input file
    if args.input is None:
        input_file = os.path.join(config.DATA_DIR, "run", f"run_{serie}.csv")
    else:
        input_file = args.input

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Ensure reynolds column exists
    if "reynolds" not in df.columns:
        cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
        re_val = 0.0
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                    re_val = cfg.get("RE_NUMBER", 0.0)
            except Exception:
                pass
        df["reynolds"] = re_val
        print(f"[Info] added missing reynolds column = {re_val}")

    # Split dataframe into N blocks to match training stages
    indices = np.arange(len(df))
    split_indices = np.array_split(indices, args.num_blocks)

    all_preds = []

    # Process each block with its corresponding model
    print(f"Generating predictions using {args.num_blocks} incremental models...")
    for i, idx_list in enumerate(split_indices):
        block_num = i + 1
        block_df = df.iloc[idx_list].reset_index(drop=True)

        try:
            # INJECTION: Holdout logic - if it's the last block, load the penultimate model to ensure a blind test.
            is_holdout = args.holdout_last and (i == len(split_indices) - 1)
            
            if is_holdout:
                print(f"--> 🛡️ [Holdout Test] Predicting block {block_num} using model strictly from block {block_num - 1}")
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

    # Combine all block predictions
    final_preds = np.vstack(all_preds)

    # Add predictions to dataframe
    df["velocity_predicted_x"] = final_preds[:, 0]
    df["velocity_predicted_y"] = final_preds[:, 1]
    df["velocity_predicted_z"] = final_preds[:, 2]

    # Determine output file
    if args.output is None:
        output_dir = os.path.join(
            config.DATA_DIR, "run", "results", f"velocity_{serie}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"velocity_{serie}.csv")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = args.output

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    del final_preds
    cleanup_memory()

    # --- METRICS CALCULATION ---
    if args.calc_metrics:
        target_cols = ["velocity_x", "velocity_y", "velocity_z"]

        if all(col in df.columns for col in target_cols):
            print(f"\n{'='*50}")
            print(f"📊 VALIDATION: ARTICLE METRICS")
            print(f"{'='*50}")

            # Data cleaning and extraction
            df_clean = df.dropna(subset=target_cols + ["velocity_predicted_x"])
            Y_true = df_clean[target_cols].values
            Y_pred = df_clean[
                ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]
            ].values

            # Get FS for the filter
            with open(
                os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
            ) as fh:
                cfg = json.load(fh)
            fs = cfg["FS_HOTFILM"]

            # 1. Raw RMSE (point-to-point)
            rmse_global = metrics.calculate_rmse(Y_pred, Y_true)

            # 2. Delta Parameter (Signal filtered at 2Hz and normalized)
            deltas = calculate_delta_metrics(Y_true, Y_pred, fs)
            delta_general = np.mean(deltas)

            # 3. Skewness Parameter (Derivative distribution)
            # 3. Skewness Parameter (Derivative distribution) - Calculado de forma segura por blocos
            sk_preds_list = []
            sk_trues_list = []

            for idx_list in split_indices:
                block_clean = df.iloc[idx_list].dropna(subset=target_cols + ["velocity_predicted_x"])
                if len(block_clean) > 10:  # Garante que há pontos suficientes para a derivada
                    b_true = block_clean[target_cols].values
                    b_pred = block_clean[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values
                    
                    sk_preds_list.append(validation_metrics.calculate_velocity_derivative_skewness(b_pred, fs))
                    sk_trues_list.append(validation_metrics.calculate_velocity_derivative_skewness(b_true, fs))

            # Tira a média dos blocos para evitar outliers de descontinuidade
            skewness_pred = {
                key: np.nanmean([d[key] for d in sk_preds_list]) for key in sk_preds_list[0]
            } if sk_preds_list else {'u_longitudinal': 0, 'u_lateral': 0, 'u_vertical': 0}

            skewness_true = {
                key: np.nanmean([d[key] for d in sk_trues_list]) for key in sk_trues_list[0]
            } if sk_trues_list else {'u_longitudinal': 0, 'u_lateral': 0, 'u_vertical': 0}

            # Prepare output text for GLOBAL dataset
            output_text = f"{'='*50}\n"
            output_text += f"📊 VALIDATION: ARTICLE METRICS (Global Dataset)\n"
            output_text += f"{'='*50}\n\n"
            output_text += f"Records evaluated: {len(df_clean)}\n"
            output_text += f"Global Raw RMSE:  {rmse_global:.6f}\n"
            output_text += f"{'-'*50}\n"
            output_text += f"Delta_u1 (X-axis):  {deltas[0]:.4f}\n"
            output_text += f"Delta_u2 (Y-axis):  {deltas[1]:.4f}\n"
            output_text += f"Delta_u3 (Z-axis):  {deltas[2]:.4f}\n"
            output_text += f"Delta General:      {delta_general:.4f}\n"
            output_text += f"{'-'*50}\n"
            output_text += f"Skewness S_k (Pred vs True) [Expected u1 ~ -0.3]:\n"
            output_text += f"u1 (Longitudinal):  Pred={skewness_pred['u_longitudinal']:7.4f} | True={skewness_true['u_longitudinal']:7.4f}\n"
            output_text += f"u2 (Lateral):       Pred={skewness_pred['u_lateral']:7.4f} | True={skewness_true['u_lateral']:7.4f}\n"
            output_text += f"u3 (Vertical):      Pred={skewness_pred['u_vertical']:7.4f} | True={skewness_true['u_vertical']:7.4f}\n"
            output_text += f"{'-'*50}\n"

            print(output_text)
            
            # Save global metrics file
            metrics_file = os.path.join(output_dir, f"delta_metrics_{serie}.txt")
            with open(metrics_file, "w") as f:
                f.write(output_text)
            print(f"Global Metrics saved to {metrics_file}")
            
            # INJECTION: Isolate and calculate metrics STRICTLY for the blind holdout block
            if args.holdout_last:
                print(f"\n{'='*50}")
                print(f"🛡️ VALIDATION: BLIND HOLDOUT BLOCK ONLY")
                print(f"{'='*50}")
                
                last_block_indices = split_indices[-1]
                df_holdout = df.iloc[last_block_indices].dropna(subset=target_cols + ["velocity_predicted_x"])
                
                if len(df_holdout) > 0:
                    Y_true_h = df_holdout[target_cols].values
                    Y_pred_h = df_holdout[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values
                    
                    rmse_h = metrics.calculate_rmse(Y_pred_h, Y_true_h)
                    deltas_h = calculate_delta_metrics(Y_true_h, Y_pred_h, fs)
                    delta_general_h = np.mean(deltas_h)
                    skewness_pred_h = validation_metrics.calculate_velocity_derivative_skewness(Y_pred_h, fs)
                    skewness_true_h = validation_metrics.calculate_velocity_derivative_skewness(Y_true_h, fs)
                    
                    holdout_text = f"{'='*50}\n"
                    holdout_text += f"🛡️ BLIND HOLDOUT METRICS (Untainted Data)\n"
                    holdout_text += f"{'='*50}\n\n"
                    holdout_text += f"Records evaluated: {len(df_holdout)} (Block {args.num_blocks})\n"
                    holdout_text += f"Holdout Raw RMSE:  {rmse_h:.6f}\n"
                    holdout_text += f"{'-'*50}\n"
                    holdout_text += f"Delta_u1 (X-axis):  {deltas_h[0]:.4f}\n"
                    holdout_text += f"Delta_u2 (Y-axis):  {deltas_h[1]:.4f}\n"
                    holdout_text += f"Delta_u3 (Z-axis):  {deltas_h[2]:.4f}\n"
                    holdout_text += f"Delta General:      {delta_general_h:.4f}\n"
                    holdout_text += f"{'-'*50}\n"
                    holdout_text += f"Skewness S_k (Pred vs True):\n"
                    holdout_text += f"u1 (Longitudinal):  Pred={skewness_pred_h['u_longitudinal']:7.4f} | True={skewness_true_h['u_longitudinal']:7.4f}\n"
                    holdout_text += f"u2 (Lateral):       Pred={skewness_pred_h['u_lateral']:7.4f} | True={skewness_true_h['u_lateral']:7.4f}\n"
                    holdout_text += f"u3 (Vertical):      Pred={skewness_pred_h['u_vertical']:7.4f} | True={skewness_true_h['u_vertical']:7.4f}\n"
                    holdout_text += f"{'-'*50}\n"
                    
                    print(holdout_text)
                    
                    # Save exclusively as blind test text log
                    blind_metrics_file = os.path.join(output_dir, f"blind_delta_metrics_{serie}.txt")
                    with open(blind_metrics_file, "w") as f:
                        f.write(holdout_text)
                    print(f"Blind Holdout Metrics saved to {blind_metrics_file}")
                
            del df_clean, Y_true, Y_pred
            cleanup_memory()
        else:
            warning_msg = "\n[Warning] Velocity columns not found for metrics calculation."
            print(warning_msg)
            metrics_file = os.path.join(output_dir, f"delta_metrics_{serie}.txt")
            with open(metrics_file, "w") as f:
                f.write(warning_msg)

    print("\nIncremental block prediction complete.")

    # --- GENERATE VALIDATION PLOTS ---
    if args.calc_metrics:
        try:
            generate_validation_plots(df, output_dir, serie, fs)
            generate_dissipation_series_plot(split_indices, df, output_dir, serie, fs)
        except Exception as e:
            print(f"[Proteção de Crash] Falha ao gerar gráficos de validação: {e}")

    # --- AUTOMATIC SPECTRAL ANALYSIS (MULTI-PLOT STYLE) ---
    print(f"\n{'='*50}")
    print("📊 STARTING SPECTRAL VALIDATION (PRED VS SONIC)")
    print(f"{'='*50}")

    # 1. Load Sonic Data safely (Ground Truth)
    sonic_file = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    sonic_df = None
    if os.path.exists(sonic_file):
        print(f"[Spectral] Loading sonic reference from {sonic_file}")
        # Optimize memory by loading only necessary columns and preserve the time index
        sonic_df = pd.read_csv(
            sonic_file,
            usecols=["time", "velocity_x", "velocity_y", "velocity_z"],
        )
        # If sonic file is massive, take a representative sample to save RAM
        if len(sonic_df) > 500000:
            sonic_df = sonic_df.iloc[:500000].reset_index(drop=True)
    else:
        print(f"[Warning] Sonic file not found. Spectra will show predictions only.")

    # 2. Setup Plotting Environment
    spectral_dir = os.path.join(output_dir, "plots_spectral")
    os.makedirs(spectral_dir, exist_ok=True)

    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs_hf = spectral_utils.estimate_sampling_frequency(df, "time")
    if fs_hf is None:
        fs_hf = cfg.get("FS_HOTFILM", 2000)

    fs_sonic = None
    if sonic_df is not None:
        fs_sonic = spectral_utils.estimate_sampling_frequency(sonic_df, "time")
    if fs_sonic is None:
        fs_sonic = cfg.get("FS_SONIC", 20.0)

    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    # 3. Generate Global Spectrum (Full Prediction vs Sonic)
    print("Generating global spectral comparison...")
    try:
        spectral_utils.plot_combined_spectrum(
            df,  
            pred_cols,
            fs_hf,
            f"Global Spectral Analysis (Pred vs Sonic) - Serie {serie}",
            os.path.join(spectral_dir, f"combined_spectrum_global_{serie}.png"),
            sonic_df=sonic_df,
            fs_sonic=fs_sonic,
        )
    except Exception as e:
        print(f"[Proteção de Crash] Falha ao gerar espectro global: {e}")

    # 4. Generate Block-wise Spectra (Individual Incremental Steps)
    print(f"Generating spectral plots for {args.num_blocks} data blocks...")
    for i, idx_list in enumerate(split_indices):
        try:
            # Pass only the necessary slice and columns to the plotting function to save RAM
            block_df_subset = df.iloc[idx_list][pred_cols]

            spectral_utils.plot_combined_spectrum(
                block_df_subset,
                pred_cols,
                fs_hf,
                f"Spectral Analysis Block {i+1} vs Sonic - Serie {serie}",
                os.path.join(spectral_dir, f"combined_spectrum_block_{i+1}_{serie}.png"),
                sonic_df=sonic_df,
                fs_sonic=fs_sonic,
            )
            print(f" -> Saved block {i+1}/{args.num_blocks}")
        except Exception as e:
            print(f"[Proteção de Crash] Falha ao gerar espectro do bloco {i+1}: {e}")
        finally:
            if 'block_df_subset' in locals(): del block_df_subset
            cleanup_memory()

    print(f"\n[Done] All spectral plots are available in: {spectral_dir}")


if __name__ == "__main__":
    main()
    