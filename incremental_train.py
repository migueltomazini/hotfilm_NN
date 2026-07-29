"""Incremental / online training for hot-film neural network.

This script is designed to process a time series dataset (real or synthetic) that
is naturally divided into chronological blocks. It trains an MLP model on the first 
available block and then fine-tunes it sequentially on each new block as it "arrives". 

In `--scattered` mode, it employs a strict 10/10/80 data split strategy to prevent 
data leakage:
    - 10% of each block is aggregated for an initial global training.
    - 10% of each specific block is used for block-specific fine-tuning.
    - The remaining 80% of each block is strictly reserved for blind evaluation.

CRITICAL OPTIMIZATIONS APPLIED:
    - Low-Pass Filtering (Kit et al. 2016): Voltages and velocities are block-averaged 
      during training to match the sonic anemometer's poor high-frequency response.
    - Spectral Magnification Correction (Kit et al. 2016): Post-processes predictions 
      by anchoring low-frequency spectral energy to the reliable sonic anemometer data.
    - Memory Stream Optimization: Keeps the massive TensorDataset on CPU memory and shuffles 
      mini-batches to the hardware device dynamically.

Usage examples:
    python3 incremental_train.py 0610 --num-blocks 10
    python3 incremental_train.py 0610 --num-blocks 10 --scattered --holdout-last
"""

import os
import copy
import json
import argparse
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

from utils import config, data_loader, hyperparameter_optimization, metrics
from train_mlp import MLP

input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE
device = torch.device("cpu")

# =============================================================================
# DATA PREPROCESSING (Kit et al. 2016)
# =============================================================================
def apply_block_averaging(df: pd.DataFrame, window: int = 600) -> pd.DataFrame:
    """
    Applies block averaging (rolling mean) to voltages and velocities to match 
    the sonic anemometer's low-frequency response, as strictly recommended by 
    Kit et al. (2016).
    """
    df_smoothed = df.copy()
    cols_to_smooth = [
        "voltage_x", "voltage_y", "voltage_z", 
        "velocity_x", "velocity_y", "velocity_z"
    ]
    # Centered rolling mean prevents phase shifting between signals
    for col in cols_to_smooth:
        if col in df_smoothed.columns:
            df_smoothed[col] = df_smoothed[col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
    return df_smoothed


# =============================================================================
# SPECTRAL MAGNIFICATION CORRECTION (Kit et al. 2016)
# =============================================================================
def apply_spectral_magnification(y_pred, y_true, fs, f_cutoff=1.66):
    """
    Applies Spectral Magnification Correction as described in Kit et al. (2016).
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

def evaluate_block_magnified(model, scaler, df, fs, device):
    """Evaluates the block and applies spectral magnification before calculating RMSE."""
    X_raw = df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = df[["velocity_x", "velocity_y", "velocity_z"]].values
    X_scaled = scaler.transform(X_raw)
    
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        
    preds_mag, k_factors = apply_spectral_magnification(preds, Y_raw, fs)
    rmse = metrics.calculate_rmse(preds_mag, Y_raw)
    return {"rmse": rmse, "k_x": k_factors[0], "k_y": k_factors[1], "k_z": k_factors[2]}


# =============================================================================
# STANDARD TRAINING LOOP (Memory Optimized)
# =============================================================================
def train_on_block(model, scaler, df, epochs, device, lr, batch_size, freeze=False):
    """
    Trains the block using standard MSE.
    """
    if freeze:
        params_to_train = []
        layers = list(model.children())
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                for param in layer.parameters():
                    param.requires_grad = True
                    params_to_train.append(param)
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        optimizer = optim.Adam(params_to_train, lr=lr)
    else:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=lr)

    X_raw = df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = df[["velocity_x", "velocity_y", "velocity_z"]].values
    
    # EXTREMELY IMPORTANT: The scaler MUST be pre-fitted on RAW high-frequency data
    if not hasattr(scaler, "mean_"):
        raise RuntimeError("Scaler must be fitted on raw data before calling train_on_block!")
        
    X_scaled = scaler.transform(X_raw)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)

    criterion = nn.MSELoss()

    model.train()
    print(f"--> Starting training for {epochs} epochs (Batch Size: {batch_size})")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in dataloader:
            bx = bx.to(device)
            by = by.to(device)
            
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        log_interval = max(1, epochs / 10)
        if (epoch + 1) % int(log_interval) == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch [{epoch+1}/{epochs}] | Avg MSE Loss: {avg_loss:.6f}")
            
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Incremental training with block-wise metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument(
        "--num-blocks",
        dest="num_blocks",
        type=int,
        default=None,
        help="number of blocks to split the dataset into",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="path to an existing .pth file for warm start",
    )
    parser.add_argument(
        "--scattered", action="store_true", help="Use scattered training mode (10/10/80 split)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=20.0,
        help="Percentage of data to train on each block in sequential mode (default: 20)",
    )
    parser.add_argument(
        "--reverse-data", action="store_true", help="Inverts chronological order"
    )
    parser.add_argument(
        "--holdout-last", action="store_true", help="Reserves the final block strictly for testing."
    )

    args = parser.parse_args()
    NUM_BLOCKS = args.num_blocks
    SCATTERED = args.scattered
    PERCENTAGE = args.percentage
    serie = args.serie

    df_path = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Training data not found: {df_path}")
    df_raw = pd.read_csv(df_path)

    if "reynolds" not in df_raw.columns:
        cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
        re_val = 0.0
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    cfg_temp = json.load(fh)
                re_val = cfg_temp.get("RE_NUMBER", 0.0)
            except Exception:
                pass
        df_raw["reynolds"] = re_val
        print(f"[Info] added missing reynolds column = {re_val}")

    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs = cfg["FS_HOTFILM"]

    if args.reverse_data:
        print("\n" + "!" * 70)
        print("🧪 [EXPERIMENTAL MODE] FLAG --reverse-data ACTIVATED!")
        df_raw = df_raw.iloc[::-1].reset_index(drop=True)
        serie = f"{serie}_rev"
        print("!" * 70 + "\n")

    # --- APPLY LOW-PASS FILTERING TO TRAINING DATA (Kit et al. 2016) ---
    print("\n[Preprocessing] Applying 600-point block averaging to training sets...")
    df_smoothed = apply_block_averaging(df_raw, window=600)

    if NUM_BLOCKS is not None:
        print(f"Splitting into exactly {NUM_BLOCKS} blocks...")
        blocks_raw = data_loader.split_dataframe_into_n_blocks(df_raw, NUM_BLOCKS)
        blocks_smooth = data_loader.split_dataframe_into_n_blocks(df_smoothed, NUM_BLOCKS)
    else:
        blocks_raw = data_loader.prepare_blocks(df_raw, block_size=None, gap_threshold=None)
        blocks_smooth = data_loader.prepare_blocks(df_smoothed, block_size=None, gap_threshold=None)

    if len(blocks_raw) == 0:
        print("No blocks extracted from the dataset. Exiting.")
        return

    print(f"Dataset split into {len(blocks_raw)} blocks")

    # ---------------------------------------------------------
    # STRICT DATA SPLITTING
    # ---------------------------------------------------------
    if SCATTERED:
        initial_blocks = []
        finetune_blocks = []
        eval_blocks = []
        
        for b_raw, b_smooth in zip(blocks_raw, blocks_smooth):
            n = len(b_raw)
            idx_10 = int(n * 0.10)
            idx_20 = int(n * 0.20)
            
            # 0% - 10%: Initial global training pool (Smoothed Data)
            initial_blocks.append(b_smooth.iloc[:idx_10].reset_index(drop=True))
            # 10% - 20%: Block-specific fine-tuning (Smoothed Data)
            finetune_blocks.append(b_smooth.iloc[idx_10:idx_20].reset_index(drop=True))
            # 20% - 100%: Blind evaluation / testing (Raw High-Frequency Data)
            eval_blocks.append(b_raw.iloc[idx_20:].reset_index(drop=True))
            
        opt_df = pd.concat(initial_blocks).reset_index(drop=True)
    else:
        # Sequential mode
        opt_df = blocks_smooth[0].iloc[: int(len(blocks_smooth[0]) * PERCENTAGE / 100)].reset_index(drop=True)

    print("[Optimization] Optimizing hyperparameters...")
    # NOTE: Even for HPO, we should idealistically use raw scaling, but we will leave opt_scaler 
    # to fit on opt_df for now to not break external module dependencies. The final training scaler will be rigid.
    X_opt = opt_df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_opt = opt_df[["velocity_x", "velocity_y", "velocity_z"]].values

    opt_scaler = StandardScaler()
    X_opt_scaled = opt_scaler.fit_transform(X_opt)

    split_opt = int(0.8 * len(X_opt_scaled))
    X_train_opt = X_opt_scaled[:split_opt]
    Y_train_opt = Y_opt[:split_opt]
    X_val_opt = X_opt_scaled[split_opt:]
    Y_val_opt = Y_opt[split_opt:]

    best_params = hyperparameter_optimization.optimize_hyperparameters(
        X_train_opt,
        Y_train_opt,
        X_val_opt,
        Y_val_opt,
        fs,
        serie,
        device,
        suffix="_incremental",
    )

    if args.base_model is not None and os.path.exists(args.base_model):
        try:
            import train_mlp
            h_layers, h_size = train_mlp.get_base_model_params(
                os.path.basename(args.base_model)
            )
        except Exception:
            h_layers, h_size = best_params["hidden_layers"], best_params["hidden_size"]
        model = MLP(input_size, output_size, h_size, h_layers).to(device)
        model.load_state_dict(torch.load(args.base_model, map_location=device))
        scaler = joblib.load(args.base_model.replace(".pth", ".joblib"))
        print(f"Loaded base model and scaler from {args.base_model}")
    else:
        model = MLP(
            input_size,
            output_size,
            best_params["hidden_size"],
            best_params["hidden_layers"],
        ).to(device)
        
        # ---------------------------------------------------------
        # NEW: GLOBAL SCALER FITTED EXCLUSIVELY ON RAW DATA
        # ---------------------------------------------------------
        scaler = StandardScaler()
        print("[Info] Fitting StandardScaler globally on RAW high-frequency training data to prevent out-of-distribution explosion...")
        if SCATTERED:
            raw_train_list = []
            for b_raw in blocks_raw:
                idx_10 = int(len(b_raw) * 0.10)
                raw_train_list.append(b_raw.iloc[:idx_10])
            raw_train_df = pd.concat(raw_train_list).reset_index(drop=True)
            scaler.fit(raw_train_df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values)
        else:
            idx_train = int(len(blocks_raw[0]) * (PERCENTAGE / 100.0))
            raw_train_df = blocks_raw[0].iloc[:idx_train].reset_index(drop=True)
            scaler.fit(raw_train_df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values)
        # ---------------------------------------------------------

    results = []
    out_folder = os.path.join(config.MODEL_DIR, "incremental")
    os.makedirs(out_folder, exist_ok=True)
    
    if SCATTERED:
        initial_df = pd.concat(initial_blocks).reset_index(drop=True)
        print(f"Initial training with {len(initial_df)} scattered smoothed samples (10% of each block)")
        
        model = train_on_block(
            model,
            scaler,
            initial_df,
            epochs=best_params["epochs"],
            device=device,
            lr=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            freeze=False
        )

        initial_state = copy.deepcopy(model.state_dict())

        for i in range(len(blocks_raw)):
            is_holdout = args.holdout_last and (i == len(blocks_raw) - 1)
            print(f"\n===== Processing block {i+1}/{len(blocks_raw)} =====")
            
            block_model = MLP(
                input_size,
                output_size,
                best_params["hidden_size"],
                best_params["hidden_layers"],
            ).to(device)
            block_model.load_state_dict(initial_state)

            finetune_df = finetune_blocks[i]
            eval_df = eval_blocks[i]

            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            else:
                print(f"--> Fine-tuning on subsequent 10% of smoothed data ({len(finetune_df)} samples).")
                if len(finetune_df) > 0:
                    block_model = train_on_block(
                        block_model,
                        scaler,
                        finetune_df,
                        epochs=best_params["epochs_finetune"],
                        device=device,
                        lr=best_params["learning_rate"],
                        batch_size=best_params["batch_size"],
                        freeze=True
                    )

            print(f"--> Evaluating strictly on unseen 80% of RAW HF data ({len(eval_df)} samples) with Spectral Correction.")
            metrics_dict = evaluate_block_magnified(block_model, scaler, eval_df, fs, device)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(eval_df)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(
                block_model.state_dict(),
                os.path.join(out_folder, f"model_{bloc_name}.pth"),
            )
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\n")
                    f.write(f"Mode: SCATTERED\n")
                    f.write(f"RMSE (Magnified) on unseen HF Block {i+1}: {metrics_dict['rmse']:.6f}\n")
                print(f"--> Blind holdout text log saved to {log_path}")

    else:
        # Sequential Mode
        for i, (block_raw, block_smooth) in enumerate(zip(blocks_raw, blocks_smooth)):
            is_holdout = args.holdout_last and (i == len(blocks_raw) - 1)
            print(f"\n===== Processing block {i+1}/{len(blocks_raw)} ({len(block_raw)} samples) =====")

            idx_train = int(len(block_raw) * (PERCENTAGE / 100.0))
            train_df = block_smooth.iloc[:idx_train].reset_index(drop=True)
            eval_df = block_raw.iloc[idx_train:].reset_index(drop=True)

            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            else:
                print(f"--> Training on initial {PERCENTAGE}% smoothed data ({len(train_df)} samples).")
                if i == 0:
                    model = train_on_block(
                        model,
                        scaler,
                        train_df,
                        epochs=best_params["epochs"],
                        device=device,
                        lr=best_params["learning_rate"],
                        batch_size=best_params["batch_size"],
                        freeze=False
                    )
                else:
                    model = train_on_block(
                        model,
                        scaler,
                        train_df,
                        epochs=best_params["epochs_finetune"],
                        device=device,
                        lr=best_params["learning_rate"],
                        batch_size=best_params["batch_size"],
                        freeze=True
                    )

            print(f"--> Evaluating strictly on unseen {100 - PERCENTAGE}% of RAW HF data ({len(eval_df)} samples) with Spectral Correction.")
            metrics_dict = evaluate_block_magnified(model, scaler, eval_df, fs, device)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(eval_df)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(
                model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth")
            )
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\n")
                    f.write(f"Mode: SEQUENTIAL\n")
                    f.write(f"RMSE (Magnified) on unseen HF Block {i+1}: {metrics_dict['rmse']:.6f}\n")
                print(f"--> Blind holdout text log saved to {log_path}")

    results_df = pd.DataFrame(results)
    res_path = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie}", "block_metrics.csv"
    )
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    results_df.to_csv(res_path, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    if args.holdout_last:
        ax.plot(results_df["block"][:-1], results_df["rmse"][:-1], marker="o", label="Trained Blocks")
        ax.plot(results_df["block"].iloc[-1], results_df["rmse"].iloc[-1], marker="*", color="red", markersize=10, label="Blind Holdout")
        ax.legend()
    else:
        ax.plot(results_df["block"], results_df["rmse"], marker="o", label="Test RMSE")
        
    ax.set_ylabel("RMSE (Magnified)")
    ax.set_xlabel("Block Index")
    ax.set_title("Incremental Training RMSE per Block (Unseen Data)")
    ax.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie}", "block_evolution.png"
    )
    fig.savefig(plot_path)
    plt.close(fig)

    final_model_path = os.path.join(
        config.MODEL_DIR, "incremental", f"model_{serie}_final.pth"
    )
    final_scaler_path = os.path.join(
        config.MODEL_DIR, "incremental", f"scaler_{serie}_final.joblib"
    )
    final_state = block_model.state_dict() if SCATTERED else model.state_dict()
    torch.save(final_state, final_model_path)
    joblib.dump(scaler, final_scaler_path)

if __name__ == "__main__":
    main()