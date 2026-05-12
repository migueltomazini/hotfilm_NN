"""Incremental / online training for hot-film neural network.

This script is designed to process a time series dataset (real or synthetic) that
is naturally divided into chronological blocks, either due to data gaps or by
choice. It trains an MLP model on the first available block and then fine-tunes
it sequentially on each new block as it "arrives". Alternatively, with the
--scattered flag, it trains initially with scattered initial data from each
block and then fine-tunes with each full block. The --percentage flag controls
the percentage of data used from each block in sequential mode. At every step the script
calculates the same error metrics used elsewhere in the project (RMSE, spectral
slope, isotropy ratio) and records them in a CSV so the user can monitor the
model evolution.

The implementation reuses most of the machinery from ``train_mlp.py`` but adds
functions for segmentation and incremental updates.  Because the 0610 data is
empty in this repo, the segmentation rules here are generic: you can split by
fixed row count, by minimum duration between samples, or by an explicit gap
threshold (sensible for handling real datasets with irregular recording).

Usage examples:
    python3 incremental_train.py 0610 --num-blocks 10
    python3 incremental_train.py 0610 --num-blocks 10 --scattered
    python3 incremental_train.py 0610 --num-blocks 10 --percentage 50
    python3 incremental_train.py 0610 --num-blocks 10 --reverse-data
    python3 incremental_train.py 0610 --num-blocks 10 --holdout-last
    python3 incremental_train.py 0610 --num-blocks 10 --scattered --no-finetune
    python3 incremental_train.py 0610 --num-blocks 10 --scattered --subsequent-finetune-pct 20

If ``--num-blocks`` is provided the file will be chunked into N contiguous blocks
of approximately equal size.

The final model and per-block metrics are saved under ``data/train/results``.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import List, Optional

import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from utils import config, metrics, physics, data_loader, hyperparameter_optimization

# reuse network definition from train_mlp
import train_mlp
from train_mlp import MLP, VoltageVelocityDataset

# configuration constants
input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# DATA SEGMENTATION HELPERS
# ---------------------------------------------------------------------------


def split_dataframe_into_n_blocks(
    df: pd.DataFrame, n_blocks: int
) -> List[pd.DataFrame]:
    """Split dataframe into exactly N blocks of approximately equal size.

    Uses numpy.array_split to ensure exactly N blocks are created.
    Block sizes may differ by at most 1 sample.

    Args:
        df: input dataframe.
        n_blocks: desired number of blocks.

    Returns:
        list of exactly n_blocks dataframes.
    """
    if n_blocks <= 0:
        return [df]
    if n_blocks >= len(df):
        return [df.iloc[[i]].reset_index(drop=True) for i in range(len(df))]
    # use numpy to split indices evenly
    indices = np.arange(len(df))
    split_indices = np.array_split(indices, n_blocks)
    blocks = [df.iloc[idx_list].reset_index(drop=True) for idx_list in split_indices]
    return blocks


def prepare_blocks(
    df: pd.DataFrame,
    block_size: Optional[int] = None,
    gap_threshold: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Return a list of blocks according to the requested strategy."""
    blocks = [df]
    if gap_threshold is not None:
        # apply gap-based splitting first
        new_blocks = []
        for b in blocks:
            new_blocks.extend(split_dataframe_by_gap(b, gap_threshold=gap_threshold))
        blocks = new_blocks
    if block_size is not None:
        new_blocks = []
        for b in blocks:
            new_blocks.extend(split_dataframe_fixed_size(b, block_size))
        blocks = new_blocks
    # remove empty blocks that may arise from tiny gaps
    blocks = [b for b in blocks if len(b) > 0]
    return blocks


# ---------------------------------------------------------------------------
# TRAINING / EVALUATION UTILS
# ---------------------------------------------------------------------------


def train_on_block(
    model: torch.nn.Module,
    scaler: StandardScaler,
    block: pd.DataFrame,
    epochs: int,
    freeze: bool = False,
    lr: float = 1e-4,
    batch_size: int = 32,
) -> torch.nn.Module:
    """Fit or fine-tune ``model`` on a single block of data.

    If ``freeze`` is True then only the output layer is trained (useful when the
    model comes from a previous block and you want to adapt conservatively;
    otherwise all parameters are updated).
    """
    # extract the raw arrays
    X_raw = block[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = block[["velocity_x", "velocity_y", "velocity_z"]].values

    # scale or fit
    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] == X_raw.shape[1]:
        X_scaled = scaler.transform(X_raw)
    else:
        scaler.fit(X_raw)
        X_scaled = scaler.transform(X_raw)
    # training/validation split (simple halo method)
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    Y_train, Y_val = Y_raw[:split], Y_raw[split:]

    train_loader = DataLoader(
        VoltageVelocityDataset(X_train, Y_train, device), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(VoltageVelocityDataset(X_val, Y_val, device), batch_size=batch_size)

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
    model: torch.nn.Module, scaler: StandardScaler, block: pd.DataFrame, fs: float
) -> dict:
    """Compute error metrics for a single block and return a dict."""
    X_raw = block[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_raw = block[["velocity_x", "velocity_y", "velocity_z"]].values
    if hasattr(scaler, "mean_") and scaler.mean_.shape[0] == X_raw.shape[1]:
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = scaler.fit_transform(X_raw)  # should not happen normally
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled).float().to(device))
    preds_np = preds.cpu().numpy()
    rmse = metrics.calculate_rmse(preds_np, Y_raw)
    return {"rmse": rmse}


# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Incremental training with block-wise metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument(
        "--num-blocks",
        "--num_blocks",
        dest="num_blocks",
        type=int,
        default=None,
        help="number of blocks to split the dataset into (approximately equal size)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="path to an existing .pth file for warm start",
    )
    parser.add_argument(
        "--scattered",
        action="store_true",
        help="Use scattered training mode: train initially with initial data from each block, then fine-tune with full blocks.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of data to use from each block in sequential mode (default 100.0)",
    )
    parser.add_argument(
        "--reverse-data",
        action="store_true",
        help="Inverts the chronological order of the dataset before training.",
    )
    parser.add_argument(
        "--holdout-last",
        action="store_true",
        help="Reserves the final block strictly for testing (no training). Used to validate model predictive capability.",
    )
    # --- INJECTION: New flags for Scattered Tests ---
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="[Test 1] Disables fine-tuning in scattered mode. Evaluates the initial global model on all blocks.",
    )
    parser.add_argument(
        "--subsequent-finetune-pct",
        type=float,
        default=None,
        help="[Test 2] Percentage of block data to use for fine-tuning, starting immediately AFTER the initial scattered chunk.",
    )
    
    args = parser.parse_args()
    NUM_BLOCKS = args.num_blocks
    SCATTERED = args.scattered
    PERCENTAGE = args.percentage

    serie = args.serie
    # load training data already prepared by create_csv.py
    df_path = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(
            f"Training data not found: {df_path}\nRun `python3 create_csv.py train {serie}` first."
        )
    df = pd.read_csv(df_path)
    # ensure reynolds feature exists (read from config if absent)
    if "reynolds" not in df.columns:
        cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
        re_val = 0.0
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    cfg_temp = json.load(fh)
                re_val = cfg_temp.get("RE_NUMBER", 0.0)
            except Exception:
                pass
        df["reynolds"] = re_val
        print(f"[Info] added missing reynolds column = {re_val}")
    # read FS from config JSON to pass to physics
    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs = cfg["FS_HOTFILM"]

    if args.reverse_data:
        print("\n" + "!"*70)
        print("🧪 [EXPERIMENTAL MODE] FLAG --reverse-data ACTIVATED!")
        print("Inverting the entire chronological order of the DataFrame...")
        print("!"*70 + "\n")
        
        df = df.iloc[::-1].reset_index(drop=True)
        serie = f"{serie}_rev"
        print(f"[Info] Series identifier changed to '{serie}' to prevent overwriting original models.\n")

    # segmentation
    if NUM_BLOCKS is not None:
        print(f"Splitting into exactly {NUM_BLOCKS} blocks...")
        blocks = split_dataframe_into_n_blocks(df, NUM_BLOCKS)
    else:
        blocks = prepare_blocks(df, block_size=None, gap_threshold=None)

    if not SCATTERED:
        blocks = [
            b.iloc[: int(len(b) * PERCENTAGE / 100)].reset_index(drop=True)
            for b in blocks
        ]

    if len(blocks) == 0:
        print("No blocks extracted from the dataset. Exiting.")
        return

    print(f"Dataset split into {len(blocks)} blocks")

    # Hyperparameter optimization
    print("[Optimization] Optimizing hyperparameters...")
    if SCATTERED:
        # Use scattered initial data for optimization
        opt_blocks = [block.iloc[: len(block) // len(blocks)] for block in blocks]
        opt_df = pd.concat(opt_blocks).reset_index(drop=True)
    else:
        # Use first block for optimization
        opt_df = blocks[0].copy()

    # Prepare optimization data
    X_opt = opt_df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_opt = opt_df[["velocity_x", "velocity_y", "velocity_z"]].values

    # Simple scaler for optimization
    from sklearn.preprocessing import StandardScaler
    opt_scaler = StandardScaler()
    X_opt_scaled = opt_scaler.fit_transform(X_opt)

    split_opt = int(0.8 * len(X_opt_scaled))
    X_train_opt = X_opt_scaled[:split_opt]
    Y_train_opt = Y_opt[:split_opt]
    X_val_opt = X_opt_scaled[split_opt:]
    Y_val_opt = Y_opt[split_opt:]

    best_params = hyperparameter_optimization.optimize_hyperparameters(
        X_train_opt, Y_train_opt, X_val_opt, Y_val_opt, fs, serie, device, suffix="_incremental"
    )

    print(f"[Optimization] Best params: {best_params}")

    # model and scaler initialization
    if args.base_model is not None and os.path.exists(args.base_model):
        try:
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
        model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
        scaler = StandardScaler()

    results = []
    if SCATTERED:
        # Initial training with scattered data: small portions from each block.
        initial_blocks = [block.iloc[: len(block) // len(blocks)] for block in blocks]
        initial_df = pd.concat(initial_blocks).reset_index(drop=True)
        print(f"Initial training with {len(initial_df)} scattered samples")
        model = train_on_block(
            model, scaler, initial_df, epochs=best_params["epochs"],
            lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
        )

        # Preserve the initial model state so all block fine-tunings start from it.
        initial_state = copy.deepcopy(model.state_dict())

        out_folder = os.path.join(config.MODEL_DIR, "incremental")
        os.makedirs(out_folder, exist_ok=True)

        for i, block in enumerate(blocks):
            is_holdout = args.holdout_last and (i == len(blocks) - 1)
            
            print(f"\n===== Processing block {i+1}/{len(blocks)} ({len(block)} samples) =====")
            
            block_model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
            block_model.load_state_dict(initial_state)
            
            # --- INJECTION: Handling Test 1 and Test 2 Scenarios ---
            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            elif args.no_finetune:
                print("--> 🛑 [Test 1] NO FINE-TUNING MODE: Evaluating global scattered model only.")
            else:
                # Determine which data to use for fine-tuning
                if args.subsequent_finetune_pct is not None:
                    # Calculate index right after the initial chunk
                    start_idx = len(block) // len(blocks)
                    # Calculate number of rows representing X% of the entire block
                    num_rows = int(len(block) * (args.subsequent_finetune_pct / 100.0))
                    end_idx = min(start_idx + num_rows, len(block))
                    
                    finetune_block = block.iloc[start_idx:end_idx].reset_index(drop=True)
                    print(f"--> 📉 [Test 2] Fine-tuning on subsequent {args.subsequent_finetune_pct}% of data ({len(finetune_block)} samples).")
                else:
                    finetune_block = block
                    print("--> Fine-tuning on the entire block (Standard Scattered).")
                
                # Check if finetune_block has data to avoid empty tensor errors
                if len(finetune_block) > 0:
                    block_model = train_on_block(
                        block_model, scaler, finetune_block, epochs=best_params["epochs_finetune"],
                        freeze=True, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )
                else:
                    print("--> ⚠️ Warning: Fine-tune block is empty. Skipping fine-tuning for this block.")

            metrics_dict = evaluate_block(block_model, scaler, block, fs)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(block)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(
                block_model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth")
            )
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\n")
                    f.write(f"Mode: SCATTERED\n")
                    f.write(f"RMSE on unseen Block {i+1}: {metrics_dict['rmse']:.6f}\n")
                print(f"--> Blind holdout text log saved to {log_path}")

    else:
        # iterate through blocks: first block trains from scratch, others fine-tune
        for i, block in enumerate(blocks):
            is_holdout = args.holdout_last and (i == len(blocks) - 1)
            
            print(f"\n===== Processing block {i+1}/{len(blocks)} ({len(block)} samples) =====")
            
            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            else:
                if i == 0:
                    model = train_on_block(
                        model, scaler, block, epochs=best_params["epochs"],
                        lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )
                else:
                    # freeze early layers after first block
                    model = train_on_block(
                        model, scaler, block, epochs=best_params["epochs_finetune"],
                        freeze=True, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )
            
            metrics_dict = evaluate_block(model, scaler, block, fs)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(block)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)
            
            # optional: save intermediate model
            bloc_name = f"{serie}_block{i+1}"
            out_folder = os.path.join(config.MODEL_DIR, "incremental")
            os.makedirs(out_folder, exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth")
            )
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))

            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\n")
                    f.write(f"Mode: SEQUENTIAL\n")
                    f.write(f"RMSE on unseen Block {i+1}: {metrics_dict['rmse']:.6f}\n")
                print(f"--> Blind holdout text log saved to {log_path}")

    # save results table
    results_df = pd.DataFrame(results)
    res_path = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie}", "block_metrics.csv"
    )
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    results_df.to_csv(res_path, index=False)
    print(f"Block-wise metrics stored in {res_path}")

    # generate evolution plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Differentiate holdout in plot if it exists
    if args.holdout_last:
        ax.plot(results_df["block"][:-1], results_df["rmse"][:-1], marker="o", label="Trained Blocks")
        ax.plot(results_df["block"].iloc[-1], results_df["rmse"].iloc[-1], marker="*", color="red", markersize=10, label="Blind Holdout")
        ax.legend()
    else:
        ax.plot(results_df["block"], results_df["rmse"], marker="o")
        
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plot_path = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie}", "block_evolution.png"
    )
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Evolution plots saved to {plot_path}")

    # save final model
    final_model_path = os.path.join(
        config.MODEL_DIR, "incremental", f"model_{serie}_final.pth"
    )
    final_scaler_path = os.path.join(
        config.MODEL_DIR, "incremental", f"scaler_{serie}_final.joblib"
    )
    torch.save(model.state_dict(), final_model_path)
    joblib.dump(scaler, final_scaler_path)
    print(f"\nFinal model saved to {final_model_path}")
    print(f"Final scaler saved to {final_scaler_path}")

    print("\nIncremental training complete.")


if __name__ == "__main__":
    main()