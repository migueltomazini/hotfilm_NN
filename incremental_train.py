"""Incremental / online training for hot-film neural network.

This script is designed to process a time series dataset (real or synthetic) that
is naturally divided into chronological blocks, either due to data gaps or by
choice. It trains an MLP model on the first available block and then fine-tunes
it sequentially on each new block as it "arrives". At every step the script
calculates the same error metrics used elsewhere in the project (RMSE, spectral
slope, isotropy ratio) and records them in a CSV so the user can monitor the
model evolution.

The implementation reuses most of the machinery from ``train_mlp.py`` but adds
functions for segmentation and incremental updates.  Because the 0610 data is
empty in this repo, the segmentation rules here are generic: you can split by
fixed row count, by minimum duration between samples, or by an explicit gap
threshold (sensible for handling real datasets with irregular recording).

Usage examples:
    python3 incremental_train.py <serie> [--num-blocks N]
    python3 incremental_train.py 0610 --num-blocks 10
    python3 incremental_predict.py --num-blocks 10 0610 --calc-metrics --input data/train/train_df_0610.csv

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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from utils import config, metrics, physics, data_loader

# reuse network definition from train_mlp
import train_mlp
from train_mlp import MLP, VoltageVelocityDataset

# configuration constants
input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE
EPOCHS = 50  # reduced for incremental training
EPOCHS_FINETUNE = 25  # reduced for fine-tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        VoltageVelocityDataset(X_train, Y_train, device), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(VoltageVelocityDataset(X_val, Y_val, device), batch_size=32)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.output_layer.parameters():
            param.requires_grad = True

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
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
    slope = physics.calculate_spectral_slope(preds_np, fs)
    iso = physics.calculate_isotropy_ratio(preds_np, fs)
    return {"rmse": rmse, "slope": slope, "isotropy": iso}


# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Incremental training with block-wise metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    # allow both hyphen and underscore variants for familiarity
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
    args = parser.parse_args()
    NUM_BLOCKS = args.num_blocks

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

    # segmentation
    if NUM_BLOCKS is not None:
        print(f"Splitting into exactly {NUM_BLOCKS} blocks...")
        blocks = split_dataframe_into_n_blocks(df, NUM_BLOCKS)
    else:
        blocks = prepare_blocks(df, block_size=None, gap_threshold=None)

    blocks = [b.iloc[: int(len(b) * 0.2)].reset_index(drop=True) for b in blocks]

    if len(blocks) == 0:
        print("No blocks extracted from the dataset. Exiting.")
        return

    print(f"Dataset split into {len(blocks)} blocks")

    # model and scaler initialization
    if args.base_model is not None and os.path.exists(args.base_model):
        # attempt to read the architecture from the metadata of the specified base model
        try:
            h_layers, h_size = train_mlp.get_base_model_params(
                os.path.basename(args.base_model)
            )
        except Exception:
            h_layers, h_size = 2, 64  # fallback defaults in case metadata is missing
        model = MLP(input_size, output_size, h_size, h_layers).to(device)
        model.load_state_dict(torch.load(args.base_model, map_location=device))
        scaler = joblib.load(args.base_model.replace(".pth", ".joblib"))
        print(f"Loaded base model and scaler from {args.base_model}")
    else:
        # start from scratch simple architecture
        model = MLP(input_size, output_size, 64, 2).to(device)
        scaler = StandardScaler()

    results = []
    # iterate through blocks: first block trains from scratch, others fine-tune
    for i, block in enumerate(blocks):
        print(
            f"\n===== Processing block {i+1}/{len(blocks)} ({len(block)} samples) ====="
        )
        if i == 0:
            model = train_on_block(model, scaler, block, epochs=EPOCHS)
        else:
            # freeze early layers after first block
            model = train_on_block(
                model, scaler, block, epochs=EPOCHS_FINETUNE, freeze=True
            )

        metrics_dict = evaluate_block(model, scaler, block, fs)
        metrics_dict["block"] = i
        metrics_dict["samples"] = len(block)
        results.append(metrics_dict)

        # optional: save intermediate model
        bloc_name = f"{serie}_block{i+1}"
        out_folder = os.path.join(config.MODEL_DIR, "incremental")
        os.makedirs(out_folder, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth")
        )
        joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))

    # save results table
    results_df = pd.DataFrame(results)
    res_path = os.path.join(
        config.DATA_DIR, "train", "results", f"results_{serie}", "block_metrics.csv"
    )
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    results_df.to_csv(res_path, index=False)
    print(f"Block-wise metrics stored in {res_path}")

    # generate evolution plots
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].plot(results_df["block"], results_df["rmse"], marker="o")
    ax[0].set_ylabel("RMSE")
    ax[1].plot(results_df["block"], results_df["slope"], marker="o")
    ax[1].set_ylabel("Spectral slope")
    ax[2].plot(results_df["block"], results_df["isotropy"], marker="o")
    ax[2].set_ylabel("Isotropy")
    ax[2].set_xlabel("Block index")
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
