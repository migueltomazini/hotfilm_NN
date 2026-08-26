"""Wind velocity prediction using trained MLP models.

This module loads a pre-trained Multi-Layer Perceptron (MLP) model to predict
wind velocity components from hot-film voltage measurements. It handles model
architecture reconstruction from metadata, input scaling using pre-trained scalers,
and output validation against synthetic ground truth when available.

Usage:
    python3 run_mlp.py <series_id> <model_filename> [--block-size SIZE] [--gap GAP]

Example:
    python3 run_mlp.py 21180 model_mlp_21180.pth
"""

import argparse
import os
import sys
import time
import logging
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Import utility modules
from utils import config, metrics, physics, spectral_utils

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global Constants
INPUT_DF_NAME = "voltage"
OUTPUT_DF_NAME = "velocity"


def get_model_metadata(model_id: str, dir_base: str) -> tuple:
    """Retrieve network architecture parameters from training metadata.

    Loads the hyperparameters CSV file generated during model training and extracts
    the network's layer count and hidden layer size. Falls back to defaults if
    expected columns are not found.

    Args:
        model_id: The model identifier matching the training session.
        dir_base: Base directory containing the data folders.

    Returns:
        tuple: (num_hidden_layers, hidden_size) for network reconstruction.

    Raises:
        FileNotFoundError: If the metadata file is not found.
    """
    path = os.path.join(dir_base, "data", "train", "results", f"results_{model_id}", f"hyperparameters_{model_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found at {path}")
    
    df_meta = pd.read_csv(path)
    
    # Handle column name variations from different training runs
    layers = (
        int(df_meta["Layers"].iloc[0])
        if "Layers" in df_meta.columns
        else int(df_meta.get("hidden_layers", [1])[0])
    )
    size = (
        int(df_meta["Size"].iloc[0])
        if "Size" in df_meta.columns
        else int(df_meta.get("hidden_size", [114])[0])
    )
    return layers, size


class MLP(nn.Module):
    """Multi-Layer Perceptron for voltage-to-velocity prediction.

    Converts hot-film voltage measurements to velocity components using
    configurable fully-connected layers with ReLU activations.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D input tensors
        if x.dim() == 3:
            x = x.squeeze(1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class VoltageVelocityDataset(Dataset):
    """PyTorch Dataset for voltage-to-velocity prediction pairs.

    Loads voltage features and corresponding velocity targets onto the
    specified device for model training or inference.
    """

    def __init__(self, data: pd.DataFrame, device: torch.device):
        """Initialize dataset from DataFrame.

        Args:
            data: DataFrame with columns for voltage components and velocities.
            device: PyTorch device (CPU or CUDA) where tensors will be stored.
        """
        self.X = (
            torch.tensor(
                data[
                    [
                        f"{INPUT_DF_NAME}_x",
                        f"{INPUT_DF_NAME}_y",
                        f"{INPUT_DF_NAME}_z",
                    ]
                ].values
            )
            .float()
            .to(device)
        )
        self.Y = (
            torch.tensor(
                data[
                    [
                        f"{OUTPUT_DF_NAME}_x",
                        f"{OUTPUT_DF_NAME}_y",
                        f"{OUTPUT_DF_NAME}_z",
                    ]
                ].values
            )
            .float()
            .to(device)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def validate_synthetic_results(serie: str, df_predicted: pd.DataFrame, dest_path: str, block_size: int = None, gap: float = None):
    """Compare predictions against synthetic ground truth if available.

    If ``block_size`` or ``gap`` are provided, the method also computes per-block
    RMSE statistics and spectral metrics.

    Args:
        serie: Series identifier.
        df_predicted: DataFrame containing model predictions.
        dest_path: Output directory for validation reports.
        block_size: Fixed block length for metric segmentation.
        gap: Gap threshold (s) for metrics segmentation.
    """
    from utils.data_loader import prepare_blocks

    # Calculate sampling frequency dynamically
    fs = spectral_utils.estimate_sampling_frequency(df_predicted, "time")
    if fs is None:
        fs = config.FS_HOTFILM_DEFAULT

    ref_path = f"./data/raw/{serie}/hotfilm_vel_{serie}.csv"
    df_block_summary = []

    if os.path.exists(ref_path):
        logging.info(f"Synthetic ground truth found: {ref_path}. Validating...")
        try:
            skip_rows = 0
            with open(ref_path, "r") as f:
                first_line = f.readline().strip()
                try:
                    float(first_line.split(",")[0])
                except (ValueError, IndexError):
                    skip_rows = 1
                    
            n_predictions = len(df_predicted)
            df_ref = pd.read_csv(
                ref_path,
                sep=",",
                names=["time", "velocity_x", "velocity_y", "velocity_z"],
                skiprows=skip_rows,
                nrows=n_predictions,
                low_memory=False,
            )
            for col in ["time", "velocity_x", "velocity_y", "velocity_z"]:
                df_ref[col] = pd.to_numeric(df_ref[col], errors="coerce")
            df_ref = df_ref.dropna()
        except Exception as e:
            logging.warning(f"Could not parse reference file: {e}")
            return
            
        if len(df_ref) == 0:
            logging.warning("No valid numeric data found in reference file.")
            return

        pred_x = df_predicted[f"{OUTPUT_DF_NAME}_predicted_x"].values
        pred_y = df_predicted[f"{OUTPUT_DF_NAME}_predicted_y"].values
        pred_z = df_predicted[f"{OUTPUT_DF_NAME}_predicted_z"].values
        ref_x = df_ref["velocity_x"].values
        ref_y = df_ref["velocity_y"].values
        ref_z = df_ref["velocity_z"].values

        min_len = min(len(pred_x), len(ref_x))
        if len(pred_x) != len(ref_x):
            logging.info(f"Aligning data lengths ({len(pred_x)} vs {len(ref_x)}). Truncating to {min_len} rows.")
            pred_x, pred_y, pred_z = pred_x[:min_len], pred_y[:min_len], pred_z[:min_len]
            ref_x, ref_y, ref_z = ref_x[:min_len], ref_y[:min_len], ref_z[:min_len]

        mask_x = ~np.isnan(pred_x) & ~np.isnan(ref_x)
        mask_y = ~np.isnan(pred_y) & ~np.isnan(ref_y)
        mask_z = ~np.isnan(pred_z) & ~np.isnan(ref_z)

        rmse_x = metrics.calculate_rmse(pred_x[mask_x], ref_x[mask_x])
        rmse_y = metrics.calculate_rmse(pred_y[mask_y], ref_y[mask_y])
        rmse_z = metrics.calculate_rmse(pred_z[mask_z], ref_z[mask_z])

        logging.info("--- ERROR ANALYSIS (Predicted vs Synthetic Ground Truth) ---")
        logging.info(f"RMSE Velocity X: {rmse_x:.12f}")
        logging.info(f"RMSE Velocity Y: {rmse_y:.12f}")
        logging.info(f"RMSE Velocity Z: {rmse_z:.12f}")
        logging.info("----------------------------------------------------------")

        # Compute block-wise RMSE if segmentation requested
        if block_size is not None or gap is not None:
            merged = df_predicted.copy()
            merged["ref_x"] = np.concatenate([ref_x, np.zeros(len(df_predicted) - len(ref_x))])[: len(df_predicted)]
            merged["ref_y"] = np.concatenate([ref_y, np.zeros(len(df_predicted) - len(ref_y))])[: len(df_predicted)]
            merged["ref_z"] = np.concatenate([ref_z, np.zeros(len(df_predicted) - len(ref_z))])[: len(df_predicted)]
            
            blocks = prepare_blocks(merged, block_size=block_size, gap_threshold=gap)
            for bi, b in enumerate(blocks):
                rx, ry, rz = b["ref_x"].values, b["ref_y"].values, b["ref_z"].values
                px = b[f"{OUTPUT_DF_NAME}_predicted_x"].values
                py = b[f"{OUTPUT_DF_NAME}_predicted_y"].values
                pz = b[f"{OUTPUT_DF_NAME}_predicted_z"].values
                
                bx_rmse = metrics.calculate_rmse(px, rx)
                by_rmse = metrics.calculate_rmse(py, ry)
                bz_rmse = metrics.calculate_rmse(pz, rz)
                
                logging.info(f"Block {bi}: RMSE_x={bx_rmse:.6f}, RMSE_y={by_rmse:.6f}, RMSE_z={bz_rmse:.6f}")
                df_block_summary.append({
                    "block": bi,
                    "rmse_x": bx_rmse,
                    "rmse_y": by_rmse,
                    "rmse_z": bz_rmse,
                })
                
            if df_block_summary:
                outpath = os.path.join(dest_path, "block_rmse.csv")
                pd.DataFrame(df_block_summary).to_csv(outpath, index=False)
                logging.info(f"Block RMSE table saved to {outpath}")
    else:
        # No reference file; compute physics metrics per block if segmented
        if block_size is not None or gap is not None:
            merged = df_predicted.copy()
            blocks = prepare_blocks(merged, block_size=block_size, gap_threshold=gap)
            for bi, b in enumerate(blocks):
                arr = b[
                    [
                        f"{OUTPUT_DF_NAME}_predicted_x",
                        f"{OUTPUT_DF_NAME}_predicted_y",
                        f"{OUTPUT_DF_NAME}_predicted_z",
                    ]
                ].values
                slope = physics.calculate_spectral_slope(arr, fs=fs)
                iso = physics.calculate_isotropy_ratio(arr, fs=fs)
                logging.info(f"Block {bi}: slope={slope:.4f}, isotropy={iso:.4f}")


def run_model(serie: str, model_filename: str, block_size: int = None, gap_threshold: float = None):
    """Execute wind velocity prediction pipeline.

    Workflow:
        1. Reconstruct model architecture from training metadata.
        2. Load input data and corresponding scaler.
        3. Scale input features using pre-trained statistics.
        4. Generate velocity predictions in safe memory batches.
        5. Export results to CSV.
        6. Validate against synthetic ground truth if available.
        
    Args:
        serie: Series identifier.
        model_filename: Filename of the .pth model.
        block_size: Optional block size for segmented metrics.
        gap_threshold: Optional gap threshold for segmented metrics.
    """
    model_dir = config.MODEL_DIR
    dir_base = "."
    
    local_model = os.path.join(model_dir, model_filename)
    local_data = os.path.join(dir_base, "data", "run", f"run_{serie}.csv")
    local_dest = os.path.join(dir_base, "data", "run", "results", f"velocity_{serie}")

    logging.info(f"Starting Prediction Pipeline for Series: {serie}")
    logging.info(f"Model: {local_model}")
    logging.info(f"Data:  {local_data}")
    logging.info(f"Dest:  {local_dest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Processing device configured to: {device}")

    # Reconstruct network architecture based on metadata
    model_id = model_filename.replace("model_mlp_", "").replace(".pth", "")
    h_layers, h_size = get_model_metadata(model_id, dir_base)
    
    model = MLP(
        input_dim=config.INPUT_SIZE, 
        output_dim=config.OUTPUT_SIZE, 
        hidden_dim=h_size, 
        num_hidden_layers=h_layers
    ).to(device)
    
    model.load_state_dict(torch.load(local_model, map_location=device))
    model.eval()

    # Load input data and apply feature scaling
    data_in = pd.read_csv(local_data)
    data_in = data_in.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    scaler_path = os.path.join(model_dir, f"scaler_{model_id}.joblib")
    scaler = joblib.load(scaler_path)

    # Normalize input features using training statistics
    X_raw = data_in[[f"{INPUT_DF_NAME}_x", f"{INPUT_DF_NAME}_y", f"{INPUT_DF_NAME}_z"]].values
    X_scaled = scaler.transform(X_raw)

    # Perform predictions in batches to avoid GPU memory issues
    batch_size = 1024 
    predictions_list = []
    
    logging.info("Executing batched inference...")
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i : i + batch_size]
            X_tensor = torch.tensor(batch).float().to(device)
            batch_predictions = model(X_tensor)
            
            # Explicitly move to CPU and clear tensors to free VRAM
            predictions_list.append(batch_predictions.cpu().numpy())
            del X_tensor
            del batch_predictions

    # Concatenate all batch predictions
    pred_np = np.concatenate(predictions_list, axis=0)

    # Combine inputs and predictions for output
    results_df = pd.DataFrame(
        pred_np,
        columns=[
            f"{OUTPUT_DF_NAME}_predicted_x",
            f"{OUTPUT_DF_NAME}_predicted_y",
            f"{OUTPUT_DF_NAME}_predicted_z",
        ],
    )
    df_final = pd.concat([data_in, results_df], axis=1)

    # Save results to disk (optimized for large datasets)
    os.makedirs(local_dest, exist_ok=True)
    output_file = os.path.join(local_dest, f"velocity_{serie}.csv")

    logging.info("Saving results to disk...")
    if len(df_final) > 1000000:
        df_final.to_csv(output_file, index=False)
    else:
        df_final.to_csv(output_file, index=False, float_format="%.12f")

    logging.info(f"Execution finished. Results saved at: {output_file}")

    # Validation against reference data
    validate_synthetic_results(serie, df_final, local_dest, block_size=block_size, gap=gap_threshold)


def main():
    parser = argparse.ArgumentParser(description="Run trained MLP on new voltage data")
    parser.add_argument("serie", help="Series identifier (e.g. 21180)")
    parser.add_argument("model_filename", help="Trained .pth model file stored in models directory")
    parser.add_argument("--block-size", type=int, default=None, help="Fixed block length for metrics")
    parser.add_argument("--gap", type=float, default=None, help="Gap threshold (s) for metrics segmentation")
    
    args = parser.parse_args()
    
    start_time = time.time()
    run_model(args.serie, args.model_filename, args.block_size, args.gap)
    logging.info(f"Total execution time: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()