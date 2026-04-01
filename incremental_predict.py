"""Use incrementally trained block models to predict velocities on new/unseen data.

This script takes the sequence of models and scalers produced by incremental_train.py
(e.g., model_0610_block1.pth, model_0610_block2.pth) and applies them sequentially
to corresponding blocks of new data. This ensures that the fine-tuned state of
each block is used for its respective data segment.

Usage examples:
    python3 incremental_predict.py 0610 --num-blocks 10
    python3 incremental_predict.py 0610 --num-blocks 10 --calc-metrics

The script reads from data/run/run_df_<serie>.csv, splits it into N blocks,
applies the specific model for each block, and saves combined predictions.
"""

import os
import json
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import butter, filtfilt

from utils import config, metrics, physics, spectral_utils
import train_mlp
from train_mlp import MLP, VoltageVelocityDataset

device = torch.device(
    "cuda" if torch.device("cuda" if torch.cuda.is_available() else "cpu") else "cpu"
)


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

    # Infer model architecture (using defaults 64, 2 as per training script)
    model = MLP(config.INPUT_SIZE, config.OUTPUT_SIZE, 64, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, scaler


def predict_on_block(
    model: MLP, scaler: StandardScaler, df: pd.DataFrame
) -> np.ndarray:
    """Generate predictions for a specific dataframe block.

    Args:
        model: trained MLP model for this specific block.
        scaler: fitted StandardScaler for this specific block.
        df: dataframe with voltage columns and reynolds.

    Returns:
        numpy array of shape (N, 3) with predicted velocities.
    """
    X_raw = df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    X_scaled = scaler.transform(X_raw)

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled).float().to(device))
    return preds.cpu().numpy()


def calculate_delta_metrics(y_true, y_pred, fs, cutoff=2.0):
    """
    Calcula o parâmetro Delta conforme Freire et al. (2023) Eq. 16.
    y_true/y_pred: arrays (N, 3)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Filtro de 4ª ordem (Butterworth)
    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    deltas = []
    for i in range(3):
        # Filtro filtfilt para fase zero (não atrasar o sinal)
        s_true = filtfilt(b, a, y_true[:, i])
        s_pred = filtfilt(b, a, y_pred[:, i])

        # Normalização (Z-score)
        s_true_norm = (s_true - np.mean(s_true)) / np.std(s_true)
        s_pred_norm = (s_pred - np.mean(s_pred)) / np.std(s_pred)

        # RMS da diferença
        delta = np.sqrt(np.mean((s_pred_norm - s_true_norm) ** 2))
        deltas.append(delta)

    return deltas


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

        # Load the specific model saved during incremental_train.py
        model, scaler = load_block_model_and_scaler(serie, block_num)

        preds = predict_on_block(model, scaler, block_df)
        all_preds.append(preds)

    # Combine all block predictions
    final_preds = np.vstack(all_preds)

    # Add predictions to dataframe
    df["velocity_x_pred"] = final_preds[:, 0]
    df["velocity_y_pred"] = final_preds[:, 1]
    df["velocity_z_pred"] = final_preds[:, 2]

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

    # --- CÁLCULO DE MÉTRICAS (FOCO NO ARTIGO) ---
    if args.calc_metrics:
        target_cols = ["velocity_x", "velocity_y", "velocity_z"]

        if all(col in df.columns for col in target_cols):
            print(f"\n{'='*50}")
            print(f"📊 VALIDAÇÃO: MÉTRICAS DO ARTIGO (FREIRE ET AL, 2023)")
            print(f"{'='*50}")

            # Limpeza e extração de dados
            df_clean = df.dropna(subset=target_cols + ["velocity_x_pred"])
            Y_true = df_clean[target_cols].values
            Y_pred = df_clean[
                ["velocity_x_pred", "velocity_y_pred", "velocity_z_pred"]
            ].values

            # Pegar FS para o filtro
            with open(
                os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
            ) as fh:
                cfg = json.load(fh)
            fs = cfg["FS_HOTFILM"]

            # 1. RMSE Bruto (ponto a ponto)
            rmse_global = metrics.calculate_rmse(Y_pred, Y_true)

            # 2. Parâmetro Delta (Sinal filtrado em 2Hz e normalizado)
            deltas = calculate_delta_metrics(Y_true, Y_pred, fs)

            print(f"Registros avaliados: {len(df_clean)}")
            print(f"RMSE Global Bruto:  {rmse_global:.6f}")
            print(f"{'-'*50}")
            print(f"Delta_u1 (Eixo X):  {deltas[0]:.4f}")
            print(f"Delta_u2 (Eixo Y):  {deltas[1]:.4f}")
            print(f"Delta_u3 (Eixo Z):  {deltas[2]:.4f}")
            print(f"{'-'*50}")
        else:
            print(
                "\n[Aviso] Colunas de velocidade não encontradas para cálculo de métricas."
            )

    print("\nIncremental block prediction complete.")

    # --- AUTOMATIC SPECTRAL ANALYSIS ---
    print(f"\n{'='*50}")
    print(f"📊 GENERATING AUTOMATIC SPECTRAL ANALYSIS")
    print(f"{'='*50}")

    # Define output directory for plots
    spectral_dir = os.path.join(output_dir, "plots_spectral")
    os.makedirs(spectral_dir, exist_ok=True)

    # Get FS from config
    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs_hf = cfg["FS_HOTFILM"]

    pred_cols = ["velocity_x_pred", "velocity_y_pred", "velocity_z_pred"]

    # 1. Global Spectrum (All data combined)
    print("Generating global spectrum...")
    global_plot_path = os.path.join(spectral_dir, f"spectrum_global_{serie}.png")
    spectral_utils.plot_spectral_density(
        df, pred_cols, fs_hf, f"Global Spectrum - Serie {serie}", global_plot_path
    )

    # 2. Block-wise Spectrums
    print(f"Generating spectra for {args.num_blocks} incremental blocks...")
    # split_indices was defined earlier in main()
    for i, idx_list in enumerate(split_indices):
        block_df = df.iloc[idx_list]
        block_plot_path = os.path.join(
            spectral_dir, f"spectrum_block_{i+1}_{serie}.png"
        )
        spectral_utils.plot_spectral_density(
            block_df,
            pred_cols,
            fs_hf,
            f"Block {i+1} Spectrum - Serie {serie}",
            block_plot_path,
        )

    print(f"Spectral plots saved to: {spectral_dir}")
    print("\nIncremental block prediction complete.")


if __name__ == "__main__":
    main()
