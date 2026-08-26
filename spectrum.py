"""Spectrum analysis and plotting tool for hot-film velocity predictions.

This module generates combined power spectral density (PSD) plots, comparing 
the neural network predicted velocities against the sonic anemometer reference data.
It handles automatic frequency estimation and fallback reference loading.
"""

import os
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils import spectral_utils, config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================================================================
# ROBUST DATA LOADING 
# ==============================================================================
def load_velocity_data(filepath: str) -> pd.DataFrame:
    """Load velocity data ensuring the correct columns exist, bypassing missing headers.
    
    Args:
        filepath: Path to the CSV or DAT file containing velocity data.
        
    Returns:
        DataFrame containing formatted time and velocity components.
    """
    if filepath.endswith('.dat'):
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
    else:
        df = pd.read_csv(filepath, sep=",")
        # If standard columns do not exist, force headerless reading
        if "velocity_x" not in df.columns:
            df = pd.read_csv(filepath, sep=",", header=None)
            
    # Rename the first 4 columns to match the script's standard
    if len(df.columns) >= 4:
        df = df.rename(columns={0: "time", 1: "velocity_x", 2: "velocity_y", 3: "velocity_z"})
        
    # If the first row was originally text (e.g., ['t', 'u', 'v', 'w']), discard it
    if isinstance(df["time"].iloc[0], str):
        df = df.iloc[1:].reset_index(drop=True)
        df = df.astype(float)
        
    return df

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Spectrum plotter for comparing NN predictions vs Sonic reference.",
        epilog="Usage: python3 spectrum.py <SERIE> [predicted_path] [sonic_path]"
    )
    parser.add_argument("serie", help="Series identifier (e.g., 0610)")
    parser.add_argument("predicted_path", nargs="?", default=None, help="Path to predicted CSV")
    parser.add_argument("sonic_path", nargs="?", default=None, help="Path to sonic CSV/DAT reference")

    args = parser.parse_args()
    serie = args.serie

    # Load predicted data
    predicted_path = args.predicted_path or f"./data/run/results/velocity_{serie}/velocity_{serie}.csv"
    if not os.path.exists(predicted_path):
        logging.error(f"Predicted data not found at {predicted_path}. Exiting.")
        return
        
    logging.info(f"Loading predicted data from {predicted_path}")
    data_predicted = pd.read_csv(predicted_path, sep=",")

    # Load reference sonic data
    if args.sonic_path and os.path.exists(args.sonic_path):
        logging.info(f"Loading custom sonic reference data from {args.sonic_path}")
        data_sonic = load_velocity_data(args.sonic_path)
    else:
        sonic_raw_dat = f"./data/raw/{serie}/sonic_{serie}.dat"
        sonic_raw_csv = f"./data/raw/{serie}/sonic_{serie}.csv"
        sonic_train_path = f"./data/train/train_df_{serie}.csv"

        if os.path.exists(sonic_raw_dat):
            logging.info(f"Loading sonic raw data from {sonic_raw_dat}")
            data_sonic = load_velocity_data(sonic_raw_dat)
        elif os.path.exists(sonic_raw_csv):
            logging.info(f"Loading sonic raw data from {sonic_raw_csv}")
            data_sonic = load_velocity_data(sonic_raw_csv)
        elif os.path.exists(sonic_train_path):
            logging.info(f"Loading sonic training data from {sonic_train_path}")
            data_sonic = load_velocity_data(sonic_train_path)
        else:
            logging.error("No valid sonic reference data found. Spectrum generation requires a reference.")
            return

    # Calculation of Mean
    mean_velocity = data_sonic["velocity_x"].mean()
    logging.info(f"Longitudinal mean velocity (u1_bar) calculated from Sonic: {mean_velocity:.3f} m/s")

    # Define columns to look for
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    logging.info("Generating modularized spectrum plot...")
    output_path = f"data/run/results/velocity_{serie}/graphics/Combined_Periodogram.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Estimate sampling frequencies
    fs_pred_actual = spectral_utils.estimate_sampling_frequency(data_predicted, "time")
    if fs_pred_actual is None:
        fs_pred_actual = config.FS_HOTFILM_DEFAULT

    fs_sonic_actual = spectral_utils.estimate_sampling_frequency(data_sonic, "time")
    if fs_sonic_actual is None:
        fs_sonic_actual = config.FS_SONIC_DEFAULT

    logging.info(f"Using fs_pred = {fs_pred_actual} Hz, fs_sonic = {fs_sonic_actual} Hz")

    try:
        spectral_utils.plot_combined_spectrum(
            data_predicted,
            pred_cols,
            fs_pred_actual,
            f"Spectral Comparison - Serie {serie}",
            output_path,
            sonic_df=data_sonic,
            fs_sonic=fs_sonic_actual,
        )
        logging.info(f"Plot successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate spectrum plot: {e}")

if __name__ == "__main__":
    main()