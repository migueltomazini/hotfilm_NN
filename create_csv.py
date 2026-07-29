"""CSV data preparation from raw hot-film measurements.

This module processes raw voltage data from hot-film sensors and sonic anemometer
readings into standardized CSV formats for model training and inference. It handles
time synchronization between different sampling rates.

Usage:
    python3 create_csv.py <mode> <series_id>

Modes:
    train: Merge voltage and velocity reference data for model training
    run:   Prepare voltage data only for model inference

Example:
    python3 create_csv.py train 5940
    python3 create_csv.py run 21180
"""

import pandas as pd
import sys
import os

# Import utility modules
from utils import data_loader

info = """
Check the manual inside the following folder to place the correct data to generate CSV files:
    manual/manual.txt

Usage: python3 create_csv.py <mode: train/run> <series_id>
"""

# mode and serie are mandatory
if len(sys.argv) < 3:
    print(info)
    sys.exit()

mode = sys.argv[1]
serie = sys.argv[2]


# Function to create CSV files for training
def train_create_CSV():
    """Merge voltage and velocity data for supervised training.

    Loads hot-film voltage measurements and corresponding sonic anemometer
    velocity references, synchronizes by timestamp. Output is rounded to 12 
    decimals for numerical precision.
    """
    voltage_df = data_loader.load_voltage_data(serie)
    velocity_df = data_loader.load_velocity_data(serie)
    df_final = data_loader.synchronize_and_merge(voltage_df, velocity_df)

    print("\nFinal training DataFrame (preview):\n")
    print(df_final.head())

    output_path = f"./data/train/train_df_{serie}.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nTraining dataset ready. Saved to: {output_path}\n")


# Function to create CSV files for running
def run_create_CSV():
    """Prepare voltage data for model inference.

    Loads hot-film voltage measurements formatting for input to 
    pre-trained prediction models.
    """
    voltage_data = data_loader.load_run_data(serie)

    output_path = f"./data/run/run_{serie}.csv"
    voltage_data.to_csv(output_path, index=False)
    print(f"\nInference dataset ready. Saved to: {output_path}\n")


if mode == "train":
    train_create_CSV()
elif mode == "run":
    run_create_CSV()
else:
    print("Invalid mode. Please use 'train' or 'run'.")