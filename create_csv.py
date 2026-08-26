"""CSV data preparation from raw hot-film measurements.

This module processes raw voltage data from hot-film sensors and sonic anemometer
readings into standardized CSV formats for model training and inference. It handles
time synchronization between different sampling rates.

Usage:
    python3 create_csv.py train <series_id>
    python3 create_csv.py run <series_id>
"""

import os
import logging
import argparse
import pandas as pd

# Import utility modules
from utils import data_loader

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_create_CSV(serie: str):
    """Merge voltage and velocity data for supervised training.

    Loads hot-film voltage measurements and corresponding sonic anemometer
    velocity references, synchronizes by timestamp. Output is rounded to 12 
    decimals for numerical precision.
    
    Args:
        serie: The dataset series identifier.
    """
    logging.info(f"Starting training CSV generation for series {serie}...")
    try:
        voltage_df = data_loader.load_voltage_data(serie)
        velocity_df = data_loader.load_velocity_data(serie)
        df_final = data_loader.synchronize_and_merge(voltage_df, velocity_df)

        logging.info("Final training DataFrame (preview):")
        logging.info(f"\n{df_final.head()}")

        output_dir = "./data/train"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"train_df_{serie}.csv")
        
        df_final.to_csv(output_path, index=False)
        logging.info(f"Training dataset ready. Saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create training CSV for series {serie}. Error: {e}")

def run_create_CSV(serie: str):
    """Prepare voltage data for model inference.

    Loads hot-film voltage measurements formatting for input to 
    pre-trained prediction models.
    
    Args:
        serie: The dataset series identifier.
    """
    logging.info(f"Starting inference CSV generation for series {serie}...")
    try:
        voltage_data = data_loader.load_run_data(serie)

        output_dir = "./data/run"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"run_{serie}.csv")
        
        voltage_data.to_csv(output_path, index=False)
        logging.info(f"Inference dataset ready. Saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create inference CSV for series {serie}. Error: {e}")

def main():
    """Main execution function to parse arguments and route to the correct mode."""
    parser = argparse.ArgumentParser(
        description="CSV data preparation from raw hot-film measurements.",
        epilog="Check the manual inside the 'manual/' folder to place the correct data before generating CSV files."
    )
    parser.add_argument(
        "mode", 
        choices=["train", "run"], 
        help="Mode of operation: 'train' to merge voltage/velocity, 'run' for inference voltage only."
    )
    parser.add_argument(
        "serie", 
        type=str, 
        help="Series identifier (e.g., 5940, 21180)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_create_CSV(args.serie)
    elif args.mode == "run":
        run_create_CSV(args.serie)

if __name__ == "__main__":
    main()