import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils import spectral_utils, config

print("Starting spectrum.py")

info_output = """
Check the manual inside the following folder to place the correct data to generate the spectrum:
    manuals/manual.txt
Usage: python3 script_name.py <SERIE> [predicted_path] [sonic_path]
"""

# Implementação robusta de argumentos para aceitar os caminhos passados pelo terminal
parser = argparse.ArgumentParser(description="Spectrum plotter", usage=info_output)
parser.add_argument("serie", help="Series identifier")
parser.add_argument("predicted_path", nargs="?", default=None, help="Path to predicted CSV")
parser.add_argument("sonic_path", nargs="?", default=None, help="Path to sonic CSV/DAT")

try:
    args = parser.parse_args()
except SystemExit:
    sys.exit(1)

SERIE = args.serie


# ==============================================================================
# FUNÇÃO ROBUSTA DE CARREGAMENTO (Evita o KeyError)
# ==============================================================================
def load_velocity_data(filepath):
    """Carrega dados garantindo a existência das colunas corretas (ignora cabeçalhos ausentes)"""
    if filepath.endswith('.dat'):
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
    else:
        df = pd.read_csv(filepath, sep=",")
        # Se as colunas padrão não existirem, força a leitura sem cabeçalho
        if "velocity_x" not in df.columns:
            df = pd.read_csv(filepath, sep=",", header=None)
            
    # Renomeia as primeiras 4 colunas para o padrão do script
    if len(df.columns) >= 4:
        df = df.rename(columns={0: "time", 1: "velocity_x", 2: "velocity_y", 3: "velocity_z"})
        
    # Se a primeira linha originalmente era texto (ex: ['t', 'u', 'v', 'w']), descartá-la
    if isinstance(df["time"].iloc[0], str):
        df = df.iloc[1:].reset_index(drop=True)
        df = df.astype(float)
        
    return df

# ==============================================================================
# CARREGAMENTO DOS DADOS (Respeitando os argumentos do terminal)
# ==============================================================================
predicted_path = args.predicted_path or f"./data/run/results/velocity_{SERIE}/velocity_{SERIE}.csv"
data_predicted = pd.read_csv(predicted_path, sep=",")

if args.sonic_path and os.path.exists(args.sonic_path):
    print(f"[Spectrum] Loading custom sonic data from {args.sonic_path}")
    data_sonic = load_velocity_data(args.sonic_path)
else:
    sonic_raw_dat = f"./data/raw/{SERIE}/sonic_{SERIE}.dat"
    sonic_raw_csv = f"./data/raw/{SERIE}/sonic_{SERIE}.csv"
    sonic_train_path = f"./data/train/train_df_{SERIE}.csv"

    if os.path.exists(sonic_raw_dat):
        print(f"[Spectrum] Loading sonic raw data from {sonic_raw_dat}")
        data_sonic = load_velocity_data(sonic_raw_dat)
    elif os.path.exists(sonic_raw_csv):
        print(f"[Spectrum] Loading sonic raw data from {sonic_raw_csv}")
        data_sonic = load_velocity_data(sonic_raw_csv)
    else:
        print(f"[Spectrum] Loading sonic training data from {sonic_train_path}")
        data_sonic = load_velocity_data(sonic_train_path)


# Calculation of Mean
MEAN_VELOCITY = data_sonic["velocity_x"].mean()
print(
    f"Longitudinal mean velocity (u1_bar) calculated from Sonic: {MEAN_VELOCITY:.3f} m/s"
)

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

# Executes processing and plots the periodogram for the 3 components
# Define columns to look for
pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

# This replaces the 3 individual function calls and manual plotting
print("Generating modularized spectrum plot...")
output_path = f"data/run/results/velocity_{SERIE}/graphics/Combined_Periodogram.png"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

fs_pred_actual = spectral_utils.estimate_sampling_frequency(data_predicted, "time")
if fs_pred_actual is None:
    fs_pred_actual = config.FS_HOTFILM_DEFAULT

fs_sonic_actual = spectral_utils.estimate_sampling_frequency(data_sonic, "time")
if fs_sonic_actual is None:
    fs_sonic_actual = config.FS_SONIC_DEFAULT

print(
    f"[Spectrum] Using fs_pred = {fs_pred_actual} Hz, fs_sonic = {fs_sonic_actual} Hz"
)

try:
    spectral_utils.plot_combined_spectrum(
        data_predicted,
        pred_cols,
        fs_pred_actual,
        f"Spectral Comparison - Serie {SERIE}",
        output_path,
        sonic_df=data_sonic,
        fs_sonic=fs_sonic_actual,
    )
    print(f"Plot saved to {output_path}")
except Exception as e:
    print(f"Failed to generate plot: {e}")

plt.show()