"""Wind velocity prediction using trained MLP models.

This module loads a pre-trained Multi-Layer Perceptron (MLP) model to predict
wind velocity components from hot-film voltage measurements. It handles model
architecture reconstruction from metadata, input scaling using pre-trained scalers,
and output validation against synthetic ground truth when available.

Usage:
    python3 run_mlp.py <series_id> <model_filename>

Example:
    python3 run_mlp.py 21180 model_mlp_21180.pth
"""

import os
import sys
import time
import threading
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

info_output = '''
Usage: python3 run_mlp.py <series_id> <model_filename>

Check the manual inside the following folder for data preparation details:
    manuals/manual.txt
'''

# Global Definitions
# Input size for the model
input_size = 4          # Maintained 4 inputs (x,y,z + reynolds)
output_size = 3

# Will be overwritten automatically by metadata loading
hidden_layers = 1
hidden_size = 114

START_TIME = time.time()
train_data_df=''
input_df_name = "voltage"
output_df_name = "velocity"
model_local = './models'
caminho_local = '.'
caminho_cluster = '/home/lucasdu/algoritmo/2_cluster_architecture'
dir_base = caminho_local

if (len(sys.argv) < 2 or sys.argv[1]== '?'):
    print(info_output)
    sys.exit()

# Select processing device
device = "cpu"
print(f"Processing device: {device}\n")
    
# ______________________________________________-_- RODAR -_-______________________________________________________

SERIE = sys.argv[1]                                     
local_modelo = (f"{model_local}/{sys.argv[2]}")                   
local_data = (f"{dir_base}/data/run/run_{SERIE}.csv")              
local_destino = (f"{dir_base}/data/run/results/velocity_{SERIE}")                             

# Identifica o ID do modelo para buscar metadados e scaler
MODEL_ID = sys.argv[2].replace("model_mlp_", "").replace(".pth", "")

print("\n\n -- -- -- -- - -- -- -- ")
print("Series name:\t", SERIE)
print("\n\n Network processing voltage data and generating output ")
print(f"\n\t - Model used: \t\t{local_modelo}\n\t - Using data from: \t\t{local_data}\n\t - Will be saved at destination: \t{local_destino} ")
print("\n\n -- -- -- -- - -- -- -- ")

def get_model_metadata(model_id):
    """Retrieve network architecture parameters from training metadata.

    Loads the hyperparameters CSV file generated during model training and extracts
    the network's layer count and hidden layer size. Falls back to defaults if
    expected columns are not found.

    Args:
        model_id: The model identifier matching the training session.

    Returns:
        tuple: (num_hidden_layers, hidden_size) for network reconstruction.

    Raises:
        FileNotFoundError: If metadata file is not found.
    """
    path = f'{dir_base}/data/train/results/results_{model_id}/hyperparameters_{model_id}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found at {path}")
    df_meta = pd.read_csv(path)
    # Handle column name variations from different training runs
    layers = int(df_meta['Layers'].iloc[0]) if 'Layers' in df_meta.columns else int(df_meta.get('hidden_layers', [1])[0])
    size = int(df_meta['Size'].iloc[0]) if 'Size' in df_meta.columns else int(df_meta.get('hidden_size', [114])[0])
    return layers, size

def processing(start_time):
    """Display elapsed time with visual feedback during model execution.

    Args:
        start_time: Reference time (from time.time()) to calculate elapsed duration.
    """
    for i in range(50000):
        current_time = time.time() - start_time
        # Alternating display pattern for visual feedback
        if i % 2 == 0:
            print(f'\r |{current_time:4.0f}|.', end='')
        else:
            print(f'\r.|{current_time:4.0f}| ', end='')
        time.sleep(.5)

class MLP(nn.Module):
    """Multi-Layer Perceptron for voltage-to-velocity prediction.

    Converts hot-film voltage measurements to velocity components using
    configurable fully-connected layers with ReLU activations.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
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

    def __init__(self, data):
        """Initialize dataset from DataFrame.

        Args:
            data: DataFrame with columns for voltage components, reynolds number, and velocities.
        """
        self.X = torch.tensor(data[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values).float().to(device)
        self.Y = torch.tensor(data[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']].values).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def validate_synthetic_results(serie, df_predicted):
    """Compare predictions against synthetic ground truth if available.

    Loads reference velocity data and calculates RMSE for each velocity component.
    Handles length mismatches by truncating longer arrays.
    """
    ref_path = f"./data/run/raw/collected_data_{serie}/hotfilm_vel_{serie}.csv"
    
    if os.path.exists(ref_path):
        print(f"\n[Validation] Synthetic ground truth found: {ref_path}")
        try:
            # Reference file columns: time, vel_x, vel_y, vel_z
            df_ref = pd.read_csv(ref_path, sep=',', names=['time', 'velocity_x', 'velocity_y', 'velocity_z'])
        except Exception as e:
            print(f"-> Warning: Could not parse reference file: {e}")
            return
        
        pred_x = df_predicted[f'{output_df_name}_predicted_x'].values
        pred_y = df_predicted[f'{output_df_name}_predicted_y'].values
        pred_z = df_predicted[f'{output_df_name}_predicted_z'].values
        
        ref_x = df_ref['velocity_x'].values
        ref_y = df_ref['velocity_y'].values
        ref_z = df_ref['velocity_z'].values
        
        min_len = min(len(pred_x), len(ref_x))
        if len(pred_x) != len(ref_x):
            print(f"-> Note: Aligning data lengths ({len(pred_x)} vs {len(ref_x)}). Truncating to {min_len} rows.")
            pred_x, pred_y, pred_z = pred_x[:min_len], pred_y[:min_len], pred_z[:min_len]
            ref_x, ref_y, ref_z = ref_x[:min_len], ref_y[:min_len], ref_z[:min_len]
        
        rmse_x = np.sqrt(np.mean((pred_x - ref_x)**2))
        rmse_y = np.sqrt(np.mean((pred_y - ref_y)**2))
        rmse_z = np.sqrt(np.mean((pred_z - ref_z)**2))
        
        print("-" * 45)
        print(f"ERROR ANALYSIS (Predicted vs Synthetic Ground Truth)")
        print(f"RMSE Velocity X: {rmse_x:.12f}")
        print(f"RMSE Velocity Y: {rmse_y:.12f}")
        print(f"RMSE Velocity Z: {rmse_z:.12f}")
        print("-" * 45)
    else:
        pass

def export_data_run(df, predictions, destino):
    """Save predictions to CSV file with input features.

    Combines input features and model predictions into a single DataFrame
    and exports with high precision formatting.

    Args:
        df: Input DataFrame with voltage and reynolds features.
        predictions: Model output predictions.
        destino: Directory path for output file.
    """
    if not os.path.exists(destino):
        os.makedirs(destino)

    predictions = pd.DataFrame(predictions.squeeze())
    predictions.columns = [f'{output_df_name}_predicted_x', f'{output_df_name}_predicted_y', f'{output_df_name}_predicted_z']
    df_exp = pd.concat([df, predictions], axis=1)

    print(f"\n\nInput DataFrame:\n\n{df.head()}")
    df_exp.to_csv(f'{local_destino}/velocity_{SERIE}.csv', index=False, float_format='%.12f')

def runModel():
    """Execute wind velocity prediction pipeline.

    Workflow:
        1. Reconstruct model architecture from training metadata
        2. Load input data and corresponding scaler
        3. Scale input features using pre-trained statistics
        4. Generate velocity predictions
        5. Export results to CSV
        6. Validate against synthetic ground truth if available
    """
    # Reconstruct network architecture based on metadata
    h_layers, h_size = get_model_metadata(MODEL_ID)
    model = MLP(input_dim=4, output_dim=3, hidden_dim=h_size, num_hidden_layers=h_layers)
    model.load_state_dict(torch.load(local_modelo, map_location=device))
    model.eval()

    # Load input data and apply feature scaling
    data_in = pd.read_csv(local_data)
    scaler_path = os.path.join(model_local, f'scaler_{MODEL_ID}.joblib')
    scaler = joblib.load(scaler_path)

    # Normalize input features using training statistics
    X_raw = data_in[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values
    X_scaled = scaler.transform(X_raw)

    # Perform predictions without computing gradients
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float().to(device)
        predictions = model(X_tensor)

    # Combine inputs and predictions for output
    pred_np = predictions.cpu().numpy()
    results_df = pd.DataFrame(pred_np, columns=[f'{output_df_name}_predicted_x', f'{output_df_name}_predicted_y', f'{output_df_name}_predicted_z'])
    df_final = pd.concat([data_in, results_df], axis=1)

    # Save results to disk
    if not os.path.exists(local_destino):
        os.makedirs(local_destino)
    output_file = os.path.join(local_destino, f'velocity_{SERIE}.csv')
    df_final.to_csv(output_file, index=False, float_format='%.12f')

    print(f"\nExecution finished. Results saved at: {output_file}")

    # Optional validation against reference data
    validate_synthetic_results(SERIE, df_final)

if __name__ == "__main__":
    print("\n\t Predicting wind velocity...")
    runModel()
    print(f"\nTotal execution time: {time.time() - START_TIME:.2f} s")
