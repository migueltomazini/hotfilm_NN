import os
import sys
import time
import threading
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')  # Use a backend that supports GUI (e.g., TkAgg)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib
import optuna  # For automatic hyperparameter tuning

# Suppress system/driver warnings
warnings.filterwarnings("ignore")

info_output = '''
Check the manual inside the following folder to place the correct data for training:
    manuals/manual.txt

Usage: python3 train_mlp.py <SERIE1> <SERIE2> ...
'''

''' 
    @author Lucas Duarte    
    *
    * Universal MLP Training with Optuna Optimization
    * Includes Reynolds Number as 4th input feature
'''
__author__ = "Lucas Sales Duarte"
__email__ = "lucassalesduarte026@gmail.com"
__status__ = "Production"

# --- Global Configurations ---
input_size = 4          # [voltage_x, voltage_y, voltage_z, reynolds]
output_size = 3         # [velocity_x, velocity_y, velocity_z]

# Default Hyperparameters (Will be overwritten by Optuna)
EPOCHS = 256
learning_rate = 0.001
batch_size = 32
hidden_layers = 2
hidden_size = 16

# MODES
EXPORT_DATA = True    
GRAPHS = True         
SAVE = True           
GPU = 0               
LOCAL = 1             

START_TIME = time.time()
input_df_name = "voltage"
output_df_name = "velocity"
model_local = './models'
caminho_local = '.'
caminho_cluster = '/home/lucasdu/algoritmo/2_cluster_architecture'
dir_base = caminho_local if LOCAL == 1 else caminho_cluster

if len(sys.argv) < 2 or sys.argv[1] == '?':
    print(info_output)
    sys.exit()

# Handle multiple series for Universal Model
SERIES_LIST = sys.argv[1:]
SERIE_IDENTIFIER = "_".join(SERIES_LIST)

device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
print(f"Processing Device: {device}\n")

# --- Time counting thread ---
stop_counting = 1
def processing(start_time):
    global stop_counting
    for i in range(86400):
        if stop_counting == 0: break 
        current_time = time.time() - start_time
        print(f'\r |{current_time:4.0f}|.' if i % 2 == 0 else f'\r.|{current_time:4.0f}| ', end='')
        time.sleep(.5)

# --- Neural Network Architecture ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure input is [batch, features]
        if x.dim() == 3: x = x.squeeze(1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

# --- Custom Dataset ---
class VoltageVelocityDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = torch.tensor(X_data).float().to(device)
        self.Y = torch.tensor(Y_data).float().to(device)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# --- Evaluation and Export Functions ---
def trained_info(data, predicted_df):
    output_df = data.reset_index(drop=True)
    diff = predicted_df.values - output_df.values
    diff_abs = np.abs(diff)
    return diff_abs.mean(), diff_abs.max(), (diff_abs[diff_abs > 0].min() if np.any(diff_abs > 0) else 0)

def export_data(df, predictions, train_loss, val_loss, accuracy, dir_base_local):
    pred_np = predictions.cpu().detach().numpy()
    if pred_np.ndim == 3: pred_np = pred_np.squeeze(1)
    predictions_df = pd.DataFrame(pred_np, columns=['predicted_x', 'predicted_y', 'predicted_z'])
    
    df_exp = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    dest_path = os.path.join(dir_base_local, f'data/train/results/results_{SERIE_IDENTIFIER}')
    if not os.path.exists(dest_path): os.makedirs(dest_path)
        
    df_exp.to_csv(f'{dest_path}/results_predict_{SERIE_IDENTIFIER}.csv', index=False, float_format='%.12f')
    pd.DataFrame(train_loss).to_csv(f'{dest_path}/results_train_{SERIE_IDENTIFIER}.csv', index=False)
    pd.DataFrame(val_loss).to_csv(f'{dest_path}/results_val_{SERIE_IDENTIFIER}.csv', index=False)
    
    m_err, max_err, min_err = trained_info(df[['velocity_x', 'velocity_y', 'velocity_z']], predictions_df)
    
    df_hyper = pd.DataFrame({
        'Mean_Diff': [m_err], 'Max_Diff': [max_err], 'Min_Diff': [min_err],
        'RMSE': [accuracy], 'Epochs': [EPOCHS], 'Layers': [hidden_layers],
        'Size': [hidden_size], 'LR': [learning_rate], 'Batch': [batch_size],
        'Time_s': [time.time()-START_TIME], 'Date': [time.strftime("%d/%m/%Y %H:%M")]
    })
    df_hyper.to_csv(f'{dest_path}/hyperparameters_{SERIE_IDENTIFIER}.csv', index=False)

def show_graphs(data, predictions, see_train_loss, see_val_loss):
    # Showing data
    shown = predictions
    if torch.is_tensor(shown):
        shown = pd.DataFrame(shown.squeeze().cpu().numpy(), columns = ['eixo_x','eixo_y','eixo_z'])

    # Define o caminho completo para a pasta de gráficos e cria recursivamente
    caminho_graficos = os.path.join(dir_base, f"data/train/train_results/results_{SERIE}/graphics/")
    if not os.path.exists(caminho_graficos):
        os.makedirs(caminho_graficos)
    
    # shown = shown.assign(original_x=data[[f'{output_df_name}_x']],original_y=data[[f'{output_df_name}_y']],original_z   =data[[f'{output_df_name}_z    ']])
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)

    # Figura 0: Eixo X
    plt.figure(0)
    # Plotting both the curves simultaneously
    plt.plot(data.time, data.velocity_x, color='r',alpha=1, label='data_eixo_x')
    plt.plot(data.time, shown.eixo_x, color='g',alpha=1, label='processed_x')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo X")
    plt.legend()
    plt.savefig(os.path.join(caminho_graficos, "Velocidade por tempo eixo x.png"), format='png')
    
    # Figura 1: Eixo Y
    plt.figure(1)
    # print(predictions)
    plt.plot(data.time, data.velocity_y, color='r', alpha=1 , label='data_eixo_y')
    plt.plot(data.time, shown.eixo_y, color='g', alpha=1 , label='processed_y')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo Y")
    plt.legend()
    plt.savefig(os.path.join(caminho_graficos, "Velocidade por tempo eixo y.png"), format='png')
    
    # Figura 2: Eixo Z
    plt.figure(2)
    plt.plot(data.time, data.velocity_z, color='r', alpha=1, label='data_eixo_z')
    plt.plot(data.time, shown.eixo_z, color='g', alpha=1 , label='processed_z')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo Z")
    plt.legend()
    plt.savefig(os.path.join(caminho_graficos, "Velocidade por tempo eixo z.png"), format='png')
    
    # Figura 3: Erro de Treino
    plt.figure(3)
    plt.title("Evolução do erro de treino ao longo do tempo")
    plt.plot(see_train_loss[see_train_loss.columns[0]], see_train_loss[see_train_loss.columns[1]],color='g', label='train')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()
    plt.savefig(os.path.join(caminho_graficos, "Erro do treino.png"), format='png')

    # Figura 4: Erro de Validação
    plt.figure(4)
    plt.title("Evolução do erro da validação ao longo do tempo")
    plt.plot(see_val_loss[see_val_loss.columns[0]], see_val_loss[see_val_loss.columns[1]],color='r', label='validation')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()
    plt.savefig(os.path.join(caminho_graficos, "Erro de validação.png"), format='png')

    # Mostrar os gráficos
    plt.show()

# --- Optuna Objective Function ---
def objective(trial, X_train, Y_train, X_val, Y_val):
    # Suggesting parameters for the trial
    n_layers = trial.suggest_int('hidden_layers', 1, 4)
    n_size = trial.suggest_int('hidden_size', 16, 128)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    b_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train), batch_size=b_size, shuffle=True)
    val_loader = DataLoader(VoltageVelocityDataset(X_val, Y_val), batch_size=b_size)

    model = MLP(input_size, output_size, n_size, n_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Fast training for trial evaluation
    for epoch in range(50): 
        model.train()
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward(); optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader: val_loss += criterion(model(X), Y).item()
    return val_loss / len(val_loader)

# --- Final Training Function ---
def train_final(X_train, Y_train, X_val, Y_val):
    train_loader = DataLoader(VoltageVelocityDataset(X_train, Y_train), batch_size, shuffle=True)
    val_loader = DataLoader(VoltageVelocityDataset(X_val, Y_val), batch_size)

    mlp = MLP(input_size, output_size, hidden_size, hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    t_history, v_history = [], []
    idx_t, idx_v = 0, 0

    for epoch in range(EPOCHS):
        mlp.train()
        epoch_t_loss = 0
        for X, Y in train_loader:
            optimizer.zero_grad()
            loss = criterion(mlp(X), Y)
            loss.backward(); optimizer.step()
            t_history.append([idx_t, loss.item()]); idx_t += 1
            epoch_t_loss += loss.item() * X.shape[0]
            
        mlp.eval()
        epoch_v_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                loss = criterion(mlp(X), Y)
                v_history.append([idx_v, loss.item()]); idx_v += 1
                epoch_v_loss += loss.item() * X.shape[0]

        if epoch % 32 == 0:
            print(f"| Epoch {epoch:4} | Train Loss {epoch_t_loss/len(X_train):.6f} | Val Loss {epoch_v_loss/len(X_val):.6f} |")
    return mlp, np.array(t_history), np.array(v_history)

def main():
    global stop_counting, hidden_layers, hidden_size, learning_rate, batch_size
    timeCounter_thread = threading.Thread(target=processing, args=(START_TIME,))
    timeCounter_thread.daemon= True; timeCounter_thread.start()

    dfs = []
    for s in SERIES_LIST:
        df = pd.read_csv(f'{dir_base}/data/train/train_df_{s}.csv')
        if 'reynolds' not in df.columns:
            json_path = f'{dir_base}/data/config/config_{s}.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    cfg = json.load(f)
                    df['reynolds'] = cfg.get('RE_NUMBER', cfg.get('RE_EXPECTED', 0.0))
            else: df['reynolds'] = 0.0
        dfs.append(df)
    
    df_total = pd.concat(dfs, ignore_index=True)
    X_raw = df_total[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values
    Y_raw = df_total[['velocity_x', 'velocity_y', 'velocity_z']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    if not os.path.exists(model_local): os.makedirs(model_local)
    joblib.dump(scaler, f'{model_local}/scaler_{SERIE_IDENTIFIER}.joblib')

    indices = np.random.permutation(len(X_scaled))
    split = int(0.9 * len(X_scaled))
    X_train, X_val = X_scaled[indices[:split]], X_scaled[indices[split:]]
    Y_train, Y_val = Y_raw[indices[:split]], Y_raw[indices[split:]]

    print(f"\n[Optuna] Starting automatic optimization for series: {SERIES_LIST}")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, Y_train, X_val, Y_val), n_trials=50)

    print("\nBest parameters found:", study.best_params)
    hidden_layers, hidden_size = study.best_params['hidden_layers'], study.best_params['hidden_size']
    learning_rate, batch_size = study.best_params['learning_rate'], study.best_params['batch_size']

    print("\nStarting final high-precision training...")
    model, t_loss, v_loss = train_final(X_train, Y_train, X_val, Y_val)
    stop_counting = 0
    
    with torch.no_grad():
        all_X = torch.tensor(X_scaled).float().to(device)
        all_Y = torch.tensor(Y_raw).float().to(device)
        predictions = model(all_X)
        accuracy = ((predictions - all_Y) ** 2).mean().sqrt().item()

    if SAVE: torch.save(model.state_dict(), f'{model_local}/model_mlp_{SERIE_IDENTIFIER}.pth')
    if EXPORT_DATA: export_data(df_total, predictions, t_loss, v_loss, accuracy, dir_base)
    if GRAPHS: show_graphs(df_total, predictions, t_loss, v_loss)
    print(f"\n\t Training finished! Model: model_mlp_{SERIE_IDENTIFIER}.pth")

if __name__ == "__main__":
    main()