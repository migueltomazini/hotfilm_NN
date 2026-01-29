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
matplotlib.use('TkAgg')  # Use a backend that supports GUI (e.g., TkAgg)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

info_output = '''

Confira o manual dentro da pasta a seguir para colocar os dados corretos para treinamento:
    manuals/manual.txt

'''
''' 
    @author lucas
    *
    *
    *
    *
'''
__author__ = "Lucas Sales Duarte"
__email__ = "lucassalesduarte026@gmail.com"
__status__ = "Production"

#   Definições Globais
input_size = 4          # Mantido 4 entradas (x,y,z + reynolds)
output_size = 3

# Serão sobrescritas automaticamente pelo carregamento de metadados
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

# Seleção do dispositivo de processamento
device = "cpu"
print(f"Device de processamento: {device}\n")
    
# ______________________________________________-_- RODAR -_-______________________________________________________

SERIE = sys.argv[1]                                     
local_modelo = (f"{model_local}/{sys.argv[2]}")                   
local_data = (f"{dir_base}/data/run/run_{SERIE}.csv")              
local_destino = (f"{dir_base}/data/run/results/velocity_{SERIE}")                             

# Identifica o ID do modelo para buscar metadados e scaler
MODEL_ID = sys.argv[2].replace("model_mlp_", "").replace(".pth", "")

print("\n\n -- -- -- -- - -- -- -- ")
print("Nome de série:\t",SERIE)
print("\n\n Rede processando dados de tensão e gerando dados de saída ")
print(f"\n\t - Modelo usado: \t\t{local_modelo}\n\t - Usará os dados de:\t\t{local_data}\n\t - Será salvo no destino em: \t{local_destino} ")
print("\n\n -- -- -- -- - -- -- -- ")

def get_model_metadata(model_id):
    """ Busca os parâmetros da rede nos metadados do treino """
    path = f'{dir_base}/data/train/results/results_{model_id}/hyperparameters_{model_id}.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadados não encontrados em {path}")
    df_meta = pd.read_csv(path)
    layers = int(df_meta['Layers'].iloc[0]) if 'Layers' in df_meta.columns else int(df_meta.get('hidden_layers', [1])[0])
    size = int(df_meta['Size'].iloc[0]) if 'Size' in df_meta.columns else int(df_meta.get('hidden_size', [114])[0])
    return layers, size

#   Definindo a Thread de tempo
def processing(start_time):
    for i in range(50000):
        current_time = time.time()- start_time
        if i %2==0:
            print(f'\r |{current_time:4.0f}|.',end='')
        else:
            print(f'\r.|{current_time:4.0f}| ',end='')
        time.sleep(.5)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

class VoltageVelocityDataset(Dataset):
    def __init__(self, data):
        self.X = (torch.tensor(data[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values).float()).to(device)
        self.Y = (torch.tensor(data[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']].values).float()).to(device)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def export_data_run(df,predictions,destino):
    if not os.path.exists(destino):
        os.makedirs(destino)

    predictions = pd.DataFrame(predictions.squeeze())
    predictions.columns = [f'{output_df_name}_predicted_x', f'{output_df_name}_predicted_y', f'{output_df_name}_predicted_z']
    df_exp = pd.concat([df, predictions], axis=1)

    print(f"\n\nDataFrame de entrada:\n\n{df.head()}")
    df_exp.to_csv(f'{local_destino}/velocity_{SERIE}.csv', index=False, float_format='%.12f')

def runModel(local_modelo,local_data,local_destino):
    # Carregamento automático dos parâmetros
    h_layers, h_size = get_model_metadata(MODEL_ID)
    
    model = MLP(input_dim=input_size, output_dim=output_size, hidden_dim=h_size, num_hidden_layers=h_layers)
    model.load_state_dict(torch.load(local_modelo))
    model.to(device)
    model.eval()
    
    data_in = pd.read_csv(local_data, sep=',')
    
    # Carregamento do Scaler salvo no treino
    scaler_path = f'{model_local}/scaler_{MODEL_ID}.joblib'
    scaler = joblib.load(scaler_path)

    # Preparação dos dados com 4 colunas e aplicação do scaler
    X_raw = data_in[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z', 'reynolds']].values
    X_scaled = scaler.transform(X_raw)
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled).float().to(device)
        predictions = model(X_tensor)
    
    export_data_run(data_in, predictions, local_destino)

def main():
    print("\n\t Previsão pela rede neural !")
    runModel(local_modelo=local_modelo, local_data=local_data, local_destino=local_destino)
    print(f"\nA execução total do código durou: { time.time()- START_TIME:.2f} segundos")
    print(f"\nOs resultados podem ser encontrados em {local_destino}")

if __name__ == "__main__":
    main()