import os
import sys
import time
import threading

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
input_size = 3          # define o formato de entrada dos dados, no caso, 3 entradas para 3 saída (x,y,z)
output_size = 3

hidden_layers = 2       # 2 comum
hidden_size = 8         # 8 comum

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
# Rodar a rede em dados de entrada
# $ python3 model_mlp_v{VERSION}.py run name_result model.pth data_voltage.csv data_complete_with_velocity.csv

SERIE = sys.argv[1]                                     #   Nome da do resultado
local_modelo = (f"{model_local}/{sys.argv[2]}")                   #   Local do modelo     
local_data = (f"{dir_base}/data/run/run_{SERIE}.csv")              #   Local dos dados de entrada tensão
local_destino = (f"{dir_base}/data/run/run_results/velocity_{SERIE}")                             #   Local de saida desejado para os resultados da rede
print("\n\n -- -- -- -- - -- -- -- ")
print("Nome de série:\t",SERIE)
print("\n\n Rede processando dados de tensão e gerando dados de saída ")
print(f"\n\t - Modelo usado: \t\t{local_modelo}\n\t - Usará os dados de:\t\t{local_data}\n\t - Será salvo no destino em: \t{local_destino} ")
print("\n\n -- -- -- -- - -- -- -- ")


'''
    Definição da classe que controla os parâmetros da arquitetura da rede
        - Número de camadas e neurônios de cada rede
        - formato de entrada e saída da rede
        - definição das funções de ativação

'''

#   Definindo a Thread de tempo
def processing(start_time):
    
    # print('|Processing|')
    for i in range(50000):
        current_time = time.time()- start_time
        if i %2==0:
            print(f'\r |{current_time:4.0f}|.',end='')
        else:
            print(f'\r.|{current_time:4.0f}| ',end='')
        time.sleep(.5)
        
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=hidden_size, num_hidden_layers=hidden_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

'''
    Uso de uma classe para criar os objetos dataset para treino
        - Separados por dataset de treino e de validação
'''


class VoltageVelocityDataset(Dataset):
    def __init__(self, data):
        self.X = (torch.tensor(
            data[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z']].values).float().unsqueeze(1)).to(device)
        self.Y = (torch.tensor(
            data[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']].values).float().unsqueeze(1)).to(device)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def export_data_run(df,predictions,destino):
    if not os.path.exists(destino):
        os.makedirs(destino)

    predictions =pd.DataFrame(predictions.squeeze())
    predictions.columns =  [f'{output_df_name}_predicted_x', f'{output_df_name}_predicted_y', f'{output_df_name}_predicted_z']
    df_exp = pd.concat([df, predictions], axis=1)

    # df_exp = pd.DataFrame[['time',f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z',f'{output_df_name}_predicted_x', f'{output_df_name}_predicted_y', f'{output_df_name}_predicted_z']]
    print(f"\n\nDataFrame de entrada:\n\n{df}")
    print(f"\n\nDataFrame resultante junto com a entrada:\n\n{df_exp}")
    df_exp.to_csv(f'{local_destino}/velocity_{SERIE}.csv', index=False)

def runModel(local_modelo,local_data,local_destino):
    model = MLP(input_dim=input_size, output_dim=output_size, hidden_dim=hidden_size, num_hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(local_modelo))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Prepare the data (adjust this part based on your data loading requirements)
    # For example, you might need to load a CSV file similar to what you did in the main function
    data_in = pd.read_csv(local_data,sep=',')
    predictions=pd.DataFrame()

    # resultado = pd.DataFrame(predictions.squeeze().numpy(), columns=['voltage_x','voltage_y','voltage_z'])
    # print(f"Resultado:\n{resultado} \n\nAccuracy {accuracy}")
    # Make predictions
    with torch.no_grad():
        X = torch.tensor(data_in[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z']].values).float().unsqueeze(1).to(device)
        # Y = torch.tensor(data_in[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']].values).float().unsqueeze(1).to(device)
        predictions = model(X)
    
    # Evaluate the model (calculate mean squared error)
    # mse = nn.MSELoss()
    # loss = mse(predictions, Y)
    # print("Mean Squared Error:", loss.item())
    
    # # Export predicted data
    export_data_run(data_in, predictions,local_destino)

def main():
    
    print("\n\t Previsão pela rede neural !")
    
    runModel(local_modelo=local_modelo, local_data=local_data, local_destino=local_destino)
    
    print(f"\nA execução total do código durou: { time.time()- START_TIME:.2f} segundos")
    print(f"\nOs resultados pode sem encontrado em {local_destino}")


main()



