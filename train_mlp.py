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
    -- -- -- -- -- -- Como usar o algorítmo -- -- -- -- -- -- 
    
    -- -- -- --  -- -- -- --  -- -- --
        -> Caso 1 
        
        - Treinar e rodar a rede com os mesmos dados:
    O dataframe tem de ser do tipo .CSV:
    - time, voltage_x, voltage_y, voltage_z, velocity_x, velocity_y, velocity_z

    python3 model_mlp_v{VERSION}.py train {nome final do arquivo exportado e controle de saida} {nome do dataframe de dados voltage-velocity}
    exemplo:
    $ python3 model_mlp_v{VERSION}.py train nome_de_controle nome_dataframe_treino
    
    -- -- -- --  -- -- -- --  -- -- --

        -> Caso 2 
        
        - Rodar no banco de dados com modelo salvo :
    formato dos dados de entrada
    - time, voltage_x, voltage_y, voltage_z

    python3 model_mlp_v{VERSION}.py "run" {nomear a saida} {nome do modelo salvo} {nome dos dados tensão de entrada} {nome dos dados velocidade de saida} 
    exemplo:

    $ python3 model_mlp_v{VERSION}.py run  name_result model.pth  data_voltage.csv data_complete_with_velocity.csv
    
    -- -- -- --  -- -- -- --  -- -- --
'''
''' 
    @author Lucas Duarte    
    *
    *
    *
    *
'''
__author__ = "Lucas Sales Duarte"
__email__ = "lucassalesduarte026@gmail.com"
__status__ = "Production"

# Hiper parâmetros

EPOCHS = 1000    #2000
input_size = 3          # define o formato de entrada dos dados, no caso, 3 entradas para 3 saída (x,y,z)
output_size = 3

hidden_layers = 2       # 2 comum
hidden_size = 8         # 8 comum


learning_rate = 0.01
batch_size = 32
VERSION = 11.0       # controle de versionamento

# MODOS
EXPORT_DATA = True    # Exporta arquivos .csv para analizar o resultado da rede
GRAPHS = True         # Mostrar os gráficos
SAVE = True           # Salvar o modelo
GPU =  1              # 0 para uso da CPU                   | 1 para uso da GPU 

LOCAL = 1             # 0 para no cluster                   | 1 para TREINO no notebook


#   Definições Globais
    
START_TIME = time.time()
input_df_name = "voltage"
output_df_name = "velocity"
model_local = './modelos'
caminho_local = '.'
caminho_cluster = '/home/lucasdu/algoritmo/2_cluster_architecture'
dir_base = ''
if LOCAL == 1:
    dir_base = caminho_local
elif LOCAL == 0:
    dir_base = caminho_cluster
    GRAPHS = False
    
    
    
# Seleção do dispositivo de processamento
device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")
print(f"Device de processamento: {device}\n")
    
if (len(sys.argv) < 2 or sys.argv[1]== '?'):
    print(info_output)
    sys.exit()
    

SERIE = sys.argv[1]            # Nome do treino em si - serie

# ______________________________________________-_- TREINO -_-______________________________________________________


# Carregar a rede 
df_train = pd.read_csv(f'{dir_base}/dados/treino/train_df_{SERIE}.csv', sep=",")

# Cabeçalho 
print(f' -- -- Tipos dos dados do df -- --  \n----------------------------\n{df_train.dtypes}\n----------------------------')
local_data = (f'{dir_base}/dados/treino/train_df_{SERIE}.csv')              #   Local dos dados de entrada tensão
local_destino = (f"{dir_base}/dados/treino/resultados_train/resultado_{SERIE}")                             #   Local de saida desejado para os resultados da rede
print("\n\n -- -- -- -- - -- -- -- ")
print("Nome de série:\t",SERIE)
print("\n\n Rede processando dados")
print(f"\n\t - Modelo usado: \t\t{model_local}\n\t - Usará os dados de:\t\t{local_data}\n\t - Será salvo no destino em: \t{local_destino} ")
print("\n\n -- -- -- -- - -- -- -- ")
sys.stdout.flush()

#   Definindo a Thread de contagem de tempo
stop_counting = 1
def processing(start_time):
        
    global stop_counting
    for i in range(86400): # um dia de tempo
        if stop_counting == 0:
            break 
        current_time = time.time()- start_time
        if i %2==0:
            print(f'\r |{current_time:4.0f}|.',end='')
        else:
            print(f'\r.|{current_time:4.0f}| ',end='')
        time.sleep(.5)
        
    

'''
    Definição da classe que controla os parâmetros da arquitetura da rede
        - Número de camadas e neurônios de cada rede
        - formato de entrada e saída da rede
        - definição das funções de ativação
'''
# Definindo a classe nn que contém o modelo
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

# Preparador dos doados para o formato necessário
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

# Função que exporta os metadados do treinamento
def export_data(df, predictions, see_train_loss, see_val_loss, accuracy, dir_base_local):
    
    predictions = pd.DataFrame(predictions.squeeze().cpu().numpy(), columns=['predicted_x','predicted_y','predicted_z'])
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)
    measure_df = df[['velocity_x', 'velocity_y', 'velocity_z']]
    df_exp = df[['time',f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z',f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']]
    df_exp = df_exp.join(predictions)

    caminho_completo = os.path.join(dir_base_local, f'dados/treino/resultados_train/resultado_{SERIE}')
    if not os.path.exists(caminho_completo):
        os.makedirs(caminho_completo)
    print('\nArquivo final:\n',df_exp)
    df_exp.to_csv(f'{caminho_completo}/resultado_predict_{SERIE}.csv', index=False)
    see_train_loss.to_csv(f'{caminho_completo}/resultado_train_{SERIE}.csv', index=False)
    see_val_loss.to_csv(f'{caminho_completo}/resultado_val_{SERIE}.csv', index=False)
    diff_media, diff_max, diff_min = trained_info(measure_df, predictions)
    print(
        f'\nMédia da diferença:\t{diff_media:6.6f}\nMáxima diferença:\t{diff_max:6.6f}\nMínima diferença:\t{diff_min:6.6f}\n')
    model_name_local = (f'{model_local}/model_mlp_{SERIE}.pth')
    
    # Calcular o tempo do fim do treino
    formatted_time = time.strftime("%d/%m/%Y %H:%M", time.localtime(time.time()))
    df_hyper = pd.DataFrame(
        {
            'Média': [diff_media],
            'Máximo': [diff_max],
            'Mínimo': [diff_min],
            'Accuracy (RMSE)': [accuracy],
            'Epochs': [EPOCHS],
            'hidden_layers': [hidden_layers],
            'hidden_size': [learning_rate],
            'learning_rate': [learning_rate],
            'batch_size': [batch_size],
            'hidden_size': [hidden_size],
            'model_name_local': [model_name_local],
            'data_treinamento': [formatted_time],
            
        })
    for column in df_hyper.columns:
        print(f"{column}: {df_hyper[column].values[0]}")
        
    df_hyper.to_csv(
        f'{caminho_completo}/hyperparameters_{SERIE}.csv', index=False)


def show_graphs(data, predictions, see_train_loss, see_val_loss):
    # Showing data
    shown = predictions
    if torch.is_tensor(shown):
        shown = pd.DataFrame(shown.squeeze().cpu().numpy(), columns = ['eixo_x','eixo_y','eixo_z'])
    
    # shown = shown.assign(original_x=data[[f'{output_df_name}_x']],original_y=data[[f'{output_df_name}_y']],original_z   =data[[f'{output_df_name}_z    ']])
    see_train_loss = pd.DataFrame(see_train_loss)
    see_val_loss = pd.DataFrame(see_val_loss)
    see_train_loss = see_train_loss.drop(0)
    see_val_loss = see_val_loss.drop(0)

    plt.figure(0)
    # Plotting both the curves simultaneously
    plt.plot(data.time, data.velocity_x, color='r',alpha=1, label='data_eixo_x')
    plt.plot(data.time, shown.eixo_x, color='g',alpha=1, label='processed_x')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo X")
    plt.legend()
    
    plt.figure(1)
    # print(predictions)
    plt.plot(data.time, data.velocity_y, color='r', alpha=1 , label='data_eixo_y')
    plt.plot(data.time, shown.eixo_y, color='g', alpha=1 , label='processed_y')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo Y")
    plt.legend()
    
    plt.figure(2)
    plt.plot(data.time, data.velocity_z, color='r', alpha=1, label='data_eixo_z')
    plt.plot(data.time, shown.eixo_z, color='g', alpha=1 , label='processed_z')
    plt.xlabel("time")
    plt.ylabel("Velocity")
    plt.title("Comparação da velocidade provida da rede e do dataset no eixo Z")
    plt.legend()
    
    plt.figure(3)
    plt.title("Evolução do erro de treino ao longo do tempo")
    plt.plot(see_train_loss[see_train_loss.columns[0]], see_train_loss[see_train_loss.columns[1]],color='g', label='train')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    plt.figure(4)
    plt.title("Evolução do erro da validação ao longo do tempo")
    plt.plot(see_val_loss[see_val_loss.columns[0]], see_val_loss[see_val_loss.columns[1]],color='r', label='validation')
    plt.xlabel("Interação")
    plt.ylabel("Erro")
    plt.legend()

    # Mostrar os gráficos
    plt.show()


def train(data):
    # Dividir os dados em dois segmentos: treino e validação numa relação de 80% para 20%
        
    train_data = data.sample(frac=0.9, random_state=42)
    val_data = data.drop(train_data.index)

    # Criar de fato os datasets e os dataloaders para treinamento e validação com uso das classes
    train_dataset = VoltageVelocityDataset(train_data)
    val_dataset = VoltageVelocityDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)

    # Define a rede MLP e a função de perda (loss function) como error mean squared
    mlp = MLP(input_dim=input_size, output_dim=output_size, hidden_dim=hidden_size,num_hidden_layers=hidden_layers)
    criterion = nn.MSELoss()

    # Definindo o uso de CPU ou GPU para processamento da rede
    mlp = mlp.to(device)

    # Define o tipo de ativação e o learning rate
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)

    # treino da rede MLP no data set todo
    
    # ________________-_- TREINO -_-________________

    # definição dos dataframes que guardarão os dados da evolução dos erros ao longo do treino
    see_train_loss = np.empty([1, 2]).astype(float)
    see_val_loss = np.empty([1, 2]).astype(float)
    idx_train = 0
    idx_val = 0

    for epoch in range(EPOCHS):
        train_loss = 0.0
        for X, Y in train_loader:
            
            # Feedforward
            X,Y = X.to(device),Y.to(device)
            outputs = mlp(X)
            loss = criterion(outputs, Y)

            # Backpropagation e ativação dos neurônios
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            see_train_loss = np.append(
                see_train_loss, [idx_train, loss.item()]).reshape(-1, 2)
            train_loss += loss.item() * X.shape[0]
            idx_train = idx_train+1
            
        # Cálculo, por rodada, do erro da validação
        with torch.no_grad():
            val_loss = 0.0
            for X, Y in val_loader:
                X,Y = X.to(device),Y.to(device)
                outputs = mlp(X)
                loss = criterion(outputs, Y)
                see_val_loss = np.append(see_val_loss, [idx_val, loss.item()]).reshape(-1, 2)
                val_loss += loss.item() * X.shape[0]
                idx_val = idx_val+1


        # Mostrar o progresso
        if epoch % 100 == 0:
            print("| Epoch {:4} | train loss {:4.4f} | val loss {:4.4f} |".format(epoch, train_loss / len(train_dataset), val_loss / len(val_dataset)),flush=True)
    print('\n\n idx_train | idx_val: ', idx_train,' | ', idx_val)
    return mlp, see_train_loss, see_val_loss


# Evaluate the MLP on the entire dataset
def predict(mlp, data):
    with torch.no_grad():
        X = torch.tensor(
            data[[f'{input_df_name}_x', f'{input_df_name}_y', f'{input_df_name}_z']].values).float().unsqueeze(1).to(device)
        Y = torch.tensor(
            data[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']].values).float().unsqueeze(1).to(device)
        predictions = mlp(X)    
        accuracy = ((predictions - Y) ** 2).mean().sqrt().item()
        print(f"\n-> Accuracy: {accuracy:.4f}")
        return predictions, accuracy


def trained_info(data, predicted):
    output_df = pd.DataFrame(data[[f'{output_df_name}_x', f'{output_df_name}_y', f'{output_df_name}_z']])
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().detach().numpy().squeeze()

    print(predicted)
    # predicted = pd.DataFrame(predicted, columns=[ 'velocity_x',  'velocity_y' , 'velocity_z']).astype('float64')
    predicted.columns = [ 'velocity_x',  'velocity_y' , 'velocity_z']
    # print(f'\ndatatypes: predicted\n {predicted.dtypes}\n\n output_df\n {output_df.dtypes}')
    # print('\npredicted:\n',predicted)
    # print('\ndf data:\n',output_df)
    print(output_df)
    print(predicted)
    diff = (predicted-output_df)
    print('\n -> Diferença entre o esperado e o obtido do treinamento com o set de validação:\n',diff)
    diff_media = diff.abs().mean().to_numpy()[0]
    diff_max = diff.abs().max().to_numpy()[0]
    diff_min = diff.abs().min().to_numpy()[0]
    return diff_media, diff_max, diff_min

def save_model(model):
    torch.save(model.state_dict(), f'{model_local}/model_mlp_{SERIE}.pth')


def main(dataframe):
    # Começar a thread de contagem de tempo
    global stop_counting
    timeCounter_thread = threading.Thread(target=processing, args=(START_TIME,))
    timeCounter_thread.daemon= True
    timeCounter_thread.start()
    
    # Iniciar Treinamento
    print("\n\t Treinamento em execução !")
    model, train_loss, validation_loss = train(dataframe)
    stop_counting = 0
    predicted, accuracy = predict(model, dataframe)
    train_loss = pd.DataFrame(train_loss)
    validation_loss = pd.DataFrame(validation_loss)

    train_loss = train_loss.rename(columns={train_loss.columns[0]: 'time'})
    train_loss = train_loss.rename(columns={train_loss.columns[1]: 'error_train'})
    validation_loss = validation_loss.rename(columns={validation_loss.columns[0]: 'time'})
    validation_loss = validation_loss.rename(columns={validation_loss.columns[1]: 'error_val'})

    if SAVE == True:
        save_model(model)
    if EXPORT_DATA == True:
        export_data(dataframe, predicted, train_loss, validation_loss, accuracy, dir_base)
    if GRAPHS == True:
        show_graphs(dataframe, predicted, train_loss, validation_loss)
    print("\n\t Treinamento concluído !")
    

# Rodar a main
main(df_train)




