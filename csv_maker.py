import pandas as pd
import sys


info =\
'''
    ____________________-_- Criador de csv de entrada -_-____________________
        Como usar:
    train:
        $   python3 csv_maker.py train {nome da pasta em específico} {nome da saida}

        -> Exemplo:
        $   python3 csv_maker.py train data_collected_1 df_1
        
        Observação:
            A pasta em específico tem de estar dentro do diretório: 
                dados/dados_cru/dados_cru_treino
    run:
        $   python3 csv_maker.py run {nome da pasta em específico} {nome da saida}

        -> Exemplo:
        $   python3 csv_maker.py run data_coletado_x dados_2kHz_Run.csv
        
        Observação:
            A pasta em específico tem de estar dentro do diretório: 
                dados/dados_cru/dados_cru_treino

'''
if len(sys.argv) < 2:
    print(info)
    sys.exit()

serie = sys.argv[2]
def train_create_CSV():
        
    # Lê voltage data
    voltage_data = pd.read_csv(f"./dados/dados_cru/dados_cru_treino/dado_coletado_{serie}/hotfilm_{serie}.csv", sep=',')
    voltage_column_names = ['time', 'voltage_x', 'voltage_y', 'voltage_z']
    voltage_data.columns = voltage_column_names
    voltage_data.set_index('time', inplace=True)  # Set 'time' column as index
    df_20hz = voltage_data.iloc[::100]
    print(df_20hz)


    # Read velocity data
    velocity_data = pd.read_csv(f'./dados/dados_cru/dados_cru_treino/dado_coletado_{serie}/sonic_{serie}.csv',sep=',')
    velocity_column_names = ['time', 'velocity_x', 'velocity_y', 'velocity_z']
    velocity_data.columns = velocity_column_names
    velocity_data.set_index('time', inplace=True)  # Set 'time' column as index
    print(velocity_data)
    df_final = pd.merge(voltage_data,velocity_data, on='time')
    print(df_final)
    df_final.to_csv(f'./dados/treino/train_df_{serie}.csv')
    print(f"\nDataSet pronto para treino. Está salvo como:\n->\t./dados/treino/train_df_{serie}.csv\n")

def run_create_CSV():
    voltage_2kHz = pd.read_csv(f'./dados/dados_cru/dados_cru_run/dado_coletado_{serie}/hotfilm_{serie}.csv',sep=',')
    voltage_2kHz.columns=['time','voltage_x','voltage_y','voltage_z']
    voltage_2kHz.to_csv(f'./dados/run/run_{serie}.csv',index=False)
    print(f"\nDataSet pronto para uso. Está salvo como:\n->\t./dados/run/run_{serie}.csv\n")
    
if  (sys.argv[1]=="train"):
    train_create_CSV()
elif(sys.argv[1]=="run"):
    run_create_CSV()
else:
    print(info)
