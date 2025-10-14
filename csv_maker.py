import pandas as pd
import sys



info = '''

Confira o manual dentro da pasta a seguir para colocar os dados corretos para gerar os arquivos CSV:
    manual/manual.txt
'''

if len(sys.argv) < 2 or len(sys.argv)>3:
    print(info)
    sys.exit()

serie = sys.argv[2]
def train_create_CSV():
        
    # Lê voltage data
    voltage_data = pd.read_csv(f"./data/raw_data/raw_train/collected_data_{serie}/hotfilm_{serie}.csv", sep=',')
    voltage_column_names = ['time', 'voltage_x', 'voltage_y', 'voltage_z']
    voltage_data.columns = voltage_column_names
    voltage_data.set_index('time', inplace=True)  # Set 'time' column as index
    df_20hz = voltage_data.iloc[::100]


    # Read velocity data
    velocity_data = pd.read_csv(f'./data/raw_data/raw_train/collected_data_{serie}/sonic_{serie}.csv',sep=',')
    velocity_column_names = ['time', 'velocity_x', 'velocity_y', 'velocity_z']
    velocity_data.columns = velocity_column_names
    velocity_data.set_index('time', inplace=True)  # Set 'time' column as index
    df_final = pd.merge(voltage_data,velocity_data, on='time')
    print("Dataframe de treino final:\n")
    print(df_final)
    df_final.to_csv(f'./data/train/train_df_{serie}.csv')
    print(f"\nDataSet pronto para treino. Está salvo como:\n->\t./data/train/train_df_{serie}.csv\n")

def run_create_CSV():
    voltage_2kHz = pd.read_csv(f'./data/raw_data/raw_run/collected_data_{serie}/hotfilm_{serie}.csv',sep=',')
    voltage_2kHz.columns=['time','voltage_x','voltage_y','voltage_z']
    voltage_2kHz.to_csv(f'./data/run/run_{serie}.csv',index=False)
    print(f"\nDataSet pronto para uso. Está salvo como:\n->\t./data/run/run_{serie}.csv\n")
    
if  (sys.argv[1]=="train"):
    train_create_CSV()
elif(sys.argv[1]=="run"):
    run_create_CSV()
else:
    print(info)
