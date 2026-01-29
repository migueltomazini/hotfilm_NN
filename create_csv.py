import pandas as pd
import sys
import os

info = '''
Check the manual inside the following folder to place the correct data to generate CSV files:
    manual/manual.txt

Usage: python3 create_csv.py <mode: train/run> <serie> <reynolds_value>
'''

if len(sys.argv) < 4:
    print(info)
    sys.exit()

mode = sys.argv[1]
serie = sys.argv[2]
re_value = float(sys.argv[3])

def train_create_CSV():
    # Load voltage data (High frequency)
    voltage_path = f"./data/train/raw/collected_data_{serie}/hotfilm_{serie}.csv"
    voltage_data = pd.read_csv(voltage_path, sep=',')
    voltage_data.columns = ['time', 'voltage_x', 'voltage_y', 'voltage_z']
    
    # Load velocity data (Reference - Sonic)
    velocity_path = f'./data/train/raw/collected_data_{serie}/sonic_{serie}.csv'
    velocity_data = pd.read_csv(velocity_path, sep=',')
    velocity_data.columns = ['time', 'velocity_x', 'velocity_y', 'velocity_z']
    
    # Merging based on time synchronization
    df_final = pd.merge(voltage_data, velocity_data, on='time')
    
    # Injecting Reynolds Number as a feature
    df_final['reynolds'] = re_value
    
    # Rounding to 12 decimals for precision consistency
    df_final = df_final.round(12)
    
    print("Final training dataframe preview:\n")
    print(df_final.head())
    
    output_path = f'./data/train/train_df_{serie}.csv'
    df_final.to_csv(output_path, index=False, float_format='%.12f')
    print(f"\nDataset ready for training. Saved as: {output_path}\n")

def run_create_CSV():
    # Load voltage data for prediction
    voltage_path = f'./data/run/raw/collected_data_{serie}/hotfilm_{serie}.csv'
    voltage_data = pd.read_csv(voltage_path, sep=',')
    voltage_data.columns = ['time', 'voltage_x', 'voltage_y', 'voltage_z']
    
    # Injecting Reynolds Number (Must match the feature used in training)
    voltage_data['reynolds'] = re_value
    
    # Rounding to 12 decimals
    voltage_data = voltage_data.round(12)
    
    output_path = f'./data/run/run_{serie}.csv'
    voltage_data.to_csv(output_path, index=False, float_format='%.12f')
    print(f"\nDataset ready for execution. Saved as: {output_path}\n")

if mode == "train":
    train_create_CSV()
elif mode == "run":
    run_create_CSV()
else:
    print(info)