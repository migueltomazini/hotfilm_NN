import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

info_output = '''

Confira o manual dentro da pasta a seguir para colocar os dados corretos para gerar o espectro:
    manuais/manual.txt
'''

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(info_output)
    sys.exit()

SERIE = sys.argv[1]
# Ler os dados a serem processados
dataframe = pd.read_csv(f'./dados/run/resultados_run/velocity_{SERIE}/velocity_{SERIE}.csv', sep=',')

# Cria-se a curva referência de ângulo de -5/3 de inclinação no plot log log
x_aux = np.linspace(1, 1000, 1000, endpoint=False)
expo = -5/3
y_aux = x_aux ** expo

def show_periodogram(fig_num, df):
    plt.figure(fig_num)
    freqs, power_spectrum = periodogram(df, fs=2000)
    
    plt.loglog(freqs, power_spectrum, label=df.name, alpha=0.85)
    plt.loglog(x_aux, y_aux, label='Linha auxiliar de inclinação (-5/3)', alpha=.9, linewidth=2.5)
    
    plt.xlabel("Frequência")
    plt.ylabel("Densidade Espectral")
    plt.title(f"Periodograma da linha temporal de {df.name}")
    plt.legend()

show_periodogram(0, dataframe['velocity_predicted_x'])
show_periodogram(1, dataframe['velocity_predicted_y'])
show_periodogram(2, dataframe['velocity_predicted_z'])

plt.show()
