import os
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
data_predicted = pd.read_csv(f'./dados/run/resultados_run/velocity_{SERIE}/velocity_{SERIE}.csv', sep=',')
data_sonic = pd.read_csv(f'./dados/treino/train_df_{SERIE}.csv', sep=',')

# Cria-se a curva referência de ângulo de -5/3 de inclinação no plot log log
x_aux = np.linspace(1, 1000, 1000, endpoint=False)
expo = -5/3
y_aux = x_aux ** expo

def show_periodogram(fig_num, data_predicted, data_sonic):
    """
    Calcula e plota o periodograma (Densidade Espectral de Potência)
    para séries temporais de velocidade prevista e sônica (real).
    Os resultados são plotados em escala log-log para melhor visualização.

    Args:
        fig_num (int): Número da figura a ser criada (usado como identificador de plot).
        data_predicted (pd.Series): Série temporal de velocidade predita.
        data_sonic (pd.Series): Série temporal de velocidade sônica (real).
    """

    # Criação da pasta na qual os gráficos serão salvos (run)
    if not os.path.exists(f"dados/run/resultados_run/velocity_{SERIE}/graphics/"):
        os.mkdir(f"dados/run/resultados_run/velocity_{SERIE}/graphics/")

    plt.figure(fig_num)

    # Cálculo do períodograma de ambas as séries considerando a frequência de 20 e 2000 Hz, respectivamente.
    freqs_predicted, power_spectrum_predicted = periodogram(data_predicted, fs=2000)
    freqs_sonic, power_spectrum_sonic = periodogram(data_sonic, fs=20)
    
    plt.loglog(freqs_predicted, power_spectrum_predicted, label=data_predicted.name, alpha=0.85)
    plt.loglog(freqs_sonic, power_spectrum_sonic, label=data_sonic.name, alpha=0.85)
    plt.loglog(x_aux, y_aux, label='Linha auxiliar de inclinação (-5/3)', alpha=.9, linewidth=2.5)

    plt.xlabel("Frequência")
    plt.ylabel("Densidade Espectral")

    plt.title(f"Periodograma da linha temporal de {data_sonic.name} prevista e sônica (real)")
    plt.savefig(f"dados/run/resultados_run/velocity_{SERIE}/graphics/Períodograma {data_sonic.name}.png", format='png')
    plt.legend()

show_periodogram(0, data_predicted['velocity_predicted_x'], data_sonic['velocity_x'])
show_periodogram(1, data_predicted['velocity_predicted_y'], data_sonic['velocity_y'])
show_periodogram(2, data_predicted['velocity_predicted_z'], data_sonic['velocity_z'])

plt.show()
