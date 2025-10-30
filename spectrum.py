import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

info_output = '''
Confira o manual dentro da pasta a seguir para colocar os dados corretos para gerar o espectro:
    manuals/manual.txt
'''

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(info_output)
    sys.exit()

SERIE = sys.argv[1]

# Ler os dados a serem processados
data_predicted = pd.read_csv(f'./data/run/run_results/velocity_{SERIE}/velocity_{SERIE}.csv', sep=',')
data_sonic = pd.read_csv(f'./data/train/train_df_{SERIE}.csv', sep=',')

# Cria-se a curva referência de ângulo de -5/3 de inclinação no plot log log
x_aux = np.linspace(1, 1000, 1000, endpoint=False)
expo = -5/3
y_aux = x_aux ** expo

# Constante de Viscosidade Cinemática (ν)
KINEMATIC_VISCOSITY = 15.16e-6  # m^2/s

# Média da velocidade longitudinal - streamwise (u1_bar) m/s.
MEAN_VELOCITY = data_sonic['velocity_x'].mean()

def spectral_density_to_energy_spectrum(S_f, freqs, mean_velocity):
    """
    Converte a Densidade Espectral de Potência (S_f) para o Espectro de Energia (E_k1)
    usando a Hipótese de Taylor.

    Args:
        S_f (np.array): Densidade Espectral de Potência (DEP) S(f).
        freqs (np.array): Frequências (f) correspondentes.
        mean_velocity (float): Média da velocidade longitudinal (u_bar) [m/s].

    Returns:
        np.array: Espectro de Energia Unidimensional E_alpha_alpha(k_1).
        np.array: Número de onda k_1 [rad/m].
    """
    if mean_velocity == 0:
        return np.zeros_like(S_f), np.zeros_like(freqs)
        
    # k_1 = 2*pi*f / u_bar
    wavenumbers = 2 * np.pi * freqs / mean_velocity
    # E_alpha_alpha(k_1) = (u_bar / (2*pi)) * S_alpha_alpha(f)
    energy_spectrum = (mean_velocity / (2 * np.pi)) * S_f
    
    return energy_spectrum, wavenumbers

def calculate_dissipation_rate(E11, k1_1, E22, k1_2, E33, k1_3, nu, mean_velocity):
    """
    Calcula a taxa de dissipação de energia cinética da turbulência (epsilon)
    pela integração dos espectros de dissipação.

    A integral é calculada numericamente usando a regra do trapézio (np.trapz).

    Args:
        E11 (np.array): Espectro de Energia E11 (streamwise).
        k1_1 (np.array): Números de onda k1 correspondentes a E11.
        E22 (np.array): Espectro de Energia E22 (spanwise).
        k1_2 (np.array): Números de onda k1 correspondentes a E22.
        E33 (np.array): Espectro de Energia E33 (vertical).
        k1_3 (np.array): Números de onda k1 correspondentes a E33.
        nu (float): Viscosidade cinemática [m^2/s].
        mean_velocity (float): Média da velocidade longitudinal u_bar [m/s].
        
    Returns:
        float: Taxa de dissipação de energia (epsilon) [m^2/s^3].
    """
    
    # É necessário que o vetor de número de onda k1 esteja ordenado para o trapézio
    # Funciona para E11, E22, E33 (velocidade longitudinal, transversal e vertical)
    
    # 1. Termo da componente streamwise (u1)
    integrand_11 = k1_1**2 * E11
    integral_11 = np.trapz(integrand_11, k1_1)
    term1 = 15.0 * nu * integral_11
    
    # 2. Termo da componente spanwise (u2)
    integrand_22 = k1_2**2 * E22
    integral_22 = np.trapz(integrand_22, k1_2)
    term2 = (15.0 / 2.0) * nu * integral_22
    
    # 3. Termo da componente vertical (u3)
    integrand_33 = k1_3**2 * E33
    integral_33 = np.trapz(integrand_33, k1_3)
    term3 = (15.0 / 2.0) * nu * integral_33
    
    # Epsilon (taxa de dissipação) é a média dos 3 termos
    epsilon = (1.0 / 3.0) * (term1 + term2 + term3)
    
    return epsilon

def show_periodogram_and_dissipation(fig_num, data_predicted, data_sonic, u_component):
    """
    Calcula, plota o periodograma e calcula a Dissipação de Energia para a componente longitudinal (u1).
    (Modificada para calcular e imprimir a taxa de dissipação)
    """

    # Criação da pasta na qual os gráficos serão salvos (run)
    if not os.path.exists(f"data/run/run_results/velocity_{SERIE}/graphics/"):
        os.mkdir(f"data/run/run_results/velocity_{SERIE}/graphics/")

    plt.figure(fig_num)

    # Cálculo do períodograma de ambas as séries
    # Nota: Usamos fs=2000 para hot-film (predicted) e fs=20 para sonic (real).
    freqs_predicted, power_spectrum_predicted = periodogram(data_predicted, fs=2000)
    freqs_sonic, power_spectrum_sonic = periodogram(data_sonic, fs=20)
    
    plt.loglog(freqs_predicted, power_spectrum_predicted, label=data_predicted.name, alpha=0.85)
    plt.loglog(freqs_sonic, power_spectrum_sonic, label=data_sonic.name, alpha=0.85)
    plt.loglog(x_aux, y_aux, label='Linha auxiliar de inclinação (-5/3)', alpha=.9, linewidth=2.5)

    plt.ylim(1e-18, 1e3)

    plt.xlabel("Frequência")
    plt.ylabel("Densidade Espectral")

    plt.title(f"Periodograma da linha temporal de {data_sonic.name} prevista e sônica (real)")
    plt.savefig(f"data/run/run_results/velocity_{SERIE}/graphics/Periodogram {data_sonic.name}.png", format='png')
    plt.legend()
    
    # ==========================================================================
    # CÁLCULO E EXIBIÇÃO DA TAXA DE DISSIPAÇÃO (ε)
    # 
    # Para o cálculo de epsilon, precisamos dos espectros E11, E22 e E33.
    # Assumimos que a ordem das chamadas é u1 (x), u2 (y), u3 (z).
    # ==========================================================================
    
    # 1. Converter DEP (S(f)) para Espectro de Energia (E(k1))
    E_predicted, k1_predicted = spectral_density_to_energy_spectrum(
        power_spectrum_predicted, freqs_predicted, MEAN_VELOCITY
    )
    
    # Guardar os espectros no escopo global (ou passar a função principal)
    # para usar na fórmula final de epsilon.
    global E11_pred, E22_pred, E33_pred, k1_1_pred, k1_2_pred, k1_3_pred
    
    if u_component == 'velocity_x':
        E11_pred = E_predicted
        k1_1_pred = k1_predicted
    elif u_component == 'velocity_y':
        E22_pred = E_predicted
        k1_2_pred = k1_predicted
    elif u_component == 'velocity_z':
        E33_pred = E_predicted
        k1_3_pred = k1_predicted
    
    # O cálculo final de epsilon deve ser feito após as 3 chamadas com as 3 componentes.
    if u_component == 'velocity_z' and E11_pred is not None and E22_pred is not None:
        try:
            epsilon_predicted = calculate_dissipation_rate(
                E11_pred, k1_1_pred, 
                E22_pred, k1_2_pred, 
                E33_pred, k1_3_pred, 
                KINEMATIC_VISCOSITY, MEAN_VELOCITY
            )
            print(f"\n===========================================================")
            print(f"Taxa de Dissipação de Energia (ε) (Modelo):")
            print(f"ε = {epsilon_predicted:.4e} [m^2/s^3]")
            print(f"===========================================================")
        except NameError:
             print("Aviso: As 3 componentes do espectro não foram calculadas/definidas. ε não calculado.")

# Inicializar variáveis globais para armazenar os espectros necessários para epsilon
E11_pred, E22_pred, E33_pred = None, None, None
k1_1_pred, k1_2_pred, k1_3_pred = None, None, None

show_periodogram_and_dissipation(0, data_predicted['velocity_predicted_x'], data_sonic['velocity_x'], 'velocity_x')
show_periodogram_and_dissipation(1, data_predicted['velocity_predicted_y'], data_sonic['velocity_y'], 'velocity_y')
show_periodogram_and_dissipation(2, data_predicted['velocity_predicted_z'], data_sonic['velocity_z'], 'velocity_z')

plt.show()
