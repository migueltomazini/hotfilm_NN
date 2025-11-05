import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.integrate import trapz
import json

# ==============================================================================
# 1. CONFIGURAÇÕES E LEITURA DE DADOS
# ==============================================================================

info_output = '''
Confira o manual dentro da pasta a seguir para colocar os dados corretos para gerar o espectro:
    manuals/manual.txt
Uso: python3 nome_do_script.py <SERIE>
'''

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(info_output)
    sys.exit()

SERIE = sys.argv[1]

# --- ENTRADA DO USUÁRIO PARA CÁLCULO DE DISSIPAÇÃO ---
calc_epsilon_model = input("\nVocê deseja calcular a Taxa de Dissipação (ε) a partir do modelo predito? (s/n): ").lower() == 's'
calc_validation_test = input("Você deseja realizar o Teste de Validação Teórica? (s/n): ").lower() == 's'

# Define se algum cálculo que requer constantes (JSON) é necessário
needs_config = calc_epsilon_model or calc_validation_test

# -------------------------------------------------------------
# Leitura do arquivo de configuração JSON (Condicional)
# -------------------------------------------------------------
# Variáveis default caso o JSON não seja lido
KINEMATIC_VISCOSITY = 15.16e-6  # Valor padrão do artigo
FS_HOTFILM = 2000
FS_SONIC = 20
EPSILON_EXPECTED = 1.0  # Valor placeholder
config = None

if needs_config:
    try:
        # O caminho deve ser ajustado para onde você salvou o arquivo config_SERIE.json
        with open(f'./data/config/config_{SERIE}.json', 'r') as f:
            config = json.load(f)
        
        # Atribuição das constantes de configuração
        KINEMATIC_VISCOSITY = config['KINEMATIC_VISCOSITY']
        FS_HOTFILM = config['FS_HOTFILM']
        FS_SONIC = config['FS_SONIC']
        EPSILON_EXPECTED = config['EPSILON_EXPECTED']

    except FileNotFoundError:
        print(f"\nErro Crítico: O arquivo de configuração para a série '{SERIE}' não foi encontrado.")
        print("Continuando apenas com a plotagem do periodograma.")
        # Se o JSON falhou, desabilitamos todos os cálculos de epsilon
        calc_epsilon_model = False
        calc_validation_test = False
        config = None
        
    except json.JSONDecodeError:
        print("\nErro Crítico: O arquivo de configuração JSON está mal formatado.")
        print("Continuando apenas com a plotagem do periodograma.")
        calc_epsilon_model = False
        calc_validation_test = False
        config = None


# Leitura dos dados de velocidade (série temporal)
data_predicted = pd.read_csv(f'./data/run/run_results/velocity_{SERIE}/velocity_{SERIE}.csv', sep=',')
data_sonic = pd.read_csv(f'./data/train/train_df_{SERIE}.csv', sep=',')

# Definição do expoente e da linha de referência (-5/3) para o plot log-log
x_aux = np.linspace(1, 1000, 1000, endpoint=False)
expo = -5/3
y_aux = x_aux ** expo

# Cálculo da Média e das Flutuações (u' = u - u_bar)
MEAN_VELOCITY = data_sonic['velocity_x'].mean() 
print(f"Velocidade média longitudinal (u1_bar) calculada do Sônico: {MEAN_VELOCITY:.3f} m/s")

# Séries de flutuações (u')
u_predicted_fluc = {
    'velocity_x': data_predicted['velocity_predicted_x'] - data_predicted['velocity_predicted_x'].mean(),
    'velocity_y': data_predicted['velocity_predicted_y'] - data_predicted['velocity_predicted_y'].mean(),
    'velocity_z': data_predicted['velocity_predicted_z'] - data_predicted['velocity_predicted_z'].mean()
}
u_sonic_fluc = {
    'velocity_x': data_sonic['velocity_x'] - data_sonic['velocity_x'].mean(),
    'velocity_y': data_sonic['velocity_y'] - data_sonic['velocity_y'].mean(),
    'velocity_z': data_sonic['velocity_z'] - data_sonic['velocity_z'].mean()
}

# Variáveis globais para armazenar os espectros (necessário para o cálculo final de epsilon)
E11_pred, E22_pred, E33_pred = None, None, None
k1_1_pred, k1_2_pred, k1_3_pred = None, None, None


# ==============================================================================
# 2. FUNÇÕES DE CÁLCULO E PROCESSAMENTO
# ==============================================================================

def log_bin_smoothing(freqs, spectrum, bins_per_decade=30):
    """
    Suaviza o espectro usando média em bins logarítmicos.
    """
    # Remove frequências zero e valores de espectro zero/negativos
    valid_indices = (freqs > 0) & (spectrum > 0)
    freqs = freqs[valid_indices]
    spectrum = spectrum[valid_indices]

    if len(freqs) == 0:
        return np.array([]), np.array([])
        
    log_freqs = np.log10(freqs)
    min_log_freq = log_freqs.min()
    max_log_freq = log_freqs.max()
    
    num_bins = int(np.ceil((max_log_freq - min_log_freq) * bins_per_decade))
    
    # Cria os limites dos bins em escala logarítmica
    bin_edges = np.logspace(min_log_freq, max_log_freq, num_bins + 1)
    
    # Atribui cada frequência a um bin
    bin_indices = np.digitize(freqs, bin_edges)
    
    smooth_freqs = []
    smooth_spectrum = []
    
    # Calcula a média do espectro para cada bin
    for i in range(1, num_bins + 1):
        indices_in_bin = (bin_indices == i)
        if np.any(indices_in_bin):
            smooth_freqs.append(freqs[indices_in_bin].mean())
            smooth_spectrum.append(spectrum[indices_in_bin].mean())
            
    return np.array(smooth_freqs), np.array(smooth_spectrum)

def calculate_dissipation_rate_theoretical(E11_k, k1_1, E22_k, k1_2, E33_k, k1_3, nu):
    """
    Calcula epsilon DIRETAMENTE de espectros teóricos E(k).
    """
    
    # 1. Termo da componente streamwise (u1)
    integrand_11 = k1_1**2 * E11_k
    integral_11 = np.trapz(integrand_11, k1_1)
    term1 = 15.0 * nu * integral_11
    
    # 2. Termo da componente transversal/spanwise (u2)
    integrand_22 = k1_2**2 * E22_k
    integral_22 = np.trapz(integrand_22, k1_2)
    term2 = (15.0 / 2.0) * nu * integral_22
    
    # 3. Termo da componente vertical (u3)
    integrand_33 = k1_3**2 * E33_k
    integral_33 = np.trapz(integrand_33, k1_3)
    term3 = (15.0 / 2.0) * nu * integral_33
    
    # Epsilon é a média dos 3 termos (Equação 5)
    epsilon = (1.0 / 3.0) * (term1 + term2 + term3)
    
    return epsilon

def spectral_density_to_energy_spectrum(S_f, freqs, mean_velocity):
    """Converte DEP (S_f) para Espectro de Energia E(k1) usando a Hipótese de Taylor."""
    if mean_velocity == 0:
        return np.array([]), np.array([])
        
    wavenumbers = 2 * np.pi * freqs / mean_velocity
    energy_spectrum = (mean_velocity / (2 * np.pi)) * S_f
    
    return energy_spectrum, wavenumbers

def calculate_dissipation_rate(E11, k1_1, E22, k1_2, E33, k1_3, nu):
    """Calcula epsilon pela integração (Equação 5)."""
    
    # 1. Termo u1
    term1 = 15.0 * nu * np.trapz(k1_1**2 * E11, k1_1)
    
    # 2. Termo u2
    term2 = (15.0 / 2.0) * nu * np.trapz(k1_2**2 * E22, k1_2)
    
    # 3. Termo u3
    term3 = (15.0 / 2.0) * nu * np.trapz(k1_3**2 * E33, k1_3)
    
    epsilon = (1.0 / 3.0) * (term1 + term2 + term3)
    
    return epsilon


def process_data(fig_num, data_predicted, data_sonic, u_component, calculate_epsilon_flag):
    """
    Calcula o periodograma (bruto e suavizado), plota o espectro bruto e 
    armazena os dados suavizados para o cálculo de epsilon.
    """

    # Criação da pasta onde os gráficos serão salvos
    if not os.path.exists(f"data/run/run_results/velocity_{SERIE}/graphics/"):
        os.mkdir(f"data/run/run_results/velocity_{SERIE}/graphics/")

    plt.figure(fig_num)

    # Cálculo do periodograma usando flutuações (série u')
    freqs_predicted, power_spectrum_predicted_raw = periodogram(data_predicted, fs=FS_HOTFILM)
    freqs_sonic, power_spectrum_sonic_raw = periodogram(data_sonic, fs=FS_SONIC)
    
    # -------------------------------------------------------------
    # PLOTAGEM DO ESPECTRO BRUTO
    # -------------------------------------------------------------
    
    # O periodograma é impresso com os dados não suavizados
    plt.loglog(freqs_predicted, power_spectrum_predicted_raw, label=data_predicted.name, alpha=0.85)
    plt.loglog(freqs_sonic, power_spectrum_sonic_raw, label=data_sonic.name, alpha=0.85)
    plt.loglog(x_aux, y_aux, label='Linha auxiliar de inclinação (-5/3)', alpha=.9, linewidth=2.5)

    # Ajuste de Y para melhor visualização (limpa ruído de baixo valor)
    plt.ylim(1e-18, 1e4)

    plt.xlabel("Frequência")
    plt.ylabel("Densidade Espectral")

    plt.title(f"Periodograma Bruto da linha temporal de {data_sonic.name} prevista e sônica (real)")
    plt.savefig(f"data/run/run_results/velocity_{SERIE}/graphics/Periodogram_Raw_{data_sonic.name}.png", format='png')
    plt.legend()
    
    if calculate_epsilon_flag:
        # -------------------------------------------------------------
        # SUAVIZAÇÃO PARA O CÁLCULO DE EPSILON (Dados suavizados)
        # -------------------------------------------------------------
        # O cálculo de epsilon utiliza o espectro suavizado para maior estabilidade numérica.
        freqs_predicted_smooth, power_spectrum_predicted_smooth = log_bin_smoothing(freqs_predicted, power_spectrum_predicted_raw)
        
        E_predicted, k1_predicted = spectral_density_to_energy_spectrum(
            power_spectrum_predicted_smooth, freqs_predicted_smooth, MEAN_VELOCITY
        )
        
        # Armazena os espectros nas variáveis globais
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
        
        # O cálculo final de epsilon é realizado após o processamento da última componente (u3/velocity_z)
        if u_component == 'velocity_z' and E11_pred is not None and E22_pred is not None and E33_pred is not None:
            try:
                epsilon_predicted = calculate_dissipation_rate(
                    E11_pred, k1_1_pred, 
                    E22_pred, k1_2_pred, 
                    E33_pred, k1_3_pred, 
                    KINEMATIC_VISCOSITY
                )
                print(f"\n===========================================================")
                print(f"Taxa de Dissipação de Energia (ε) (Modelo Hot-Film Predito):")
                print(f"u1_bar utilizada: {MEAN_VELOCITY:.3f} m/s")
                print(f"ε = {epsilon_predicted:.4e} [m^2/s^3]")
                print(f"Diferença relativa: {abs(epsilon_predicted - EPSILON_EXPECTED) / EPSILON_EXPECTED * 100:.2f} %")
                print(f"===========================================================")
            except Exception as e:
                 print(f"Erro no cálculo final de epsilon: {e}")


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================

# Executa o processamento e plota o periodograma para as 3 componentes
process_data(0, u_predicted_fluc['velocity_x'], u_sonic_fluc['velocity_x'], 'velocity_x', calc_epsilon_model)
process_data(1, u_predicted_fluc['velocity_y'], u_sonic_fluc['velocity_y'], 'velocity_y', calc_epsilon_model)
process_data(2, u_predicted_fluc['velocity_z'], u_sonic_fluc['velocity_z'], 'velocity_z', calc_epsilon_model)


# --- TESTE DE VALIDAÇÃO TEÓRICA ---

if calc_validation_test and config is not None:
    
    # O teste de validação utiliza constantes lidas do arquivo de configuração
    try:
        theoretical_data = pd.read_csv(
            config['THEORETICAL_MODEL']['PATH'],
            header=None,
            sep=r'\s+'
        )

        # Colunas são definidas no arquivo de configuração
        E11_COL = config['THEORETICAL_MODEL']['E11_COLUMN']
        E_TRANS_COL = config['THEORETICAL_MODEL']['E_TRANS_COLUMN']
        
        k1_theoretical = theoretical_data.iloc[:, 0].values
        E11_MM = theoretical_data.iloc[:, E11_COL].values
        E_trans_MM = theoretical_data.iloc[:, E_TRANS_COL].values
        
        # Para o teste, assume-se isotropia transversal (E22 = E33 = E_trans)
        epsilon_test = calculate_dissipation_rate_theoretical(
            E11_MM, k1_theoretical, 
            E_trans_MM, k1_theoretical, 
            E_trans_MM, k1_theoretical, # Usamos E_trans para E33
            KINEMATIC_VISCOSITY
        )
        
        epsilon_expected_test = EPSILON_EXPECTED 

        print(f"\n-----------------------------------------------------------")
        print(f"VALOR DE REFERÊNCIA ESPERADO (do modelo teórico): ε = {epsilon_expected_test:.4e} [m^2/s^3]")
        print(f"-----------------------------------------------------------")
        print(f"TESTE DE VALIDAÇÃO (Modelo Teórico de Meyers e Meneveau):")
        print(f"ε calculado do espectro teórico: {epsilon_test:.4e} [m^2/s^3]")
        print(f"Diferença relativa: {abs(epsilon_test - epsilon_expected_test) / epsilon_expected_test * 100:.2f} %")
        print(f"-----------------------------------------------------------")

    except FileNotFoundError:
        print("\nAVISO: O arquivo de dados teóricos não foi encontrado. Verifique o caminho no arquivo de configuração.")
    except Exception as e:
        print(f"\nERRO NO TESTE DE VALIDAÇÃO: {e}")
        print("Verifique o formato das colunas no arquivo teórico e no arquivo de configuração.")

plt.show()
