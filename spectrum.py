import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.integrate import trapz
import json

# ==============================================================================
# 1. CONFIGURATION AND DATA READING
# ==============================================================================

info_output = '''
Check the manual inside the following folder to place the correct data to generate the spectrum:
    manuals/manual.txt
Usage: python3 script_name.py <SERIE>
'''

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(info_output)
    sys.exit()

SERIE = sys.argv[1]

# --- USER INPUT FOR DISSIPATION CALCULATION ---
calc_epsilon_model = input("\nDo you want to calculate the Dissipation Rate (ε) from the predicted model? (y/n): ").lower() == 'y'
calc_validation_test = input("Do you want to run the Theoretical Validation Test? (y/n): ").lower() == 'y'

# Defines if any calculation requiring constants (JSON) is necessary
needs_config = calc_epsilon_model or calc_validation_test

# -------------------------------------------------------------
# Reading the JSON configuration file (Conditional)
# -------------------------------------------------------------
# Default variables if JSON is not read
KINEMATIC_VISCOSITY = 15.16e-6
FS_HOTFILM = 2000
FS_SONIC = 20
EPSILON_EXPECTED = 1.0
config = None

if needs_config:
    try:
        # The path should be adjusted to where you saved the config_SERIE.json file
        with open(f'./data/config/config_{SERIE}.json', 'r') as f:
            config = json.load(f)
        
        # Assigning configuration constants
        KINEMATIC_VISCOSITY = config['KINEMATIC_VISCOSITY']
        FS_HOTFILM = config['FS_HOTFILM']
        FS_SONIC = config['FS_SONIC']
        EPSILON_EXPECTED = config['EPSILON_EXPECTED']

    except FileNotFoundError:
        print(f"\nCritical Error: The configuration file for series '{SERIE}' was not found.")
        print("Continuing with spectrum plotting only.")
        # If JSON failed, disable all epsilon calculations
        calc_epsilon_model = False
        calc_validation_test = False
        config = None
        
    except json.JSONDecodeError:
        print("\nCritical Error: The JSON configuration file is malformed.")
        print("Continuing with spectrum plotting only.")
        calc_epsilon_model = False
        calc_validation_test = False
        config = None


# Reading velocity data (time series)
data_predicted = pd.read_csv(f'./data/run/run_results/velocity_{SERIE}/velocity_{SERIE}.csv', sep=',')
data_sonic = pd.read_csv(f'./data/train/train_df_{SERIE}.csv', sep=',')

# Definition of the exponent and the reference line (-5/3) for the log-log plot
x_aux = np.linspace(1, 1000, 1000, endpoint=False)
expo = -5/3
y_aux = x_aux ** expo

# Calculation of Mean and Fluctuations (u' = u - u_bar)
MEAN_VELOCITY = data_sonic['velocity_x'].mean() 
print(f"Longitudinal mean velocity (u1_bar) calculated from Sonic: {MEAN_VELOCITY:.3f} m/s")

# Fluctuation series (u')
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

# Global variables to store spectra (necessary for the final epsilon calculation)
E11_pred, E22_pred, E33_pred = None, None, None
k1_1_pred, k1_2_pred, k1_3_pred = None, None, None


# ==============================================================================
# 2. CALCULATION AND PROCESSING FUNCTIONS
# ==============================================================================

def log_bin_smoothing(freqs: np.ndarray, spectrum: np.ndarray, bins_per_decade: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooths the spectrum using mean in logarithmic bins.

    Args:
        freqs: Array of frequencies.
        spectrum: Array of the Power Spectral Density.
        bins_per_decade: Number of logarithmic bins per decade.

    Returns:
        A tuple containing the array of smoothed frequencies and the smoothed spectrum.
    """
    # Removes zero frequencies and zero/negative spectrum values
    valid_indices = (freqs > 0) & (spectrum > 0)
    freqs = freqs[valid_indices]
    spectrum = spectrum[valid_indices]

    if len(freqs) == 0:
        return np.array([]), np.array([])
        
    log_freqs = np.log10(freqs)
    min_log_freq = log_freqs.min()
    max_log_freq = log_freqs.max()
    
    num_bins = int(np.ceil((max_log_freq - min_log_freq) * bins_per_decade))
    
    # Creates bin edges on a logarithmic scale
    bin_edges = np.logspace(min_log_freq, max_log_freq, num_bins + 1)
    
    # Assigns each frequency to a bin
    bin_indices = np.digitize(freqs, bin_edges)
    
    smooth_freqs = []
    smooth_spectrum = []
    
    # Calculates the mean spectrum for each bin
    for i in range(1, num_bins + 1):
        indices_in_bin = (bin_indices == i)
        if np.any(indices_in_bin):
            smooth_freqs.append(freqs[indices_in_bin].mean())
            smooth_spectrum.append(spectrum[indices_in_bin].mean())
            
    return np.array(smooth_freqs), np.array(smooth_spectrum)

def calculate_dissipation_rate_theoretical(E11_k: np.ndarray, k1_1: np.ndarray, E22_k: np.ndarray, k1_2: np.ndarray, E33_k: np.ndarray, k1_3: np.ndarray, nu: float) -> float:
    """
    Calculates epsilon DIRECTLY from theoretical E(k) spectra by integration (Equation 5).

    Args:
        E11_k, E22_k, E33_k: Energy spectra of the three components.
        k1_1, k1_2, k1_3: Corresponding wavenumbers.
        nu: Kinematic viscosity.

    Returns:
        Dissipation rate (epsilon).
    """
    
    # 1. Streamwise component term (u1)
    integrand_11 = k1_1**2 * E11_k
    integral_11 = np.trapz(integrand_11, k1_1)
    term1 = 15.0 * nu * integral_11
    
    # 2. Spanwise/transversal component term (u2)
    integrand_22 = k1_2**2 * E22_k
    integral_22 = np.trapz(integrand_22, k1_2)
    term2 = (15.0 / 2.0) * nu * integral_22
    
    # 3. Vertical component term (u3)
    integrand_33 = k1_3**2 * E33_k
    integral_33 = np.trapz(integrand_33, k1_3)
    term3 = (15.0 / 2.0) * nu * integral_33
    
    # Epsilon is the average of the 3 terms (Equation 5)
    epsilon = (1.0 / 3.0) * (term1 + term2 + term3)
    
    return epsilon

def spectral_density_to_energy_spectrum(S_f: np.ndarray, freqs: np.ndarray, mean_velocity: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts PSD (S_f) to Energy Spectrum E(k1) using Taylor's Hypothesis.
    
    Args:
        S_f: Power Spectral Density.
        freqs: Frequencies.
        mean_velocity: Longitudinal mean velocity.

    Returns:
        A tuple containing the Energy Spectrum E(k1) and the Wavenumber k1.
    """
    if mean_velocity == 0:
        return np.array([]), np.array([])
        
    wavenumbers = 2 * np.pi * freqs / mean_velocity
    energy_spectrum = (mean_velocity / (2 * np.pi)) * S_f
    
    return energy_spectrum, wavenumbers

def calculate_dissipation_rate(E11: np.ndarray, k1_1: np.ndarray, E22: np.ndarray, k1_2: np.ndarray, E33: np.ndarray, k1_3: np.ndarray, nu: float) -> float:
    """
    Calculates epsilon by integrating Equation (5) from measured spectra.

    Args:
        E11, E22, E33: Energy spectra of the three components.
        k1_1, k1_2, k1_3: Corresponding wavenumbers.
        nu: Kinematic viscosity.

    Returns:
        Dissipation rate (epsilon).
    """
    
    # 1. u1 Term
    term1 = 15.0 * nu * np.trapz(k1_1**2 * E11, k1_1)
    
    # 2. u2 Term
    term2 = (15.0 / 2.0) * nu * np.trapz(k1_2**2 * E22, k1_2)
    
    # 3. u3 Term
    term3 = (15.0 / 2.0) * nu * np.trapz(k1_3**2 * E33, k1_3)
    
    epsilon = (1.0 / 3.0) * (term1 + term2 + term3)
    
    return epsilon


def process_data(fig_num: int, data_predicted: np.ndarray, data_sonic: np.ndarray, u_component: str, calculate_epsilon_flag: bool):
    """
    Main function that calculates the periodogram, plots the raw spectrum, and 
    stores the smoothed data for the epsilon calculation.

    Args:
        fig_num: Matplotlib figure number.
        data_predicted: Predicted fluctuation series.
        data_sonic: Sonic fluctuation series.
        u_component: Velocity component name.
        calculate_epsilon_flag: Flag to determine if the epsilon calculation should be executed.
    """

    # Creation of the folder where the graphs will be saved
    if not os.path.exists(f"data/run/run_results/velocity_{SERIE}/graphics/"):
        os.mkdir(f"data/run/run_results/velocity_{SERIE}/graphics/")

    plt.figure(fig_num)

    # Calculation of the periodogram using fluctuations (u' series)
    freqs_predicted, power_spectrum_predicted_raw = periodogram(data_predicted, fs=FS_HOTFILM)
    freqs_sonic, power_spectrum_sonic_raw = periodogram(data_sonic, fs=FS_SONIC)
    
    if calculate_epsilon_flag:
        # -------------------------------------------------------------
        # SMOOTHING FOR EPSILON CALCULATION (Smoothed data)
        # -------------------------------------------------------------
        # Epsilon calculation uses the smoothed spectrum for greater numerical stability.
        freqs_predicted_smooth, power_spectrum_predicted_smooth = log_bin_smoothing(freqs_predicted, power_spectrum_predicted_raw)
        
        E_predicted, k1_predicted = spectral_density_to_energy_spectrum(
            power_spectrum_predicted_smooth, freqs_predicted_smooth, MEAN_VELOCITY
        )
        
        # Stores the spectra in global variables
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
        
        # The final epsilon calculation is performed after processing the last component (u3/velocity_z)
        if u_component == 'velocity_z' and E11_pred is not None and E22_pred is not None and E33_pred is not None:
            try:
                epsilon_predicted = calculate_dissipation_rate(
                    E11_pred, k1_1_pred, 
                    E22_pred, k1_2_pred, 
                    E33_pred, k1_3_pred, 
                    KINEMATIC_VISCOSITY
                )
                print(f"\n===========================================================")
                print(f"Dissipation Rate (ε) (Predicted Hot-Film Model):")
                print(f"u1_bar used: {MEAN_VELOCITY:.3f} m/s")
                print(f"ε = {epsilon_predicted:.4e} [m^2/s^3]")
                print(f"Relative Difference: {abs(epsilon_predicted - EPSILON_EXPECTED) / EPSILON_EXPECTED * 100:.2f} %")
                print(f"===========================================================")
            except Exception as e:
                 print(f"Error in final epsilon calculation: {e}")

    # -------------------------------------------------------------
    # PLOTTING THE RAW SPECTRUM
    # -------------------------------------------------------------
    plt.loglog(freqs_predicted, power_spectrum_predicted_raw, label=f"Predicted ({u_component})", alpha=0.85)
    plt.loglog(freqs_sonic, power_spectrum_sonic_raw, label=f"Sonic ({u_component})", alpha=0.85)
    plt.loglog(x_aux, y_aux, label='Reference Slope (-5/3)', alpha=.9, linewidth=2.5)

    # Adds the smoothed spectrum to the plot if epsilon calculation was requested (for visualization)
    if calculate_epsilon_flag and 'freqs_predicted_smooth' in locals():
        plt.loglog(freqs_predicted_smooth, power_spectrum_predicted_smooth, label='Smoothed (Calculation)', alpha=0.85)

    # Adjust Y for better visualization (clears low value noise)
    plt.ylim(1e-18, 1e4)

    plt.xlabel("Frequency")
    plt.ylabel("Spectral Density")

    plt.title(f"Periodogram of {u_component} time series (Predicted and Sonic)")
    plt.legend(loc='lower left')
    plt.savefig(f"data/run/run_results/velocity_{SERIE}/graphics/Periodogram_{u_component}.png", format='png', bbox_inches='tight')

# --- Perfect Velocity Check Function ---

def check_perfect_velocity_dissipation():
    """
    Calculates the dissipation (ε) from the perfect velocity data file 
    (hotfilm-fake-vel-{SERIE}.csv) and compares it to the expected value.
    """
    print("\n--- PERFECT VELOCITY DISSIPATION CHECK ---")
    
    PERFECT_VEL_PATH = f"data/raw_data/raw_train/collected_data_{SERIE}/hotfilm-fake-vel-{SERIE}.csv"
    
    try:
        # Loads the data without using the first row as a header
        data_perfect = pd.read_csv(PERFECT_VEL_PATH, header=None) 
        
        # Defines column names manually. Assumes columns 0, 1, 2 are x, y, z velocities.
        data_perfect.columns = ['time', 'velocity_x', 'velocity_y', 'velocity_z'] 
        
    except FileNotFoundError:
        print(f"WARNING: Perfect velocity file not found at: {PERFECT_VEL_PATH}")
        return
    except Exception as e:
        print(f"ERROR reading or renaming the perfect velocity file: {e}")
        return

    # 1. Calculate Fluctuations (u' = u - u_bar)
    u_perfect_fluc = {
        'x': data_perfect['velocity_x'] - data_perfect['velocity_x'].mean(), 
        'y': data_perfect['velocity_y'] - data_perfect['velocity_y'].mean(),
        'z': data_perfect['velocity_z'] - data_perfect['velocity_z'].mean()
    }

    # 2. Calculate Raw Spectra
    freqs_x, S_x = periodogram(u_perfect_fluc['x'], fs=FS_HOTFILM)
    freqs_y, S_y = periodogram(u_perfect_fluc['y'], fs=FS_HOTFILM)
    freqs_z, S_z = periodogram(u_perfect_fluc['z'], fs=FS_HOTFILM)
    
    # 3. Smooth for Integration
    freqs_x_smooth, S_x_smooth = log_bin_smoothing(freqs_x, S_x)
    freqs_y_smooth, S_y_smooth = log_bin_smoothing(freqs_y, S_y)
    freqs_z_smooth, S_z_smooth = log_bin_smoothing(freqs_z, S_z)

    # 4. Convert PSD (S_f) to E(k1)
    E_x, k_x = spectral_density_to_energy_spectrum(S_x_smooth, freqs_x_smooth, MEAN_VELOCITY)
    E_y, k_y = spectral_density_to_energy_spectrum(S_y_smooth, freqs_y_smooth, MEAN_VELOCITY)
    E_z, k_z = spectral_density_to_energy_spectrum(S_z_smooth, freqs_z_smooth, MEAN_VELOCITY)

    # 5. Calculate Dissipation (ε)
    try:
        epsilon_perfect = calculate_dissipation_rate(E_x, k_x, E_y, k_y, E_z, k_z, KINEMATIC_VISCOSITY)
        
        print(f"ε calculated from Perfect Velocity: {epsilon_perfect:.4e} [m^2/s^3]")
        print(f"Expected Value (EPSILON_EXPECTED): {EPSILON_EXPECTED:.4e} [m^2/s^3]")
        print(f"Relative Difference: {abs(epsilon_perfect - EPSILON_EXPECTED) / EPSILON_EXPECTED * 100:.2f} %")
        
    except Exception as e:
        print(f"ERROR in dissipation calculation for Perfect Velocity: {e}")


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

# Executes processing and plots the periodogram for the 3 components
process_data(0, u_predicted_fluc['velocity_x'], u_sonic_fluc['velocity_x'], 'velocity_x', calc_epsilon_model)
process_data(1, u_predicted_fluc['velocity_y'], u_sonic_fluc['velocity_y'], 'velocity_y', calc_epsilon_model)
process_data(2, u_predicted_fluc['velocity_z'], u_sonic_fluc['velocity_z'], 'velocity_z', calc_epsilon_model)


# --- THEORETICAL VALIDATION TEST ---

if calc_validation_test and config is not None:
    
    # The theoretical validation test uses constants read from the configuration file
    try:
        # Checks dissipation using velocity data generated by the theoretical model (dat file)
        theoretical_data = pd.read_csv(
            config['THEORETICAL_MODEL']['PATH'],
            header=None,
            sep=r'\s+'
        )

        E11_COL = config['THEORETICAL_MODEL']['E11_COLUMN']
        E_TRANS_COL = config['THEORETICAL_MODEL']['E_TRANS_COLUMN']
        
        k1_theoretical = theoretical_data.iloc[:, 0].values
        E11_MM = theoretical_data.iloc[:, E11_COL].values
        E_trans_MM = theoretical_data.iloc[:, E_TRANS_COL].values
        
        # Assumes transversal isotropy for the theoretical calculation (E22 = E33 = E_trans)
        epsilon_test = calculate_dissipation_rate_theoretical(
            E11_MM, k1_theoretical, 
            E_trans_MM, k1_theoretical, 
            E_trans_MM, k1_theoretical, 
            KINEMATIC_VISCOSITY
        )
        
        epsilon_expected_test = EPSILON_EXPECTED 

        print(f"\n-----------------------------------------------------------")
        print(f"THEORETICAL VALIDATION TEST (Meyers and Meneveau Model):")
        print(f"ε calculated from theoretical spectrum: {epsilon_test:.4e} [m^2/s^3]")
        print(f"Relative Difference: {abs(epsilon_test - epsilon_expected_test) / epsilon_expected_test * 100:.2f} %")
        print(f"-----------------------------------------------------------")

    except FileNotFoundError:
        print("\nWARNING: The theoretical data file was not found. Check the path in the configuration file.")
    except Exception as e:
        print(f"\nERROR IN THEORETICAL VALIDATION TEST: {e}")
        print("Check the column format in the theoretical file and the configuration file.")

    # Executes the Perfect Velocity check
    check_perfect_velocity_dissipation()

plt.show()