import os
import numpy as np
import pandas as pd
import json

# ==============================================================================
# CONFIGURATION AND PHYSICAL CONSTANTS
# ==============================================================================
SERIE = "21180b"  # Synthetic validation series ID
N_SAMPLES = 36000 
FS = 2000       # Sampling frequency (Original high-rate)
U_MEAN = 7.66   # Mean velocity (m/s)
NU = 15.16e-6   # Kinematic viscosity of air
EPSILON = 0.009   # Target dissipation rate (m^2/s^3)
RE_NUMBER = 21180 # Example Reynolds Number to be saved as a feature

# King's Law constants (V^2 = A + B * u^0.5)
# These simulate the physical response of the hot-film
A_CONST = 1.45
B_CONST = 1.25

def meyers_meneveau_spectrum(k, epsilon, nu):
    """
    Theoretical energy spectrum E(k) based on Meyers & Meneveau (2008).
    Simplified version for synthetic generation.
    """
    C_k = 1.5  # Kolmogorov constant
    eta = (nu**3 / epsilon)**0.25
    # Inertial range with dissipation range correction
    return C_k * (epsilon**(2/3)) * (k**(-5/3)) * np.exp(-5.2 * ((k * eta)**2))

def generate_series_from_spectrum(n, fs, u_mean, epsilon, nu):
    """
    Generates a velocity time series using the Random Phase Method
    to match the desired energy spectrum.
    """
    freqs = np.fft.rfftfreq(n, d=1/fs)
    k = 2 * np.pi * freqs / u_mean
    k[0] = k[1] # Avoid division by zero
    
    # Calculate target energy at each wavenumber
    E_k = meyers_meneveau_spectrum(k, epsilon, nu)
    
    # Generate random phases
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    
    # Complex amplitudes in frequency domain
    # Energy is proportional to amplitude squared
    amplitude = np.sqrt(E_k * (fs / n) * (u_mean / (2 * np.pi)))
    z = amplitude * np.exp(1j * phases)
    
    # Return to time domain (inverse FFT)
    u_fluctuation = np.fft.irfft(z, n)
    return u_fluctuation

# ==============================================================================
# DATA GENERATION
# ==============================================================================

print(f"Generating synthetic data for Serie {SERIE}...")

# 1. Generate Velocities (u, v, w)
# Assuming different fluctuations for transverse components (v, w)
u_fluc = generate_series_from_spectrum(N_SAMPLES, FS, U_MEAN, EPSILON, NU)
v_fluc = generate_series_from_spectrum(N_SAMPLES, FS, U_MEAN, EPSILON * 0.7, NU)
w_fluc = generate_series_from_spectrum(N_SAMPLES, FS, U_MEAN, EPSILON * 0.7, NU)

u_total = U_MEAN + u_fluc
v_total = 0.0 + v_fluc
w_total = 0.0 + w_fluc

# 2. Convert Velocity to Voltage (Hot-film simulation)
# Here we apply King's Law to map Velocity -> Voltage
voltage_x = np.sqrt(A_CONST + B_CONST * np.abs(u_total)**0.5)
voltage_y = np.sqrt(A_CONST + B_CONST * np.abs(v_total + U_MEAN*0.1)**0.5) 
voltage_z = np.sqrt(A_CONST + B_CONST * np.abs(w_total + U_MEAN*0.1)**0.5)

# 3. Create DataFrames
time = np.linspace(0, N_SAMPLES/FS, N_SAMPLES)

# Training/Run format
df_main = pd.DataFrame({
    'time': time,
    'voltage_x': voltage_x,
    'voltage_y': voltage_y,
    'voltage_z': voltage_z,
    'velocity_x': u_total,
    'velocity_y': v_total,
    'velocity_z': w_total,
    'reynolds': RE_NUMBER # This is the 4th input feature we discussed
})

df_main = df_main.round(12)

# ==============================================================================
# SAVING FILES IN THE CORRECT STRUCTURE
# ==============================================================================

# Directories to ensure exist
dirs = [
    f"data/train/",
    f"data/config/",
    f"data/train/collected_data_{SERIE}/"
]

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# Save the main training file
df_main.to_csv(f"data/train/train_df_{SERIE}.csv", index=False)

# Save the "Perfect Velocity" file for spectrum.py validation
df_perfect = df_main[['time', 'velocity_x', 'velocity_y', 'velocity_z']]
df_perfect.to_csv(f"data/train/collected_data_{SERIE}/hotfilm-fake-vel_{SERIE}.csv", index=False, header=False)

# Save JSON Config for spectrum.py
config = {
    "KINEMATIC_VISCOSITY": NU,
    "FS_HOTFILM": FS,
    "FS_SONIC": 20,
    "EPSILON_EXPECTED": EPSILON,
    "THEORETICAL_MODEL": {
        "PATH": "data/theoretical_ref.dat", # Path to your MM2008 reference file
        "E11_COLUMN": 1,
        "E_TRANS_COLUMN": 2
    }
}

with open(f"data/config/config_{SERIE}.json", "w") as f:
    json.dump(config, f, indent=4)

print(f"\nSynthetic data generation complete!")
print(f"- Training data: data/train/train_df_{SERIE}.csv")
print(f"- Reference data: data/train/collected_data_{SERIE}/hotfilm-fake-vel_{SERIE}.csv")
print(f"- Config saved: data/config/config_{SERIE}.json")