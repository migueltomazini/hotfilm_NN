"""Post-processing script to compare Neural Network predictions with original article data.

RESTORATION & DYNAMIC UPDATES APPLIED:
    - Restored original coordinate mapping: X -> E11, Y -> E22, Z -> E33.
    - Preserved Sobolev training compatibility and dynamic sampling frequency (fs).
    - Generates consolidated comparison plots with dynamic Y-axis auto-scaling.
    - Conditionally renders article/Meyers & Meneveau reference curves ONLY for series '0610'.
    - Automatically detects and plots True/Synthetic velocity if available for direct comparison.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import config

# --- Physics Constants (Matching mkplot.sh and Table 1) ---
U_MEAN = 2.21           # m/s
EPSILON = 4.88e-3       # m^2/s^3
NU = 15.16e-6           # m^2/s
ETA = (NU**3 / EPSILON)**0.25            # ~ 0.000919 m
NORM_FACTOR = (EPSILON**3 / NU)**(-0.25) 
L_INTEGRAL = 13.5       

# --- Academic Plotting Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False
})

# =============================================================================
# DATA PROCESSING & DYNAMIC FS
# =============================================================================
def calculate_real_fs(df, time_col="time", fallback_fs=2000.0):
    if time_col in df.columns:
        t = df[time_col].dropna().values
        if len(t) > 1:
            dt_median = np.median(np.diff(t))
            if dt_median > 0:
                fs_real = 1.0 / dt_median
                print(f"[Info] Frequência real de amostragem calculada: {fs_real:.4f} Hz")
                return fs_real
    print(f"[Aviso] Coluna de tempo ausente ou inválida. Usando fallback: {fallback_fs} Hz")
    return fallback_fs

def log_bin_smoothing(x: np.ndarray, y: np.ndarray, bins_per_decade: int = 40) -> tuple:
    valid = (x > 0) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) == 0: return np.array([]), np.array([])
        
    log_x = np.log10(x)
    n_bins = int(np.ceil((log_x.max() - log_x.min()) * bins_per_decade))
    edges = np.logspace(log_x.min(), log_x.max(), n_bins + 1)
    indices = np.digitize(x, edges)
    
    s_x, s_y = [], []
    for i in range(1, n_bins + 1):
        mask = (indices == i)
        if np.any(mask):
            s_x.append(x[mask].mean())
            s_y.append(y[mask].mean())
    return np.array(s_x), np.array(s_y)

def compute_nn_spectrum(signal, fs=2000.0):
    """
    Compute power spectral density (PSD) from signal using Welch method.
    """
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.array([]), np.array([])
        
    default_nperseg = int(120 * fs)
    nperseg = min(default_nperseg, max(256, len(signal) // 8))
    
    f, S_f = welch(signal - np.mean(signal), fs=fs, nperseg=nperseg, detrend='constant')
    valid = f > 0
    f, S_f = f[valid], S_f[valid]
    
    k1 = (2 * np.pi * f) / U_MEAN
    E_aa = S_f * (U_MEAN / (2 * np.pi))
    k1_sm, E_aa_sm = log_bin_smoothing(k1, E_aa)
    return k1_sm, E_aa_sm

def load_data():
    ext_dir = os.path.join(config.DATA_DIR, "external")
    ts_path = os.path.join(ext_dir, "timeseries.dat")
    spec_h_path = os.path.join(ext_dir, "spec_h_smooth.dat")
    spec_s_path = os.path.join(ext_dir, "spec_s_smooth.dat")
    models_path = os.path.join(ext_dir, "models.dat")
    
    if not os.path.exists(models_path):
        return None, None, None, None

    ts_df = pd.read_csv(ts_path, sep=r'\s+', header=None, 
                        names=['u1_h', 'u1_s', 'u2_h', 'u2_s', 'u3_h', 'u3_s'])
    ts_df = ts_df.apply(pd.to_numeric, errors='coerce') 
    
    spec_h = pd.read_csv(spec_h_path, sep=r'\s+', header=None, usecols=[0,1,2,3],
                         names=['k1', 'E11', 'E22', 'E33']).dropna()
    spec_s = pd.read_csv(spec_s_path, sep=r'\s+', header=None, usecols=[0,1,2,3],
                         names=['k1', 'E11', 'E22', 'E33']).dropna()
                         
    models_df = pd.read_csv(models_path, sep=r'\s+', header=None, usecols=[0,3,4],
                            names=['k1', 'E11', 'E_trans'])
    for col in models_df.columns:
        if models_df[col].dtype == object:
            models_df[col] = models_df[col].astype(str).str.replace(',', '.').str.replace('D', 'E', case=False)
        models_df[col] = pd.to_numeric(models_df[col], errors='coerce')
    models_df = models_df.dropna().sort_values('k1').reset_index(drop=True)
    return ts_df, spec_h, spec_s, models_df

def load_perfect_velocity(serie: str):
    """Loads the true synthetic velocity data if it exists."""
    perfect_vel_path = f"data/raw/{serie}/hotfilm_vel_{serie}.csv"
    if not os.path.exists(perfect_vel_path):
        return None
        
    print(f"[Info] Perfect/Synthetic velocity data found at {perfect_vel_path}. Loading for comparison...")
    try:
        perfect_df = pd.read_csv(perfect_vel_path)
        if "velocity_x" not in perfect_df.columns:
            perfect_df = pd.read_csv(perfect_vel_path, header=None)
            if len(perfect_df.columns) >= 4:
                perfect_df.columns = ["time", "velocity_x", "velocity_y", "velocity_z"]
            else:
                perfect_df.columns = ["velocity_x", "velocity_y", "velocity_z"]
        
        # Ensure all columns are numeric
        perfect_df = perfect_df.apply(pd.to_numeric, errors='coerce')
        return perfect_df
    except Exception as e:
        print(f"[Aviso] Failed to load perfect velocity data: {e}")
        return None

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_figure_4_timeseries(model_df: pd.DataFrame, ts_article: pd.DataFrame, perfect_df: pd.DataFrame, output_dir: str, fs: float, is_0610: bool):
    """Plot Figure 4: Time series comparison."""
    print("Generating Figure 4 (Time Series + Zoom Inset)...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate common minimum length
    min_len = len(model_df)
    if is_0610 and ts_article is not None:
        min_len = min(min_len, len(ts_article))
    if perfect_df is not None:
        min_len = min(min_len, len(perfect_df))

    time_min = np.arange(min_len) / fs / 60.0

    u_x_nn = model_df['velocity_predicted_x'].values[:min_len]
    u_y_nn = model_df['velocity_predicted_y'].values[:min_len]
    u_z_nn = model_df['velocity_predicted_z'].values[:min_len]

    # X -> u1 (Longitudinal)
    if is_0610 and ts_article is not None:
        u1_s = ts_article['u1_s'].values[:min_len]
        u1_h = ts_article['u1_h'].values[:min_len]
        ax.plot(time_min, u1_s, color='#3498db', linewidth=1.0, label='Sonic (Article)')
        ax.plot(time_min, u1_h, color='#4b4b4b', linewidth=0.8, alpha=0.8, label='Hot-film (Article)')
    
    if perfect_df is not None:
        u1_perf = perfect_df['velocity_x'].values[:min_len]
        ax.plot(time_min, u1_perf, color='#2ecc71', linewidth=1.2, alpha=0.9, label='True Velocity (Synthetic)')

    ax.plot(time_min, u_x_nn, color='#e74c3c', linestyle='--', linewidth=1.2, label='Hot-film (NN Predicted)')
    ax.text(0.4, np.nanmean(u_x_nn) + 1.2, r'$u_1$ ($\mathrm{m~s^{-1}}$)', fontsize=12)

    # Y -> u2 (Lateral)
    if is_0610 and ts_article is not None:
        u2_s = ts_article['u2_s'].values[:min_len]
        u2_h = ts_article['u2_h'].values[:min_len]
        ax.plot(time_min, u2_s, color='#3498db', linewidth=1.0)
        ax.plot(time_min, u2_h, color='#4b4b4b', linewidth=0.8, alpha=0.8)
        
    if perfect_df is not None:
        u2_perf = perfect_df['velocity_y'].values[:min_len]
        ax.plot(time_min, u2_perf, color='#2ecc71', linewidth=1.2, alpha=0.9)

    ax.plot(time_min, u_y_nn, color='#e74c3c', linestyle='--', linewidth=1.2)
    ax.text(0.4, np.nanmean(u_y_nn) + 1.2, r'$u_2$ ($\mathrm{m~s^{-1}}$)', fontsize=12)

    # Z -> u3 (Vertical)
    shift_z = -2.0
    if is_0610 and ts_article is not None:
        u3_s = ts_article['u3_s'].values[:min_len]
        u3_h = ts_article['u3_h'].values[:min_len]
        ax.plot(time_min, u3_s + shift_z, color='#3498db', linewidth=1.0)
        ax.plot(time_min, u3_h + shift_z, color='#4b4b4b', linewidth=0.8, alpha=0.8)
        
    if perfect_df is not None:
        u3_perf = perfect_df['velocity_z'].values[:min_len]
        ax.plot(time_min, u3_perf + shift_z, color='#2ecc71', linewidth=1.2, alpha=0.9)

    ax.plot(time_min, u_z_nn + shift_z, color='#e74c3c', linestyle='--', linewidth=1.2)
    ax.text(0.4, np.nanmean(u_z_nn) + shift_z + 1.2, r'$u_3$ ($\mathrm{m~s^{-1}}$) (shifted by $-2$)', fontsize=12)

    # General Formatting
    ax.plot([0.1, 2.2], [-3.8, -3.8], color='blue', linewidth=2.5)
    ax.plot([2.3, 4.9], [-3.8, -3.8], color='blue', linewidth=2.5)
    ax.set_xlabel('time (min)')
    ax.set_ylabel(r'$u_i$ ($\mathrm{m~s^{-1}}$)')
    ax.set_xlim(0, 5)
    ax.set_ylim(-4.2, 9.0)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, edgecolor='black', ncol=1)

    # INSET PLOT
    axins = inset_axes(ax, width="55%", height="25%", loc='upper center', borderpad=2)
    mask = time_min <= 0.2
    
    if is_0610 and ts_article is not None:
        axins.plot(time_min[mask], u1_s[mask], color='#3498db', linewidth=1.0)
        axins.plot(time_min[mask], u1_h[mask], color='#4b4b4b', linewidth=0.8)
        axins.plot(time_min[mask], u2_s[mask], color='#3498db', linewidth=1.0)
        axins.plot(time_min[mask], u2_h[mask], color='#4b4b4b', linewidth=0.8)
        axins.plot(time_min[mask], u3_s[mask] + shift_z, color='#3498db', linewidth=1.0)
        axins.plot(time_min[mask], u3_h[mask] + shift_z, color='#4b4b4b', linewidth=0.8)
        
    if perfect_df is not None:
        axins.plot(time_min[mask], u1_perf[mask], color='#2ecc71', linewidth=1.2)
        axins.plot(time_min[mask], u2_perf[mask], color='#2ecc71', linewidth=1.2)
        axins.plot(time_min[mask], u3_perf[mask] + shift_z, color='#2ecc71', linewidth=1.2)

    axins.plot(time_min[mask], u_x_nn[mask], color='#e74c3c', linestyle='--', linewidth=1.2)
    axins.plot(time_min[mask], u_y_nn[mask], color='#e74c3c', linestyle='--', linewidth=1.2)
    axins.plot(time_min[mask], u_z_nn[mask] + shift_z, color='#e74c3c', linestyle='--', linewidth=1.2)
    
    axins.plot([0.04, 0.2], [-3.2, -3.2], color='blue', linewidth=2.5)
    axins.set_xlim(0, 0.2)
    axins.set_ylim(-3.5, 3.5)
    axins.set_xticks([0, 0.1, 0.2])
    axins.set_xticklabels(['0', '0.1', '0.2'])
    axins.text(0.13, -1.0, 'time (min)', fontsize=10)
    axins.tick_params(axis='both', which='major', labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.5)

    plt.savefig(os.path.join(output_dir, "fig4_timeseries_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure_5_spectra(model_df: pd.DataFrame, spec_h: pd.DataFrame, spec_s: pd.DataFrame, models_df: pd.DataFrame, perfect_df: pd.DataFrame, output_dir: str, fs: float, is_0610: bool):
    print("Generating Figure 5 (Energy Spectra)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    def k1eta_to_f(x): return (x / ETA) * U_MEAN / (2 * np.pi)
    def f_to_k1eta(x): return (x * 2 * np.pi * ETA) / U_MEAN
    
    components = [
        ('velocity_predicted_x', 'velocity_x', 'E11', '(a) $E_{11}$', 'E11'),
        ('velocity_predicted_y', 'velocity_y', 'E22', '(b) $E_{22}$', 'E_trans'),
        ('velocity_predicted_z', 'velocity_z', 'E33', '(c) $E_{33}$', 'E_trans')
    ]
    
    for i, (nn_col, perf_col, art_col, title, model_col) in enumerate(components):
        ax = axes[i]
        
        if is_0610 and spec_s is not None and spec_h is not None and models_df is not None:
            mask_h = (spec_h['k1'] > 0) & (spec_h[art_col] > 0)
            mask_s = (spec_s['k1'] > 0) & (spec_s[art_col] > 0)
            mask_mm = (models_df['k1'] * ETA >= 1e-4) & (models_df[model_col] > 0)
            
            ax.loglog(spec_s.loc[mask_s, 'k1'] * ETA, spec_s.loc[mask_s, art_col], color='#3498db', linewidth=1.5, label='Sonic (Article)')
            ax.loglog(spec_h.loc[mask_h, 'k1'] * ETA, spec_h.loc[mask_h, art_col], color='#4b4b4b', linewidth=1.5, label='Hot-film (Article)')
            ax.loglog(models_df['k1'][mask_mm] * ETA, models_df[model_col][mask_mm], color='darkred', linestyle='--', linewidth=2.5, label='Meyers & Meneveau')
            
        if perfect_df is not None:
            k1_perf, E_perf = compute_nn_spectrum(perfect_df[perf_col].values, fs=fs)
            if len(k1_perf) > 0:
                ax.loglog(k1_perf * ETA, E_perf, color='#2ecc71', linestyle='-', linewidth=2.0, alpha=0.9, label='True Velocity (Synthetic)')
        
        k1_nn, E_nn = compute_nn_spectrum(model_df[nn_col].values, fs=fs)
        if len(k1_nn) > 0:
            ax.loglog(k1_nn * ETA, E_nn, color='#e74c3c', linestyle='-', linewidth=2.0, label='Hot-film (NN Predicted)')
        
        ax.set_title(title)
        ax.set_xlabel(r'$k_1\eta$')
        ax.set_xlim(1e-5, 5)
        ax.set_ylim(1e-10, 5)
        ax.grid(True, which='both', alpha=0.2)
        secax = ax.secondary_xaxis('top', functions=(k1eta_to_f, f_to_k1eta))
        secax.set_xlabel('$f$ [Hz]')
        secax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        if i == 0:
            ax.set_ylabel(r'$E_{\alpha\alpha}$')
            ax.legend(loc='lower left', frameon=False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_spectra_comparison.png"), dpi=300)
    plt.close()

def plot_figure_6_dissipation(model_df: pd.DataFrame, spec_h: pd.DataFrame, spec_s: pd.DataFrame, models_df: pd.DataFrame, perfect_df: pd.DataFrame, output_dir: str, fs: float, is_0610: bool):
    print("Generating Figure 6 (Dissipation Spectra with Auto Y-Lim)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    max_y_global = 0.0
    
    components = [
        ('velocity_predicted_x', 'velocity_x', 'E11', '(a) $E_{11}$', 'E11'),
        ('velocity_predicted_y', 'velocity_y', 'E22', '(b) $E_{22}$', 'E_trans'),
        ('velocity_predicted_z', 'velocity_z', 'E33', '(c) $E_{33}$', 'E_trans')
    ]
    
    for i, (nn_col, perf_col, art_col, title, model_col) in enumerate(components):
        ax = axes[i]
        
        # NN Prediction
        k1_nn, E_nn = compute_nn_spectrum(model_df[nn_col].values, fs=fs)
        if len(k1_nn) > 0:
            Diss_nn = NORM_FACTOR * (k1_nn**2) * E_nn
            ax.plot(k1_nn * ETA, Diss_nn, color='#e74c3c', linestyle='-', linewidth=2.0, label='Hot-film (NN Predicted)')
            max_y_global = max(max_y_global, np.max(Diss_nn))
        
        # Perfect Synthetic Data
        if perfect_df is not None:
            k1_perf, E_perf = compute_nn_spectrum(perfect_df[perf_col].values, fs=fs)
            if len(k1_perf) > 0:
                Diss_perf = NORM_FACTOR * (k1_perf**2) * E_perf
                ax.plot(k1_perf * ETA, Diss_perf, color='#2ecc71', linestyle='-', linewidth=2.0, alpha=0.9, label='True Velocity (Synthetic)')
                max_y_global = max(max_y_global, np.max(Diss_perf))
        
        # Article References
        if is_0610 and spec_h is not None and models_df is not None:
            mask_h = (spec_h['k1'] > 0) & (spec_h[art_col] > 0)
            Diss_h = NORM_FACTOR * (spec_h.loc[mask_h, 'k1']**2) * spec_h.loc[mask_h, art_col]
            ax.plot(spec_h.loc[mask_h, 'k1'] * ETA, Diss_h, color='#4b4b4b', linewidth=1.5, label='Hot-film (Article)')
            
            mask_mm = (models_df['k1'] > 0) & (models_df[model_col] > 0)
            Diss_mm = NORM_FACTOR * (models_df.loc[mask_mm, 'k1']**2) * models_df.loc[mask_mm, model_col]
            ax.plot(models_df.loc[mask_mm, 'k1'] * ETA, Diss_mm, color='darkred', linestyle='--', linewidth=2.5, label='Meyers & Meneveau')
            
            if len(Diss_h) > 0: max_y_global = max(max_y_global, np.max(Diss_h))
            if len(Diss_mm) > 0: max_y_global = max(max_y_global, np.max(Diss_mm))

        ax.set_title(title)
        ax.set_xlabel(r'$k_1\eta$')
        ax.set_xlim(0, 1.0)
        ax.grid(True, which='both', alpha=0.2)
        
        if i == 0:
            ax.set_ylabel(r'$(\epsilon^3/\nu)^{-1/4} k_1^2 E_{\alpha\alpha}$')
            ax.legend(loc='upper right', frameon=False)

    y_upper_limit = max_y_global * 1.15 if max_y_global > 0 else 2.0
    for ax in axes:
        ax.set_ylim(0, y_upper_limit)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_dissipation_spectra.png"), dpi=300)
    plt.close()

def plot_figure_7_compensated_spectra(
    model_df: pd.DataFrame, 
    spec_h: pd.DataFrame, 
    spec_s: pd.DataFrame, 
    models_df: pd.DataFrame, 
    perfect_df: pd.DataFrame,
    output_dir: str, 
    fs: float, 
    is_0610: bool
):
    """
    Plot Figure 7: Compensated spectra comparison with dynamic Y-axis.
    """
    print("Generating Figure 7 (Compensated Spectra with Auto Y-Lim)...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    max_y_global = 0.0

    components = [
        ('velocity_predicted_x', 'velocity_x', 'E11', '(a) $E_{11}$', '(d) $E_{11}$', 'E11'),
        ('velocity_predicted_y', 'velocity_y', 'E22', '(b) $E_{22}$', '(e) $E_{22}$', 'E_trans'),
        ('velocity_predicted_z', 'velocity_z', 'E33', '(c) $E_{33}$', '(f) $E_{33}$', 'E_trans')
    ]
    
    for row in range(2):
        if row == 0:
            beta = 0.0
            ck = 2.0
            a_theory = ck * 2592.0 / 8113.0
            ylabel = r'$\varepsilon^{-2/3}k_1^{5/3}E_{\alpha\alpha}$'
        else:
            beta = 0.25 / 9.0
            ck = 2.3 * L_INTEGRAL**(-beta)
            a_theory = ck * 2592.0 / 8113.0
            ylabel = r'$\varepsilon^{-2/3}k_1^{5/3+\beta}E_{\alpha\alpha}$'
            
        for col, (nn_col, perf_col, art_col, title_top, title_bot, model_col) in enumerate(components):
            ax = axes[row, col]
            title = title_top if row == 0 else title_bot
            
            # --- NN Predictions ---
            k1_nn, E_nn = compute_nn_spectrum(model_df[nn_col].values, fs=fs)
            if len(k1_nn) > 0:
                x_nn = k1_nn * ETA
                y_nn = E_nn * (EPSILON**(-2.0/3.0)) * (k1_nn**(5.0/3.0 + beta))
                ax.semilogx(x_nn, y_nn, color='#e74c3c', linestyle='-', linewidth=2.0, label='Hot-film (NN Predicted)')
                max_y_global = max(max_y_global, np.max(y_nn))
            
            # --- Perfect Synthetic Data ---
            if perfect_df is not None:
                k1_perf, E_perf = compute_nn_spectrum(perfect_df[perf_col].values, fs=fs)
                if len(k1_perf) > 0:
                    x_perf = k1_perf * ETA
                    y_perf = E_perf * (EPSILON**(-2.0/3.0)) * (k1_perf**(5.0/3.0 + beta))
                    ax.semilogx(x_perf, y_perf, color='#2ecc71', linestyle='-', linewidth=2.0, alpha=0.9, label='True Velocity (Synthetic)')
                    max_y_global = max(max_y_global, np.max(y_perf))
            
            # --- Article References (Only for 0610) ---
            if is_0610:
                if row == 0 and spec_s is not None:
                    mask_s = (spec_s['k1'] > 0) & (spec_s[art_col] > 0)
                    x_s = spec_s.loc[mask_s, 'k1'] * ETA
                    y_s = spec_s.loc[mask_s, art_col] * (EPSILON**(-2.0/3.0)) * (spec_s.loc[mask_s, 'k1']**(5.0/3.0 + beta))
                    ax.semilogx(x_s, y_s, color='#3498db', linewidth=1.5, marker='o', markersize=3, label='Sonic (Article)')
                    if len(y_s) > 0: max_y_global = max(max_y_global, np.max(y_s))
                
                if spec_h is not None:
                    mask_h = (spec_h['k1'] > 0) & (spec_h[art_col] > 0)
                    x_h = spec_h.loc[mask_h, 'k1'] * ETA
                    y_h = spec_h.loc[mask_h, art_col] * (EPSILON**(-2.0/3.0)) * (spec_h.loc[mask_h, 'k1']**(5.0/3.0 + beta))
                    ax.semilogx(x_h, y_h, color='#4b4b4b', linewidth=1.5, marker='o', markersize=3, label='Hot-film (Article)')
                    if len(y_h) > 0: max_y_global = max(max_y_global, np.max(y_h))
                
                if row == 1 and models_df is not None:
                    mask_mm = (models_df['k1'] > 0) & (models_df[model_col] > 0)
                    x_mm = models_df.loc[mask_mm, 'k1'] * ETA
                    y_mm = models_df.loc[mask_mm, model_col] * (EPSILON**(-2.0/3.0)) * (models_df.loc[mask_mm, 'k1']**(5.0/3.0 + beta))
                    ax.semilogx(x_mm, y_mm, color='darkred', linestyle='--', linewidth=2.5, label='Meyers & Meneveau')
                    if len(y_mm) > 0: max_y_global = max(max_y_global, np.max(y_mm))
            
            # Theoretical Line and Formatting
            a_val = a_theory if col == 0 else a_theory * 97.0 / 72.0
            ax.axhline(y=a_val, color='black', linestyle='--', linewidth=2.0, label='Theoretical Plateau')
            
            ax.set_title(title)
            ax.set_xlim(1e-5, 1)
            ax.grid(True, which='both', alpha=0.2)
            
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel(r'$k_1\eta$')
            if col == 2:
                ax.legend(loc='upper right', frameon=False, fontsize=9)

    y_upper_limit = max_y_global * 1.15 if max_y_global > 0 else 1.5
    for ax in axes.flatten():
        ax.set_ylim(0, y_upper_limit)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig(os.path.join(output_dir, "fig7_compensated_spectra.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    args = parser.parse_args()
    serie = args.serie

    is_0610 = "0610" in serie
    
    output_dir = os.path.join(config.DATA_DIR, "run", "results", f"velocity_{serie}", "article_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    model_csv_path = os.path.join(config.DATA_DIR, "run", "results", f"velocity_{serie}", f"velocity_{serie}.csv")
    print(f"Loading NN predicted data from {model_csv_path}...")
    model_df = pd.read_csv(model_csv_path)
    
    cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
    fallback = 2000.0
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            fallback = json.load(fh).get("FS_HOTFILM", 2000.0)
            
    fs_real = calculate_real_fs(model_df, time_col="time", fallback_fs=fallback)
    
    # Load True Synthetic Data if available
    perfect_df = load_perfect_velocity(serie)
    
    # Load Article Reference Data only if it's 0610
    ts_article, spec_h, spec_s, models_df = None, None, None, None
    if is_0610:
        print("[Info] Series 0610 detected. Loading article reference data...")
        ts_article, spec_h, spec_s, models_df = load_data()
    else:
        print("[Info] Non-0610 series detected. Omitting article reference curves.")
    
    plot_figure_4_timeseries(model_df, ts_article, perfect_df, output_dir, fs=fs_real, is_0610=is_0610)
    plot_figure_5_spectra(model_df, spec_h, spec_s, models_df, perfect_df, output_dir, fs=fs_real, is_0610=is_0610)
    plot_figure_6_dissipation(model_df, spec_h, spec_s, models_df, perfect_df, output_dir, fs=fs_real, is_0610=is_0610)
    plot_figure_7_compensated_spectra(model_df, spec_h, spec_s, models_df, perfect_df, output_dir, fs=fs_real, is_0610=is_0610)
    
    print(f"\n✅ All comparison plots (Figures 4, 5, 6, and 7) successfully saved to: {output_dir}")

if __name__ == "__main__":
    main()