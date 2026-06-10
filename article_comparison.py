"""Post-processing script to compare Neural Network predictions with original article data.

AUDIT CONFIRMATION:
    - Mathematical formulations for the Dissipation Spectra strictly match the 
      `dispec.tex` section of the original `mkplot(1).sh` script.
    - Coordinate mapping strictly adheres to the physical definitions:
      NN_Y -> Longitudinal (E11)
      NN_X -> Transversal Lateral (E22)
      NN_Z -> Transversal Vertical (E33)
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
NORM_FACTOR = (EPSILON**3 / NU)**(-0.25) # Exact match to (eps**3.0/nu)**(-1.0/4.0)

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
    signal = signal[~np.isnan(signal)]
    # Welch com janela de 120s - Replica a média estatística dos blocos
    f, S_f = welch(signal - np.mean(signal), fs=fs, nperseg=int(120*fs))
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
        raise FileNotFoundError(f"Falta o arquivo models.dat! Coloque em: {models_path}")

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

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_figure_4_timeseries(model_df: pd.DataFrame, ts_article: pd.DataFrame, output_dir: str, fs: float):
    print("Generating Figure 4 (Time Series + Zoom Inset)...")
    fig, ax = plt.subplots(figsize=(14, 10)) 
    
    window = len(ts_article) 
    time_min = np.arange(window) / fs / 60.0
    
    u2_s, u2_h = ts_article['u2_s'].values, ts_article['u2_h'].values
    u1_s, u1_h = ts_article['u1_s'].values, ts_article['u1_h'].values
    u3_s, u3_h = ts_article['u3_s'].values, ts_article['u3_h'].values
    
    u_x_nn = model_df['velocity_predicted_x'].iloc[:window].values
    u_y_nn = model_df['velocity_predicted_y'].iloc[:window].values
    u_z_nn = model_df['velocity_predicted_z'].iloc[:window].values

    # u_y_nn -> u1 (Longitudinal)
    ax.plot(time_min, u1_s, color='#3498db', linewidth=1.0, label='Sonic (Article)')
    ax.plot(time_min, u1_h, color='#4b4b4b', linewidth=0.8, alpha=0.8, label='Hot-film (Article)')
    ax.plot(time_min, u_y_nn, color='#e74c3c', linestyle='--', linewidth=1.2, label='Hot-film (NN Predicted)')
    ax.text(0.4, np.nanmean(u1_s) + 1.2, '$u_1$ ($\mathrm{m~s^{-1}}$)', fontsize=12)

    # u_x_nn -> u2 (Lateral)
    ax.plot(time_min, u2_s, color='#3498db', linewidth=1.0)
    ax.plot(time_min, u2_h, color='#4b4b4b', linewidth=0.8, alpha=0.8)
    ax.plot(time_min, u_x_nn, color='#e74c3c', linestyle='--', linewidth=1.2)
    ax.text(0.4, np.nanmean(u2_s) + 1.2, '$u_2$ ($\mathrm{m~s^{-1}}$)', fontsize=12)

    # u_z_nn -> u3 (Vertical)
    shift_z = -2.0
    ax.plot(time_min, u3_s + shift_z, color='#3498db', linewidth=1.0)
    ax.plot(time_min, u3_h + shift_z, color='#4b4b4b', linewidth=0.8, alpha=0.8)
    ax.plot(time_min, u_z_nn + shift_z, color='#e74c3c', linestyle='--', linewidth=1.2)
    ax.text(0.4, np.nanmean(u3_s) + shift_z + 1.2, '$u_3$ ($\mathrm{m~s^{-1}}$) (shifted by $-2$)', fontsize=12)

    ax.plot([0.1, 2.2], [-3.8, -3.8], color='blue', linewidth=2.5)
    ax.plot([2.3, 4.9], [-3.8, -3.8], color='blue', linewidth=2.5)
    ax.set_xlabel('time (min)')
    ax.set_ylabel('$u_i$ ($\mathrm{m~s^{-1}}$)')
    ax.set_xlim(0, 5)
    ax.set_ylim(-4.2, 9.0)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, edgecolor='black', ncol=1)

    # INSET PLOT (ZOOM)
    axins = inset_axes(ax, width="55%", height="25%", loc='upper center', borderpad=2)
    mask = time_min <= 0.2
    axins.plot(time_min[mask], u1_s[mask], color='#3498db', linewidth=1.0)
    axins.plot(time_min[mask], u1_h[mask], color='#4b4b4b', linewidth=0.8)
    axins.plot(time_min[mask], u_y_nn[mask], color='#e74c3c', linestyle='--', linewidth=1.2)
    axins.plot(time_min[mask], u2_s[mask], color='#3498db', linewidth=1.0)
    axins.plot(time_min[mask], u2_h[mask], color='#4b4b4b', linewidth=0.8)
    axins.plot(time_min[mask], u_x_nn[mask], color='#e74c3c', linestyle='--', linewidth=1.2)
    axins.plot(time_min[mask], u3_s[mask] + shift_z, color='#3498db', linewidth=1.0)
    axins.plot(time_min[mask], u3_h[mask] + shift_z, color='#4b4b4b', linewidth=0.8)
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

def plot_figure_5_spectra(model_df: pd.DataFrame, spec_h: pd.DataFrame, spec_s: pd.DataFrame, models_df: pd.DataFrame, output_dir: str, fs: float):
    print("Generating Figure 5 (Energy Spectra)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    def k1eta_to_f(x): return (x / ETA) * U_MEAN / (2 * np.pi)
    def f_to_k1eta(x): return (x * 2 * np.pi * ETA) / U_MEAN
    
    # Ordem rigorosa: Y(E11), X(E22), Z(E33)
    components = [
        ('velocity_predicted_y', 'E11', '(a) $E_{11}$', 'E11'),
        ('velocity_predicted_x', 'E22', '(b) $E_{22}$', 'E_trans'),
        ('velocity_predicted_z', 'E33', '(c) $E_{33}$', 'E_trans')
    ]
    
    for i, (nn_col, art_col, title, model_col) in enumerate(components):
        ax = axes[i]
        mask_h = (spec_h['k1'] > 0) & (spec_h[art_col] > 0)
        mask_s = (spec_s['k1'] > 0) & (spec_s[art_col] > 0)
        ax.loglog(spec_s.loc[mask_s, 'k1'] * ETA, spec_s.loc[mask_s, art_col], color='#3498db', linewidth=1.5, label='Sonic (Article)')
        ax.loglog(spec_h.loc[mask_h, 'k1'] * ETA, spec_h.loc[mask_h, art_col], color='#4b4b4b', linewidth=1.5, label='Hot-film (Article)')
        k1_nn, E_nn = compute_nn_spectrum(model_df[nn_col].values, fs=fs)
        ax.loglog(k1_nn * ETA, E_nn, color='#e74c3c', linestyle='-', linewidth=2.0, label='Hot-film (NN Predicted)')
        
        k1_eta_mm = models_df['k1'] * ETA
        mask_mm = (k1_eta_mm >= 1e-4) & (models_df[model_col] > 0)
        ax.loglog(k1_eta_mm[mask_mm], models_df[model_col][mask_mm], color='darkred', linestyle='--', linewidth=2.5, label='Meyers & Meneveau')
        
        ax.set_title(title)
        ax.set_xlabel('$k_1\eta$')
        ax.set_xlim(1e-5, 5)
        ax.set_ylim(1e-10, 5)
        ax.grid(True, which='both', alpha=0.2)
        secax = ax.secondary_xaxis('top', functions=(k1eta_to_f, f_to_k1eta))
        secax.set_xlabel('$f$ [Hz]')
        secax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        if i == 0:
            ax.set_ylabel('$E_{\\alpha\\alpha}$')
            ax.legend(loc='lower left', frameon=False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_spectra_comparison.png"), dpi=300)
    plt.close()

def plot_figure_6_dissipation(model_df: pd.DataFrame, spec_h: pd.DataFrame, spec_s: pd.DataFrame, models_df: pd.DataFrame, output_dir: str, fs: float):
    print("Generating Figure 6 (Dissipation Spectra - Single Final Plot)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    # Ordem rigorosa: Y(E11), X(E22), Z(E33)
    components = [
        ('velocity_predicted_y', 'E11', '(a) $E_{11}$', 'E11'),
        ('velocity_predicted_x', 'E22', '(b) $E_{22}$', 'E_trans'),
        ('velocity_predicted_z', 'E33', '(c) $E_{33}$', 'E_trans')
    ]
    
    for i, (nn_col, art_col, title, model_col) in enumerate(components):
        ax = axes[i]
        mask_h = (spec_h['k1'] > 0) & (spec_h[art_col] > 0)
        Diss_h = NORM_FACTOR * (spec_h.loc[mask_h, 'k1']**2) * spec_h.loc[mask_h, art_col]
        ax.plot(spec_h.loc[mask_h, 'k1'] * ETA, Diss_h, color='#4b4b4b', linewidth=1.5, label='Hot-film (Article)')
        
        k1_nn, E_nn = compute_nn_spectrum(model_df[nn_col].values, fs=fs)
        Diss_nn = NORM_FACTOR * (k1_nn**2) * E_nn
        ax.plot(k1_nn * ETA, Diss_nn, color='#e74c3c', linestyle='-', linewidth=2.0, label='Hot-film (NN Predicted)')
        
        mask_mm = (models_df['k1'] > 0) & (models_df[model_col] > 0)
        Diss_mm = NORM_FACTOR * (models_df.loc[mask_mm, 'k1']**2) * models_df.loc[mask_mm, model_col]
        ax.plot(models_df.loc[mask_mm, 'k1'] * ETA, Diss_mm, color='darkred', linestyle='--', linewidth=2.5, label='Meyers & Meneveau')
        
        ax.set_title(title)
        ax.set_xlabel('$k_1\eta$')
        ax.set_xlim(0, 1.0)
        if i == 0:
            ax.set_ylim(0, 0.5)
            ax.set_ylabel('$(\\epsilon^3/\\nu)^{-1/4} k_1^2 E_{\\alpha\\alpha}$')
            ax.legend(loc='upper right', frameon=False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_dissipation_spectra.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    args = parser.parse_args()
    serie = args.serie
    
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
    
    ts_article, spec_h, spec_s, models_df = load_data()
    
    plot_figure_4_timeseries(model_df, ts_article, output_dir, fs=fs_real)
    plot_figure_5_spectra(model_df, spec_h, spec_s, models_df, output_dir, fs=fs_real)
    plot_figure_6_dissipation(model_df, spec_h, spec_s, models_df, output_dir, fs=fs_real)
    
    print(f"\n✅ All comparison plots successfully saved to: {output_dir}")

if __name__ == "__main__":
    main()