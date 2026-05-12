"""Validation metrics for algorithm efficiency and turbulence physics parameters.

Implements metrics for comparing predictions with state-of-the-art references
(Kit et al., Goldshmid et al.) including parameter counts, signal derivatives,
processing ratios, and dissipation analysis.
"""

import numpy as np
import pandas as pd
from scipy import signal


def count_trainable_parameters(input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int) -> int:
    """Count total trainable parameters in MLP architecture.
    
    Args:
        input_size: Number of input features (typically 4: voltage_x, voltage_y, voltage_z, reynolds).
        hidden_size: Number of neurons in hidden layers.
        num_hidden_layers: Number of hidden layers.
        output_size: Number of output features (typically 3: velocity_x, velocity_y, velocity_z).
    
    Returns:
        Total number of trainable parameters.
    
    Formula:
        N_params = (input_size × hidden_size) + hidden_size 
                 + (hidden_size × hidden_size)^(num_hidden_layers-1) + num_hidden_layers×hidden_size
                 + (hidden_size × output_size) + output_size
    """
    # Input to first hidden layer
    params = input_size * hidden_size + hidden_size
    
    # Hidden to hidden layers
    for _ in range(num_hidden_layers - 1):
        params += hidden_size * hidden_size + hidden_size
    
    # Last hidden to output layer
    params += hidden_size * output_size + output_size
    
    return params


def calculate_velocity_derivative_skewness(velocity_array: np.ndarray, fs: float) -> dict:
    """Calculate skewness of velocity time derivatives (physical validation metric).
    
    Skewness of du/dt validates reconstruction of physical signal shape.
    For longitudinal component, expected value ~-0.3 (Kit et al., Goldshmid et al.).
    
    Args:
        velocity_array: Velocity time series array of shape (N, 3) or (N,).
        fs: Sampling frequency in Hz.
    
    Returns:
        Dictionary with skewness values for each component (or single value if 1D input).
        Expected: S_k1 ≈ -0.3 for longitudinal component (u1).
    """
    from scipy.stats import skew
    
    # Handle 1D and 2D arrays
    if velocity_array.ndim == 1:
        velocity_array = velocity_array.reshape(-1, 1)
    
    # Calculate time derivatives using finite differences
    dt = 1.0 / fs
    derivatives = np.diff(velocity_array, axis=0) / dt
    
    # Calculate skewness for each component
    skewness = {}
    comp_names = ['u_longitudinal', 'u_lateral', 'u_vertical']
    
    for i in range(min(derivatives.shape[1], 3)):
        skewness[comp_names[i]] = skew(derivatives[:, i])
    
    return skewness


def calculate_real_time_ratio(execution_time_sec: float, data_duration_sec: float) -> float:
    """Calculate Real-Time Processing Ratio (RTR).
    
    RTR measures if model can process data at or faster than collection rate.
    RTR < 1.0: Real-time capable (faster than data collection)
    RTR = 1.0: Processing at collection rate
    RTR > 1.0: Processing slower than collection (not suitable for real-time)
    
    Args:
        execution_time_sec: Total execution time in seconds.
        data_duration_sec: Real duration of collected data in seconds.
    
    Returns:
        RTR value. Should be < 1.0 for real-time capability.
    """
    if data_duration_sec <= 0:
        return float('inf')
    
    return execution_time_sec / data_duration_sec


def calculate_dissipation_series(
    velocity_pred: np.ndarray,
    velocity_true: np.ndarray,
    fs: float,
    kinematic_viscosity: float = 15.16e-6
) -> dict:
    """Calculate dissipation rate (epsilon) for predicted vs true velocities.
    
    Args:
        velocity_pred: Predicted velocities (N, 3).
        velocity_true: True velocities (N, 3).
        fs: Sampling frequency in Hz.
        kinematic_viscosity: Kinematic viscosity (m^2/s), default for air at 20°C.
    
    Returns:
        Dictionary with dissipation metrics for both true and predicted.
    """
    from scipy.signal import periodogram
    
    # Calculate fluctuations (u' = u - u_mean)
    u_pred_fluct = velocity_pred - np.mean(velocity_pred, axis=0)
    u_true_fluct = velocity_true - np.mean(velocity_true, axis=0)
    
    results = {}
    
    for comp_idx, comp_name in enumerate(['u_x', 'u_y', 'u_z']):
        # Power Spectral Density
        freqs_pred, Pxx_pred = periodogram(u_pred_fluct[:, comp_idx], fs=fs)
        freqs_true, Pxx_true = periodogram(u_true_fluct[:, comp_idx], fs=fs)
        
        # Remove zero frequency
        valid_pred = freqs_pred > 0
        valid_true = freqs_true > 0
        
        freqs_pred = freqs_pred[valid_pred]
        Pxx_pred = Pxx_pred[valid_pred]
        freqs_true = freqs_true[valid_true]
        Pxx_true = Pxx_true[valid_true]
        
        # Wavenumber from Taylor's hypothesis (k = 2*pi*f / U_mean)
        u_mean = np.mean(velocity_true[:, 0])  # Use true mean velocity
        if u_mean > 0:
            k_pred = 2 * np.pi * freqs_pred / u_mean
            k_true = 2 * np.pi * freqs_true / u_mean
        else:
            k_pred = 2 * np.pi * freqs_pred
            k_true = 2 * np.pi * freqs_true
        
        # Store spectrum data
        results[f'{comp_name}_spectrum_pred'] = (k_pred, Pxx_pred)
        results[f'{comp_name}_spectrum_true'] = (k_true, Pxx_true)
    
    return results


def generate_1to1_scatterplot_data(
    velocity_pred: np.ndarray,
    velocity_true: np.ndarray
) -> dict:
    """Prepare data for 1:1 velocity scatterplots.
    
    Args:
        velocity_pred: Predicted velocities (N, 3).
        velocity_true: True velocities (N, 3).
    
    Returns:
        Dictionary with statistics for each component.
    """
    from scipy.stats import linregress
    
    comp_names = ['u_x', 'u_y', 'u_z']
    comp_labels = ['Longitudinal (u1)', 'Lateral (u2)', 'Vertical (u3)']
    
    scatter_data = {}
    
    for i, (name, label) in enumerate(zip(comp_names, comp_labels)):
        pred = velocity_pred[:, i]
        true = velocity_true[:, i]
        
        # Linear regression for comparison line
        slope, intercept, r_value, p_value, std_err = linregress(true, pred)
        
        # RMSE and correlation
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        corr = np.corrcoef(true, pred)[0, 1]
        
        scatter_data[name] = {
            'label': label,
            'pred': pred,
            'true': true,
            'rmse': rmse,
            'r_squared': r_value ** 2,
            'correlation': corr,
            'slope': slope,
            'intercept': intercept,
            'mean_pred': np.mean(pred),
            'mean_true': np.mean(true),
            'std_pred': np.std(pred),
            'std_true': np.std(true),
        }
    
    return scatter_data


def calculate_block_dissipation_continuity(
    blocks_metrics: list
) -> dict:
    """Analyze continuity of dissipation across sequential blocks.
    
    Measures if fine-tuning maintains physical continuity without artificial jumps.
    
    Args:
        blocks_metrics: List of metrics dictionaries for each block.
    
    Returns:
        Dictionary with continuity analysis (mean jumps, max jump, smoothness).
    """
    if len(blocks_metrics) < 2:
        return {'message': 'Not enough blocks for continuity analysis'}
    
    # Extract epsilon values from each block
    epsilons = []
    for block_metric in blocks_metrics:
        if 'epsilon' in block_metric:
            epsilons.append(block_metric['epsilon'])
    
    if len(epsilons) < 2:
        return {'message': 'Not enough epsilon values available'}
    
    epsilons = np.array(epsilons)
    
    # Calculate jumps between consecutive blocks
    jumps = np.abs(np.diff(epsilons))
    
    continuity = {
        'epsilons': epsilons.tolist(),
        'mean_jump': np.mean(jumps),
        'max_jump': np.max(jumps),
        'min_jump': np.min(jumps),
        'std_jump': np.std(jumps),
        'coefficient_of_variation': np.std(jumps) / np.mean(epsilons) if np.mean(epsilons) > 0 else float('inf'),
        'num_blocks': len(epsilons),
    }
    
    return continuity


def format_validation_report(
    param_count: int,
    skewness: dict,
    rtr: float,
    scatter_stats: dict,
    continuity: dict
) -> str:
    """Format comprehensive validation report.
    
    Args:
        param_count: Number of trainable parameters.
        skewness: Dictionary of velocity derivative skewness values.
        rtr: Real-time processing ratio.
        scatter_stats: Statistics from 1:1 scatterplot comparison.
        continuity: Block dissipation continuity metrics.
    
    Returns:
        Formatted string report for display/file output.
    """
    report = "=" * 70 + "\n"
    report += "VALIDATION REPORT: Algorithm Efficiency & Turbulence Physics\n"
    report += "=" * 70 + "\n\n"
    
    # Parameter efficiency
    report += "1. ALGORITHMIC EFFICIENCY\n"
    report += "-" * 70 + "\n"
    report += f"   Trainable Parameters:         {param_count:,}\n"
    report += f"   Real-Time Processing Ratio:   {rtr:.4f}\n"
    if rtr < 1.0:
        report += f"   Status:                       ✓ REAL-TIME CAPABLE (RTR < 1.0)\n"
    else:
        report += f"   Status:                       ✗ NOT REAL-TIME (RTR > 1.0)\n"
    report += "\n"
    
    # Physical validation
    report += "2. TURBULENCE PHYSICS VALIDATION\n"
    report += "-" * 70 + "\n"
    report += "   Velocity Derivative Skewness (S_k):\n"
    for comp, sk_value in skewness.items():
        report += f"     {comp:20s}: {sk_value:7.4f}"
        if comp == 'u_longitudinal':
            expected = -0.3
            report += f"  (Expected: ~{expected}, Δ={abs(sk_value - expected):.4f})"
        report += "\n"
    report += "\n"
    
    # Velocity comparison metrics
    report += "3. VELOCITY PREDICTION ACCURACY (1:1 Comparison)\n"
    report += "-" * 70 + "\n"
    for comp_name, stats in scatter_stats.items():
        report += f"   {stats['label']}:\n"
        report += f"     RMSE:        {stats['rmse']:.6f} m/s\n"
        report += f"     R²:          {stats['r_squared']:.6f}\n"
        report += f"     Correlation: {stats['correlation']:.6f}\n"
        report += f"     Bias (Mean): Pred={stats['mean_pred']:.6f}, True={stats['mean_true']:.6f}\n"
        report += "\n"
    
    # Block continuity
    if 'num_blocks' in continuity and continuity['num_blocks'] > 1:
        report += "4. SEQUENTIAL BLOCK CONTINUITY (Fine-tuning Quality)\n"
        report += "-" * 70 + "\n"
        report += f"   Number of Blocks:        {continuity['num_blocks']}\n"
        report += f"   Mean Epsilon Jump:       {continuity['mean_jump']:.6e} m²/s³\n"
        report += f"   Max Epsilon Jump:        {continuity['max_jump']:.6e} m²/s³\n"
        report += f"   Coeff. of Variation:     {continuity['coefficient_of_variation']:.4f}\n"
        if continuity['coefficient_of_variation'] < 0.1:
            report += "   Status:                  ✓ SMOOTH TRANSITIONS BETWEEN BLOCKS\n"
        else:
            report += "   Status:                  ⚠ SIGNIFICANT JUMPS DETECTED\n"
        report += "\n"
    
    report += "=" * 70 + "\n"
    
    return report
