"""Utility functions for loading and preprocessing data.

This module provides standardized functions for loading CSV data, handling
time synchronization, and preparing datasets for training and inference.
"""

import pandas as pd
import numpy as np
import os


def load_voltage_data(series_id: str, mode: str = 'train') -> pd.DataFrame:
    """Load voltage data from hot-film sensors.

    Args:
        series_id: Series identifier (e.g., '21180').
        mode: 'train' or 'run' to specify data path.

    Returns:
        DataFrame with voltage data.
    """
    if mode == 'train':
        path = f"./data/train/raw/collected_data_{series_id}/hotfilm_{series_id}.csv"
    else:
        path = f"./data/run/raw/collected_data_{series_id}/hotfilm_{series_id}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Voltage data file not found: {path}")

    return pd.read_csv(path, sep=',', header=None,
                       names=['time', 'voltage_x', 'voltage_y', 'voltage_z'])


def load_velocity_data(series_id: str) -> pd.DataFrame:
    """Load reference velocity data from sonic anemometer.

    Args:
        series_id: Series identifier.

    Returns:
        DataFrame with velocity data.
    """
    path = f"./data/train/raw/collected_data_{series_id}/sonic_{series_id}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Velocity data file not found: {path}")

    return pd.read_csv(path, sep=',', header=None,
                       names=['time', 'velocity_x', 'velocity_y', 'velocity_z'])


def synchronize_and_merge(voltage_df: pd.DataFrame, velocity_df: pd.DataFrame,
                         reynolds: float) -> pd.DataFrame:
    """Synchronize voltage and velocity data by time and inject Reynolds number.

    Args:
        voltage_df: DataFrame with voltage data.
        velocity_df: DataFrame with velocity data.
        reynolds: Reynolds number to inject as feature.

    Returns:
        Merged DataFrame with synchronized data.
    """
    # Merge on time (assuming time is in seconds)
    merged = pd.merge_asof(voltage_df.sort_values('time'),
                          velocity_df.sort_values('time'),
                          on='time', direction='nearest')

    # Inject Reynolds number as constant feature
    merged['reynolds'] = reynolds

    return merged.round(12)  # For numerical precision


def load_run_data(series_id: str, reynolds: float) -> pd.DataFrame:
    """Load and prepare data for inference (run mode).

    Args:
        series_id: Series identifier.
        reynolds: Reynolds number.

    Returns:
        DataFrame with voltage data and Reynolds number.
    """
    voltage_df = load_voltage_data(series_id, mode='run')
    voltage_df['reynolds'] = reynolds
    return voltage_df.round(12)