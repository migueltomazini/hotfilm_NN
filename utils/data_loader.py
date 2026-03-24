"""Utility functions for loading and preprocessing data.

This module provides standardized functions for loading CSV data, handling
time synchronization, and preparing datasets for training and inference.
"""

import pandas as pd
import numpy as np
import os


def _find_raw_file(series_id: str, prefix: str) -> str:
    """Helper: locate a CSV or DAT file for given prefix.

    Searches for `<prefix>_{series_id}.csv` first, then `.dat`. Raises if none
    found.
    """
    base_dir = f"./data/raw/{series_id}"
    for ext in ('csv', 'dat'):
        candidate = os.path.join(base_dir, f"{prefix}_{series_id}.{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Raw data file not found for {series_id} prefix {prefix}")


def _read_raw_file(path: str, columns: list[str]) -> pd.DataFrame:
    """Read a file which may be CSV or whitespace-delimited DAT.

    Args:
        path: full path to file.
        columns: list of column names to assign (no header assumed).
    """
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, sep=',', header=None, names=columns)
    else:
        # DAT file: use any whitespace delimiter, ignore comment lines
        return pd.read_csv(path, delim_whitespace=True, comment='#', header=None, names=columns)


def load_voltage_data(series_id: str) -> pd.DataFrame:
    """Load voltage data from hot-film sensors.

    The function now accepts either CSV or DAT located under
    `data/raw/{series_id}`.
    """
    path = _find_raw_file(series_id, 'hotfilm')
    return _read_raw_file(path, ['time', 'voltage_x', 'voltage_y', 'voltage_z'])


def load_velocity_data(series_id: str) -> pd.DataFrame:
    """Load reference velocity data from sonic anemometer.

    Supports CSV or DAT files located under `data/raw/{series_id}`.
    """
    path = _find_raw_file(series_id, 'sonic')
    return _read_raw_file(path, ['time', 'velocity_x', 'velocity_y', 'velocity_z'])


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
    voltage_df = load_voltage_data(series_id)
    voltage_df['reynolds'] = reynolds
    return voltage_df.round(12)

def split_dataframe_by_gap(df: pd.DataFrame, time_col: str = "time", gap_threshold: float = 0.1) -> list[pd.DataFrame]:
    """Split a dataframe whenever the interval between successive timestamps
    exceeds ``gap_threshold`` seconds. Useful for real data containing gaps.

    Args:
        df: input dataframe sorted by ``time_col``.
        time_col: name of the time column.
        gap_threshold: threshold in seconds to consider a break.

    Returns:
        List of DataFrames, each representing a contiguous block.
    """
    if df.empty:
        return []
    dt = df[time_col].diff().fillna(0)
    splits = np.where(dt > gap_threshold)[0]
    if splits.size == 0:
        return [df]
    blocks = []
    start = 0
    for idx in splits:
        blocks.append(df.iloc[start:idx].reset_index(drop=True))
        start = idx
    blocks.append(df.iloc[start:].reset_index(drop=True))
    return blocks


def split_dataframe_fixed_size(df: pd.DataFrame, block_size: int) -> list[pd.DataFrame]:
    """Divide ``df`` into sequential blocks of ``block_size`` rows. Last block may
    be shorter.
    """
    if block_size <= 0:
        return [df]
    return [df.iloc[i:i+block_size].reset_index(drop=True)
            for i in range(0, len(df), block_size)]


def prepare_blocks(df: pd.DataFrame, block_size: int | None = None, gap_threshold: float | None = None) -> list[pd.DataFrame]:
    """Helper that chains the two splitting strategies.

    The order is gap-based splitting first, then fixed-size splitting.  Blocks
    smaller than one row are dropped.
    """
    blocks = [df]
    if gap_threshold is not None:
        new_blocks = []
        for b in blocks:
            new_blocks.extend(split_dataframe_by_gap(b, time_col='time', gap_threshold=gap_threshold))
        blocks = new_blocks
    if block_size is not None:
        new_blocks = []
        for b in blocks:
            new_blocks.extend(split_dataframe_fixed_size(b, block_size))
        blocks = new_blocks
    # Drop empty
    return [b for b in blocks if len(b) > 0]

