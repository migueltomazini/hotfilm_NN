"""Configuration constants and paths for the hotfilm NN project.

This module centralizes all global constants, file paths, and configuration
settings used across the project to avoid duplication and ensure consistency.
"""

import os

# ==============================================================================
# PATH CONFIGURATIONS
# ==============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(DATA_DIR, 'config')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
RUN_DIR = os.path.join(DATA_DIR, 'run')

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

INPUT_SIZE = 4   # voltage_x, voltage_y, voltage_z, reynolds
OUTPUT_SIZE = 3  # velocity_x, velocity_y, velocity_z

# Training hyperparameters
EPOCHS = 256
EPOCHS_FINETUNE = 128

# ==============================================================================
# DATA CONFIGURATIONS
# ==============================================================================

# Column names
VOLTAGE_COLS = ['voltage_x', 'voltage_y', 'voltage_z']
VELOCITY_COLS = ['velocity_x', 'velocity_y', 'velocity_z']
TIME_COL = 'time'
REYNOLDS_COL = 'reynolds'

# ==============================================================================
# PHYSICS CONSTANTS
# ==============================================================================

# Default sampling frequencies
FS_HOTFILM_DEFAULT = 2000  # Hz
FS_SONIC_DEFAULT = 20      # Hz

# Spectral analysis ranges
SPECTRAL_MASK_LOW = 5.0    # Hz
SPECTRAL_MASK_HIGH = 50.0  # Hz

# Turbulence targets
SPECTRAL_SLOPE_TARGET = -5/3  # Kolmogorov's law
ISOTROPY_RATIO_TARGET = 4/3   # Theoretical value