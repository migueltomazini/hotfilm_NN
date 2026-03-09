"""Basic tests for utility modules."""

import numpy as np
import pytest
from utils import metrics, physics, config


def test_calculate_rmse():
    """Test RMSE calculation."""
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.1, 2.1, 3.1])
    rmse = metrics.calculate_rmse(pred, target)
    assert rmse > 0
    assert isinstance(rmse, float)


def test_calculate_mae():
    """Test MAE calculation."""
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.1, 2.1, 3.1])
    mae = metrics.calculate_mae(pred, target)
    assert mae > 0
    assert isinstance(mae, float)


def test_calculate_spectral_slope():
    """Test spectral slope calculation."""
    # Create dummy velocity signal
    velocity_signal = np.random.randn(1000, 3)
    fs = 2000
    slope = physics.calculate_spectral_slope(velocity_signal, fs)
    assert isinstance(slope, float)


def test_config_constants():
    """Test configuration constants."""
    assert config.INPUT_SIZE == 4
    assert config.OUTPUT_SIZE == 3
    assert config.EPOCHS == 256