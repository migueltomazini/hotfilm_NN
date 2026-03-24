"""Basic tests for utility modules."""

import numpy as np
import pandas as pd
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

def test_block_splitting_gap():
    """Verify segmentation by time gaps produces expected number of blocks."""
    from utils.data_loader import prepare_blocks
    df = pd.DataFrame({'time': [0.0, 0.1, 0.2, 1.0, 1.1],
                       'voltage_x': np.arange(5)})
    blocks = prepare_blocks(df, gap_threshold=0.5)
    assert len(blocks) == 2
    assert len(blocks[0]) == 3
    assert len(blocks[1]) == 2


def test_block_splitting_fixed():
    """Verify segmentation by fixed block size."""
    from utils.data_loader import prepare_blocks
    df = pd.DataFrame({'time': np.arange(10)})
    blocks = prepare_blocks(df, block_size=4)
    assert len(blocks) == 3
    assert [len(b) for b in blocks] == [4, 4, 2]



def test_load_dat_formats(tmp_path, monkeypatch):
    """Ensure CSV and DAT files are both read correctly by loaders."""
    # create fake directory structure
    series = 'TEST'
    ddir = tmp_path / 'data' / 'raw' / series
    ddir.mkdir(parents=True)
    # create a simple csv for hotfilm
    csv_path = ddir / f'hotfilm_{series}.csv'
    csv_path.write_text("0,1,2,3\n1,4,5,6\n")
    dat_path = ddir / f'sonic_{series}.dat'
    dat_path.write_text("0 0.1 0.2 0.3\n1 0.4 0.5 0.6\n")

    # monkeypatch os.path.exists to look into tmp_path when searching
    def fake_exists(path):
        # replace workspace relative prefix with tmp_path
        if path.startswith('./data/raw'):
            newp = str(tmp_path) + path[1:]
            return os.path.exists(newp)
        return os.path.exists(path)
    monkeypatch.setattr('os.path.exists', fake_exists)

    from utils.data_loader import load_voltage_data, load_velocity_data
    v = load_voltage_data(series)
    assert list(v.columns) == ['time', 'voltage_x', 'voltage_y', 'voltage_z']
    assert len(v) == 2
    u = load_velocity_data(series)
    assert list(u.columns) == ['time', 'velocity_x', 'velocity_y', 'velocity_z']
    assert len(u) == 2


def test_create_csv_without_reynolds(tmp_path, monkeypatch):
    """create_csv should default/refill Reynolds from config if missing."""
    # prepare dummy raw files
    series = 'FOO'
    rdir = tmp_path / 'data' / 'raw' / series
    rdir.mkdir(parents=True)
    (rdir / f'hotfilm_{series}.csv').write_text("0,1,2,3\n")
    (rdir / f'sonic_{series}.csv').write_text("0,0.1,0.2,0.3\n")
    # config with RE_NUMBER
    cfgdir = tmp_path / 'data' / 'config'
    cfgdir.mkdir(parents=True)
    cfgfile = cfgdir / f'config_{series}.json'
    cfgfile.write_text('{"RE_NUMBER": 42}')

    # monkeypatch paths
    monkeypatch.setenv('PWD', str(tmp_path))
    monkeypatch.chdir(str(tmp_path))

    # run create_csv train without providing reynolds
    import subprocess, sys
    subprocess.run([sys.executable, 'create_csv.py', 'train', series], check=True)
    out = pd.read_csv(f'data/train/train_df_{series}.csv')
    assert 'reynolds' in out.columns
    assert out['reynolds'].iloc[0] == 42

