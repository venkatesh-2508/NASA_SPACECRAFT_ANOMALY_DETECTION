"""
Tests: Data Loader
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a minimal valid dataset in a temp directory."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # 2-feature channel (telemetry + 1 command)
    np.save(str(train_dir / "P-1.npy"), np.random.randn(1000, 2).astype(np.float32))
    np.save(str(test_dir / "P-1.npy"), np.random.randn(500, 2).astype(np.float32))

    np.save(str(train_dir / "M-1.npy"), np.random.randn(800, 3).astype(np.float32))
    np.save(str(test_dir / "M-1.npy"), np.random.randn(400, 3).astype(np.float32))

    # Labels CSV
    labels = pd.DataFrame({
        "chan_id": ["P-1", "M-1"],
        "spacecraft": ["SMAP", "MSL"],
        "anomaly_sequences": ["[[100, 150]]", "[[200, 250]]"],
        "class": ["point", "contextual"],
    })
    labels.to_csv(str(tmp_path / "labeled_anomalies.csv"), index=False)

    return tmp_path


def test_dataset_available(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    assert loader.dataset_available()


def test_load_train_shape(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    data = loader.load_train("P-1")
    assert data.ndim == 2
    assert data.shape[0] == 1000
    assert data.shape[1] == 2


def test_load_test_shape(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    data = loader.load_test("P-1")
    assert data.shape == (500, 2)


def test_channel_ids(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    cids = loader.get_channel_ids()
    assert "P-1" in cids
    assert "M-1" in cids


def test_smap_msl_channels(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    smap = loader.get_smap_channels()
    msl = loader.get_msl_channels()
    assert "P-1" in smap
    assert "M-1" in msl


def test_load_labels(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    df = loader.load_labels()
    assert len(df) == 2
    assert "channel" in df.columns


def test_channel_labels(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    seqs = loader.get_channel_labels("P-1")
    assert isinstance(seqs, list)
    assert len(seqs) > 0
    assert seqs[0] == (100, 150)


def test_missing_file_raises(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    with pytest.raises(FileNotFoundError):
        loader.load_train("NONEXISTENT")


def test_dataset_unavailable(tmp_path):
    loader = DataLoader(str(tmp_path))
    assert not loader.dataset_available()


def test_caching(temp_dataset):
    loader = DataLoader(str(temp_dataset))
    d1 = loader.load_train("P-1")
    d2 = loader.load_train("P-1")
    assert d1 is d2  # same object (cached)

