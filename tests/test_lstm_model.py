"""
Tests: LSTM Model
"""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_import_torch():
    import torch
    assert torch.__version__


def test_model_forward_pass():
    import torch
    from src.models.lstm_model import LSTMAnomalyDetector
    model = LSTMAnomalyDetector(input_size=3, hidden_size=16, num_layers=2, dropout=0.0)
    x = torch.randn(8, 50, 3)  # batch=8, seq=50, feat=3
    out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_model_config():
    from src.models.lstm_model import LSTMAnomalyDetector
    model = LSTMAnomalyDetector(input_size=5, hidden_size=20, num_layers=2, dropout=0.3)
    cfg = model.get_config()
    assert cfg["input_size"] == 5
    assert cfg["hidden_size"] == 20
    assert cfg["num_layers"] == 2


def test_save_load_checkpoint(tmp_path):
    import torch
    from src.models.lstm_model import (
        LSTMAnomalyDetector,
        save_checkpoint,
        load_model_from_checkpoint,
    )
    model = LSTMAnomalyDetector(input_size=2, hidden_size=8, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = str(tmp_path / "test_model.pt")
    save_checkpoint(model, optimizer, epoch=5, val_loss=0.001, filepath=ckpt_path)

    loaded = load_model_from_checkpoint(ckpt_path)
    assert loaded is not None


def test_checkpoint_not_found_raises():
    from src.models.lstm_model import load_checkpoint
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/path.pt")


def test_trainer_predict():
    import torch
    import numpy as np
    from src.utils.config import load_config
    from src.models.lstm_model import LSTMAnomalyDetector
    from src.models.model_trainer import ModelTrainer

    cfg = load_config()
    model = LSTMAnomalyDetector(input_size=2, hidden_size=8, num_layers=1, dropout=0.0)
    X = np.random.randn(50, 50, 2).astype(np.float32)
    trainer = ModelTrainer(cfg)
    preds = trainer.predict(model, X)
    assert preds.shape == (50,)

