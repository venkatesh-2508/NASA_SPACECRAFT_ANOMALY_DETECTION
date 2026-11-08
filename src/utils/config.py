"""
NASA Spacecraft Anomaly Detection
Utility: Configuration loader
"""
import os
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels above this file (src/utils/config.py -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file and return as a dict."""
    path = Path(config_path) if config_path else CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths to absolute paths from project root
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    """Return an absolute Path for a path key in the config."""
    root = Path(cfg.get("_project_root", PROJECT_ROOT))
    return root / cfg["paths"][key]


def get_nested(cfg: dict, *keys, default: Any = None) -> Any:
    """Safely retrieve a nested config value."""
    obj = cfg
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj

