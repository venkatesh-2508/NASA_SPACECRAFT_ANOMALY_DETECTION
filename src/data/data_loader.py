"""
NASA Spacecraft Anomaly Detection
Module: Data Loader

Loads NASA SMAP and MSL telemetry data from NPY files and labeled anomalies CSV.
Each NPY file:
  - Column 0:  Telemetry measurement (prediction target)
  - Columns 1+: Spacecraft command features
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.helpers import infer_mission

logger = setup_logger(__name__)


class DataLoader:
    """Load and validate NASA SMAP/MSL telemetry datasets."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.labels_file = self.data_dir / "labeled_anomalies.csv"

        self._train_cache: Dict[str, np.ndarray] = {}
        self._test_cache: Dict[str, np.ndarray] = {}
        self._labels_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Dataset availability
    # ------------------------------------------------------------------

    def dataset_available(self) -> bool:
        """Return True if the minimum required dataset files exist."""
        has_train = self.train_dir.exists() and any(self.train_dir.glob("*.npy"))
        has_test = self.test_dir.exists() and any(self.test_dir.glob("*.npy"))
        has_labels = self.labels_file.exists()
        return has_train and has_test and has_labels

    def get_dataset_status(self) -> dict:
        """Return a detailed status report of the dataset."""
        status = {
            "dataset_available": False,
            "train_dir_exists": self.train_dir.exists(),
            "test_dir_exists": self.test_dir.exists(),
            "labels_file_exists": self.labels_file.exists(),
            "train_files": [],
            "test_files": [],
            "num_train_files": 0,
            "num_test_files": 0,
            "num_labels": 0,
            "errors": [],
        }

        if self.train_dir.exists():
            status["train_files"] = sorted([f.stem for f in self.train_dir.glob("*.npy")])
            status["num_train_files"] = len(status["train_files"])

        if self.test_dir.exists():
            status["test_files"] = sorted([f.stem for f in self.test_dir.glob("*.npy")])
            status["num_test_files"] = len(status["test_files"])

        if self.labels_file.exists():
            try:
                df = self.load_labels()
                status["num_labels"] = len(df)
            except Exception as e:
                status["errors"].append(f"Labels load error: {e}")

        status["dataset_available"] = (
            status["num_train_files"] > 0
            and status["num_test_files"] > 0
            and status["labels_file_exists"]
        )
        return status

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def load_labels(self) -> pd.DataFrame:
        """Load the labeled_anomalies.csv file."""
        if self._labels_df is not None:
            return self._labels_df

        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        df = pd.read_csv(self.labels_file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Standardise column names
        col_map = {
            "chan_id": "channel",
            "channel_id": "channel",
            "spacecraft": "spacecraft",
            "anomaly_sequences": "anomaly_sequences",
            "class": "anomaly_class",
            "type": "anomaly_class",
            "num_values": "num_values",
        }
        df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

        # Ensure mission column
        if "spacecraft" not in df.columns:
            if "channel" in df.columns:
                df["spacecraft"] = df["channel"].apply(infer_mission)

        self._labels_df = df
        logger.info(f"Loaded {len(df)} anomaly labels from {self.labels_file}")
        return df

    def get_channel_labels(self, channel_id: str) -> List[Tuple[int, int]]:
        """Return anomaly intervals [(start, end), ...] for a given channel."""
        try:
            df = self.load_labels()
            row = df[df["channel"].str.upper() == channel_id.upper()]
            if row.empty:
                return []
            seqs_str = row.iloc[0].get("anomaly_sequences", "[]")
            if pd.isna(seqs_str) or seqs_str == "[]":
                return []
            import ast
            seqs = ast.literal_eval(str(seqs_str))
            return [(int(s[0]), int(s[1])) for s in seqs]
        except Exception as e:
            logger.warning(f"Could not parse labels for {channel_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Telemetry loading
    # ------------------------------------------------------------------

    def load_train(self, channel_id: str) -> np.ndarray:
        """Load training telemetry for a channel. Shape: (T, features)"""
        if channel_id in self._train_cache:
            return self._train_cache[channel_id]
        path = self.train_dir / f"{channel_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Train file not found: {path}")
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._train_cache[channel_id] = data
        logger.debug(f"Loaded train [{channel_id}]: shape={data.shape}")
        return data

    def load_test(self, channel_id: str) -> np.ndarray:
        """Load testing telemetry for a channel. Shape: (T, features)"""
        if channel_id in self._test_cache:
            return self._test_cache[channel_id]
        path = self.test_dir / f"{channel_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")
        data = np.load(path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self._test_cache[channel_id] = data
        logger.debug(f"Loaded test [{channel_id}]: shape={data.shape}")
        return data

    def load_all_channels(self, split: str = "train") -> Dict[str, np.ndarray]:
        """Load all channels for the given split ('train' or 'test')."""
        directory = self.train_dir if split == "train" else self.test_dir
        result = {}
        for npy_file in sorted(directory.glob("*.npy")):
            cid = npy_file.stem
            try:
                if split == "train":
                    result[cid] = self.load_train(cid)
                else:
                    result[cid] = self.load_test(cid)
            except Exception as e:
                logger.error(f"Failed to load {split} [{cid}]: {e}")
        return result

    def get_channel_ids(self) -> List[str]:
        """Return sorted list of channel IDs (intersection of train+test)."""
        train_ids = {f.stem for f in self.train_dir.glob("*.npy")} if self.train_dir.exists() else set()
        test_ids = {f.stem for f in self.test_dir.glob("*.npy")} if self.test_dir.exists() else set()
        return sorted(train_ids & test_ids)

    def get_smap_channels(self) -> List[str]:
        """Return channel IDs belonging to SMAP mission."""
        return [c for c in self.get_channel_ids() if infer_mission(c) == "SMAP"]

    def get_msl_channels(self) -> List[str]:
        """Return channel IDs belonging to MSL mission."""
        return [c for c in self.get_channel_ids() if infer_mission(c) == "MSL"]

    def clear_cache(self) -> None:
        """Clear in-memory data cache."""
        self._train_cache.clear()
        self._test_cache.clear()
        self._labels_df = None

