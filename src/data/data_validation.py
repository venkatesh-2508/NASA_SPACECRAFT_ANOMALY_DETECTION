"""
NASA Spacecraft Anomaly Detection
Module: Data Validation

Validates dataset integrity and produces a detailed validation report.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """Validates the NASA SMAP/MSL dataset and produces a structured report."""

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def validate_all(self) -> Dict:
        """Run full dataset validation and return a report dict."""
        report = {
            "passed": False,
            "dataset_available": False,
            "train_summary": {},
            "test_summary": {},
            "label_summary": {},
            "channel_details": [],
            "warnings": [],
            "errors": [],
        }

        status = self.loader.get_dataset_status()
        report["dataset_available"] = status["dataset_available"]

        if not status["dataset_available"]:
            report["errors"].append("Dataset is not available. Please download the NASA SMAP/MSL dataset.")
            report["errors"].append(f"Expected directories: {self.loader.train_dir}, {self.loader.test_dir}")
            report["errors"].append(f"Expected labels file: {self.loader.labels_file}")
            return report

        # ------ Train summary ------
        train_channels = self.loader.get_channel_ids()
        train_shapes = {}
        for cid in train_channels:
            try:
                arr = self.loader.load_train(cid)
                train_shapes[cid] = arr.shape
            except Exception as e:
                report["errors"].append(f"Cannot load train/{cid}: {e}")

        report["train_summary"] = {
            "num_files": len(train_channels),
            "total_time_steps": int(sum(s[0] for s in train_shapes.values())),
            "feature_dims": {cid: s[1] for cid, s in train_shapes.items()},
        }

        # ------ Test summary ------
        test_shapes = {}
        for cid in train_channels:
            try:
                arr = self.loader.load_test(cid)
                test_shapes[cid] = arr.shape
            except Exception as e:
                report["errors"].append(f"Cannot load test/{cid}: {e}")

        report["test_summary"] = {
            "num_files": len(test_shapes),
            "total_time_steps": int(sum(s[0] for s in test_shapes.values())),
        }

        # ------ Label summary ------
        try:
            labels = self.loader.load_labels()
            report["label_summary"] = {
                "num_rows": len(labels),
                "columns": list(labels.columns),
                "spacecraft_counts": labels["spacecraft"].value_counts().to_dict() if "spacecraft" in labels.columns else {},
                "anomaly_class_counts": labels["anomaly_class"].value_counts().to_dict() if "anomaly_class" in labels.columns else {},
            }
        except Exception as e:
            report["errors"].append(f"Cannot load labels: {e}")

        # ------ Per-channel details ------
        for cid in train_channels:
            detail = {"channel": cid, "train_shape": None, "test_shape": None, "nan_train": False, "nan_test": False, "num_anomalies": 0}
            if cid in train_shapes:
                detail["train_shape"] = list(train_shapes[cid])
                arr = self.loader.load_train(cid)
                detail["nan_train"] = bool(np.any(np.isnan(arr)))
            if cid in test_shapes:
                detail["test_shape"] = list(test_shapes[cid])
                arr = self.loader.load_test(cid)
                detail["nan_test"] = bool(np.any(np.isnan(arr)))
            detail["num_anomalies"] = len(self.loader.get_channel_labels(cid))
            if detail["nan_train"]:
                report["warnings"].append(f"NaN values in train/{cid}")
            if detail["nan_test"]:
                report["warnings"].append(f"NaN values in test/{cid}")
            report["channel_details"].append(detail)

        report["passed"] = len(report["errors"]) == 0
        logger.info(f"Validation {'PASSED' if report['passed'] else 'FAILED'} — {len(report['errors'])} errors, {len(report['warnings'])} warnings")
        return report

    def print_report(self, report: Dict) -> None:
        """Pretty-print the validation report."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  NASA DATASET VALIDATION REPORT")
        print(sep)
        print(f"  Status     : {'PASSED ✓' if report['passed'] else 'FAILED ✗'}")
        print(f"  Available  : {report['dataset_available']}")

        if report["train_summary"]:
            ts = report["train_summary"]
            print(f"\n  TRAINING DATA")
            print(f"    Files        : {ts.get('num_files', 0)}")
            print(f"    Total steps  : {ts.get('total_time_steps', 0):,}")

        if report["test_summary"]:
            ts = report["test_summary"]
            print(f"\n  TESTING DATA")
            print(f"    Files        : {ts.get('num_files', 0)}")
            print(f"    Total steps  : {ts.get('total_time_steps', 0):,}")

        if report["label_summary"]:
            ls = report["label_summary"]
            print(f"\n  ANOMALY LABELS")
            print(f"    Total events : {ls.get('num_rows', 0)}")
            for mission, cnt in ls.get("spacecraft_counts", {}).items():
                print(f"    {mission:8s}      : {cnt} events")
            for cls, cnt in ls.get("anomaly_class_counts", {}).items():
                print(f"    {cls:8s}      : {cnt} events")

        if report["warnings"]:
            print(f"\n  WARNINGS ({len(report['warnings'])})")
            for w in report["warnings"][:10]:
                print(f"    ⚠  {w}")

        if report["errors"]:
            print(f"\n  ERRORS ({len(report['errors'])})")
            for e in report["errors"]:
                print(f"    ✗  {e}")

        print(f"\n{sep}\n")

