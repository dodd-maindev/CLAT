"""Execution runner for training and testing ablation experiment folds."""

import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional
from .sync import DriveSyncManager


class AblationExperimentRunner:
    """Executes fit_and_test runs across cross-validation folds with real-time sync."""

    def __init__(
        self,
        data_config: str = "configs/data/FGADDR.yaml",
        sync_manager: Optional[DriveSyncManager] = None,
    ) -> None:
        """Initializes runner with dataset configuration and sync manager."""
        self.data_config = data_config
        self.sync_manager = sync_manager or DriveSyncManager()

    def is_fold_completed(self, experiment_name: str, fold_idx: int) -> bool:
        """Checks if fold is already completed locally or on Google Drive."""
        self.sync_manager.restore_from_drive_if_missing(experiment_name, fold_idx)
        pattern = f"log/{experiment_name}_fold_{fold_idx}/**/test_metrics.json"
        return len(glob.glob(pattern, recursive=True)) > 0

    def execute_fold(
        self,
        config_path: str,
        experiment_name: str,
        fold_idx: int,
        dry_run: bool = False,
    ) -> int:
        """Runs fit_and_test for a specific fold, skipping if already completed."""
        if self.is_fold_completed(experiment_name, fold_idx):
            print(f"[Ablation] Fold {fold_idx} already finished. Skipping to save time!")
            return 0

        log_name = f"{experiment_name}_fold_{fold_idx}"
        command = [
            sys.executable,
            "src/main.py",
            "fit_and_test",
            "--config",
            config_path,
            "--data",
            self.data_config,
            f"--data.fold_num={fold_idx}",
            f"--trainer.logger.init_args.name={log_name}",
        ]
        print(f"\n[Ablation] Executing Fold {fold_idx} -> {' '.join(command)}")
        if dry_run:
            return 0

        ret_code = subprocess.run(command).returncode
        if ret_code == 0:
            self.sync_manager.sync_fold_results(experiment_name, fold_idx)
        return ret_code

    def collect_fold_metrics(
        self, experiment_name: str, start_fold: int, end_fold: int
    ) -> List[Dict[str, Any]]:
        """Collects test_metrics.json outputs across finished folds."""
        records: List[Dict[str, Any]] = []
        for f_idx in range(start_fold, end_fold + 1):
            self.sync_manager.restore_from_drive_if_missing(experiment_name, f_idx)
            pattern = f"log/{experiment_name}_fold_{f_idx}/**/test_metrics.json"
            matches = glob.glob(pattern, recursive=True)
            if matches:
                latest_file = sorted(matches, key=os.path.getmtime)[-1]
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["Fold"] = f_idx
                    records.append(data)
        return records
