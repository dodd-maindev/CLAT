"""Automated 10-fold cross-validation runner for FGADDR dataset using CaiT-xs24 OLAP."""

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional
import pandas as pd


class FgaddrCaiTCrossValidationRunner:
    """Coordinates execution and metric aggregation for joint FGADDR CaiT cross validation."""

    def __init__(
        self,
        config_path: str = "configs/cait_olap.yaml",
        data_config_path: str = "configs/data/FGADDR.yaml",
        experiment_prefix: str = "cait_fgaddr_olap",
        drive_sync: Optional[str] = "/content/drive/MyDrive/FGADDR_CaiT_Checkpoints",
    ) -> None:
        self.config_path, self.data_config_path = config_path, data_config_path
        self.experiment_prefix, self.drive_sync = experiment_prefix, drive_sync

    def execute_single_fold(self, fold_index: int) -> int:
        """Executes fit and test subcommands for a single cross-validation fold."""
        fold_name = f"{self.experiment_prefix}_fold_{fold_index}"
        fold_dir, bkp = f"log/{fold_name}", f"log/{fold_name}_prev"
        if os.path.exists(fold_dir):
            for item in os.listdir(fold_dir):
                if item.startswith("version_") and not item.replace("version_", "").isdigit():
                    shutil.rmtree(os.path.join(fold_dir, item), ignore_errors=True)
            if os.path.exists(bkp):
                shutil.rmtree(bkp)
            shutil.move(fold_dir, bkp)
        print(f"\n{'=' * 30} STARTING FGADDR CAIT FOLD {fold_index} {'=' * 30}\n")
        cmd = [
            sys.executable, "src/main.py", "fit_and_test",
            "--config", self.config_path, "--data", self.data_config_path,
            f"--data.fold_num={fold_index}", f"--trainer.logger.init_args.name={fold_name}",
        ]
        status = subprocess.run(cmd).returncode
        if status != 0 and os.path.exists(bkp):
            shutil.move(bkp, fold_dir)
        elif status == 0 and self.drive_sync and os.path.exists("/content/drive"):
            dest = os.path.join(self.drive_sync, fold_name)
            os.makedirs(dest, exist_ok=True)
            for f in glob.glob(f"log/{fold_name}/**/*.ckpt", recursive=True):
                shutil.copy2(f, dest)
            for f in glob.glob(f"log/{fold_name}/**/test_metrics.json", recursive=True):
                shutil.copy2(f, dest)
            os.system("sync")
            print(f"[Drive Sync] Checkpoints and metrics copied to {dest}")
        return status

    def aggregate_metrics(self, total_folds: int = 10) -> Optional[pd.DataFrame]:
        """Collects test metrics across completed folds and computes summary."""
        records: List[Dict] = []
        for fold_idx in range(total_folds):
            files = glob.glob(f"log/{self.experiment_prefix}_fold_{fold_idx}/**/test_metrics.json", recursive=True)
            if files:
                latest = sorted(files, key=os.path.getmtime)[-1]
                with open(latest, "r", encoding="utf-8") as handle:
                    entry = json.load(handle)
                    entry["Fold"] = fold_idx
                    records.append(entry)
        if not records:
            return None
        df = pd.DataFrame(records)
        df = df[["Fold"] + [c for c in df.columns if c != "Fold"]]
        os.makedirs("log", exist_ok=True)
        csv_path = "log/10folds_results_fgaddr_cait_olap.csv"
        df.to_csv(csv_path, index=False)
        if self.drive_sync and os.path.exists("/content/drive"):
            os.makedirs(self.drive_sync, exist_ok=True)
            shutil.copy2(csv_path, os.path.join(self.drive_sync, "10folds_results_fgaddr_cait_olap.csv"))
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Formats and prints summary statistics to standard output."""
        print(f"\n{'=' * 80}\n FGADDR 10-FOLD CAIT OLAP CROSS-VALIDATION SUMMARY\n{'=' * 80}")
        print(df.to_string(index=False))
        mean_s, std_s = df.mean(numeric_only=True), df.std(numeric_only=True)
        print("-" * 80 + "\nMEAN ± STD:")
        for m in [col for col in mean_s.index if col != "Fold"]:
            print(f"  {m:<20}: {mean_s[m]:.4f} ± {std_s[m]:.4f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    runner = FgaddrCaiTCrossValidationRunner()
    for f in range(start, end + 1):
        if runner.execute_single_fold(f) != 0:
            print(f"Training failed on fold {f}.")
            break
        runner.aggregate_metrics()
