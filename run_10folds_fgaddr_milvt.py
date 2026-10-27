"""Automated cross-validation pipeline runner for FGADDR dataset using MIL-VT OLAP."""

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional
import pandas as pd


def parse_target_folds(cli_args: List[str]) -> List[int]:
    """Parses command-line arguments into a list of targeted fold indices."""
    if not cli_args:
        return list(range(10))
    if len(cli_args) == 2 and cli_args[0].isdigit() and cli_args[1].isdigit():
        s, e = int(cli_args[0]), int(cli_args[1])
        if s <= e <= 9:
            return list(range(s, e + 1))
    return [int(x) for x in cli_args if x.isdigit()]

class FgaddrMilvtCrossValidationRunner:
    """Coordinates execution and metric aggregation for FGADDR MIL-VT cross validation."""

    def __init__(
        self, config_path: str = "configs/milvt_olap.yaml",
        data_config_path: str = "configs/data/FGADDR.yaml",
        prefix: str = "milvt_fgaddr_olap",
        drive_sync: Optional[str] = "/content/drive/MyDrive/FGADDR_MILVT_Checkpoints",
    ) -> None:
        self.config, self.data_config, self.prefix, self.drive_sync = config_path, data_config_path, prefix, drive_sync

    def _sync_to_drive(self, fold_name: str) -> None:
        """Safely synchronizes generated checkpoints and metrics to Google Drive."""
        if not (self.drive_sync and os.path.exists("/content/drive")):
            return
        dest = os.path.join(self.drive_sync, fold_name)
        os.makedirs(dest, exist_ok=True)
        for p in [f"log/{fold_name}/**/*.ckpt", f"log/{fold_name}/**/test_metrics.json"]:
            for f in glob.glob(p, recursive=True):
                target = os.path.join(dest, os.path.basename(f))
                if not (os.path.exists(target) and os.path.samefile(f, target)):
                    shutil.copy2(f, dest)
        os.system("sync")

    def execute_single_fold(self, fold_index: int) -> int:
        """Executes fit and test subcommands for a single cross-validation fold."""
        fold_name = f"{self.prefix}_fold_{fold_index}"
        fold_dir, bkp = f"log/{fold_name}", f"log/{fold_name}_prev"
        if os.path.exists(fold_dir):
            if os.path.exists(bkp): shutil.rmtree(bkp)
            shutil.move(fold_dir, bkp)
        print(f"\n{'=' * 30} STARTING FGADDR MIL-VT FOLD {fold_index} {'=' * 30}\n")
        cmd = [
            sys.executable, "src/main.py", "fit_and_test", "--config", self.config,
            "--data", self.data_config, f"--data.fold_num={fold_index}",
            f"--trainer.logger.init_args.name={fold_name}",
        ]
        status = subprocess.run(cmd).returncode
        if status != 0 and os.path.exists(bkp): shutil.move(bkp, fold_dir)
        elif status == 0: self._sync_to_drive(fold_name)
        return status

    def aggregate_metrics(self, total_folds: int = 10) -> Optional[pd.DataFrame]:
        """Collects test metrics across completed folds and computes summary."""
        records: List[Dict] = []
        for idx in range(total_folds):
            files = glob.glob(f"log/{self.prefix}_fold_{idx}/**/test_metrics.json", recursive=True)
            if files:
                latest = sorted(files, key=os.path.getmtime)[-1]
                with open(latest, "r", encoding="utf-8") as handle:
                    entry = json.load(handle)
                    entry["Fold"] = idx
                    records.append(entry)
        if not records:
            return None
        df = pd.DataFrame(records)[["Fold"] + [c for c in pd.DataFrame(records).columns if c != "Fold"]]
        os.makedirs("log", exist_ok=True)
        df.to_csv("log/10folds_results_fgaddr_milvt_olap.csv", index=False)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Formats and prints summary statistics to standard output."""
        print(f"\n{'=' * 80}\n FGADDR 10-FOLD MIL-VT OLAP CROSS-VALIDATION SUMMARY\n{'=' * 80}\n{df.to_string(index=False)}")
        mean_s, std_s = df.mean(numeric_only=True), df.std(numeric_only=True)
        print("-" * 80 + "\nMEAN ± STD:")
        for m in [c for c in mean_s.index if c != "Fold"]:
            print(f"  {m:<20}: {mean_s[m]:.4f} ± {std_s[m]:.4f}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    runner = FgaddrMilvtCrossValidationRunner()
    for fold in parse_target_folds(sys.argv[1:]):
        if runner.execute_single_fold(fold) != 0:
            print(f"Training failed on fold {fold}.")
            break
        runner.aggregate_metrics()
