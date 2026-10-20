"""10-fold cross validation runner for FGADR dataset using CaiT-xs24 OLAP."""

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional
import pandas as pd


class FgadrCaiTCrossValidationRunner:
    """Coordinates execution and metric aggregation for FGADR CaiT CV."""

    def __init__(
        self,
        config_path: str = "configs/cait_olap.yaml",
        data_config_path: str = "configs/data/FGADR.yaml",
        prefix: str = "cait_fgadr_olap",
        drive_sync: Optional[str] = "/content/drive/MyDrive/FGADR_CaiT_Checkpoints",
    ) -> None:
        self.config_path = config_path
        self.data_config_path = data_config_path
        self.prefix = prefix
        self.drive_sync = drive_sync

    def execute_single_fold(self, fold_index: int) -> int:
        """Executes fit and test for a single fold."""
        fold_name = f"{self.prefix}_fold_{fold_index}"
        print(f"\n{'=' * 30} STARTING FGADR CAIT FOLD {fold_index} {'=' * 30}\n")
        cmd = [
            sys.executable, "src/main.py", "fit_and_test",
            "--config", self.config_path,
            "--data", self.data_config_path,
            f"--data.fold_num={fold_index}",
            f"--trainer.logger.init_args.name={fold_name}",
        ]
        status = subprocess.run(cmd).returncode
        if status == 0 and self.drive_sync and os.path.exists("/content/drive"):
            dest = os.path.join(self.drive_sync, fold_name)
            os.makedirs(dest, exist_ok=True)
            for f in glob.glob(f"log/{fold_name}/**/*.ckpt", recursive=True):
                shutil.copy2(f, dest)
            for f in glob.glob(f"log/{fold_name}/**/test_metrics.json", recursive=True):
                shutil.copy2(f, dest)
        return status

    def aggregate_metrics(self, total_folds: int = 10) -> Optional[pd.DataFrame]:
        """Collects test metrics across completed folds."""
        records: List[Dict] = []
        for f_idx in range(total_folds):
            pattern = f"log/{self.prefix}_fold_{f_idx}/**/test_metrics.json"
            files = glob.glob(pattern, recursive=True)
            if files:
                latest = sorted(files, key=os.path.getmtime)[-1]
                with open(latest, "r", encoding="utf-8") as handle:
                    entry = json.load(handle)
                    entry["Fold"] = f_idx
                    records.append(entry)
        if not records:
            return None
        df = pd.DataFrame(records)
        df = df[["Fold"] + [c for c in df.columns if c != "Fold"]]
        os.makedirs("log", exist_ok=True)
        df.to_csv(f"log/10folds_results_{self.prefix}.csv", index=False)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Formats and prints summary statistics to standard output."""
        print(f"\n{'=' * 80}\n    FGADR CAIT-XS24 10-FOLD SUMMARY ({self.prefix})\n{'=' * 80}")
        print(df.to_string(index=False))
        mean_s, std_s = df.mean(numeric_only=True), df.std(numeric_only=True)
        print("-" * 80 + "\nMEAN ± STD:")
        for m in mean_s.index:
            if m != "Fold":
                print(f"  {m:<20}: {mean_s[m]:.4f} ± {std_s[m]:.4f}")
        print("=" * 80 + "\n")


def main() -> None:
    """Parses command line arguments and runs cross-validation folds."""
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    runner = FgadrCaiTCrossValidationRunner()
    for fold in range(start_fold, end_fold + 1):
        status = runner.execute_single_fold(fold)
        if status != 0:
            print(f"Failed fold {fold} with exit code {status}.")
            break
        runner.aggregate_metrics()


if __name__ == "__main__":
    main()
