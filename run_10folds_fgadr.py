"""Automated cross-validation pipeline runner for FGADR dataset using V7 OLAP."""

import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional
import pandas as pd


class FgadrCrossValidationRunner:
    """Coordinates execution and metric aggregation for FGADR cross validation."""

    def __init__(
        self,
        config_path: str = "configs/milvt_olap.yaml",
        data_config_path: str = "configs/data/FGADR.yaml",
        experiment_prefix: str = "milvt_fgadr_olap",
    ) -> None:
        """Initializes runner configurations and experiment log prefix."""
        self.config_path = config_path
        self.data_config_path = data_config_path
        self.experiment_prefix = experiment_prefix

    def execute_single_fold(self, fold_index: int) -> int:
        """Executes fit and test subcommands for a single cross-validation fold."""
        print(f"\n{'=' * 30} STARTING FGADR FOLD {fold_index} {'=' * 30}\n")
        command = [
            sys.executable,
            "src/main.py",
            "fit_and_test",
            "--config",
            self.config_path,
            "--data",
            self.data_config_path,
            f"--data.fold_num={fold_index}",
            f"--trainer.logger.init_args.name={self.experiment_prefix}_fold_{fold_index}",
        ]
        return subprocess.run(command).returncode

    def aggregate_metrics(self, total_folds: int = 10) -> Optional[pd.DataFrame]:
        """Collects test metrics across completed folds and computes mean and std."""
        records: List[Dict] = []
        for fold_idx in range(total_folds):
            pattern = f"log/{self.experiment_prefix}_fold_{fold_idx}/**/test_metrics.json"
            metric_files = glob.glob(pattern, recursive=True)
            if not metric_files:
                continue
            latest_file = sorted(metric_files, key=os.path.getmtime)[-1]
            with open(latest_file, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
                entry["Fold"] = fold_idx
                records.append(entry)
        if not records:
            print("No FGADR test metrics found to summarize.")
            return None
        dataframe = pd.DataFrame(records)
        columns = ["Fold"] + [col for col in dataframe.columns if col != "Fold"]
        dataframe = dataframe[columns]
        os.makedirs("log", exist_ok=True)
        dataframe.to_csv("log/10folds_results_fgadr_olap.csv", index=False)
        self._print_summary(dataframe)
        return dataframe

    def _print_summary(self, dataframe: pd.DataFrame) -> None:
        """Formats and prints summary statistics to standard output."""
        print("\n" + "=" * 80)
        print("                 FGADR 10-FOLD OLAP CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        print(dataframe.to_string(index=False))
        mean_series = dataframe.mean(numeric_only=True)
        std_series = dataframe.std(numeric_only=True)
        print("-" * 80 + "\nMEAN ± STD:")
        for metric_name in mean_series.index:
            if metric_name != "Fold":
                val_mean, val_std = mean_series[metric_name], std_series[metric_name]
                print(f"  {metric_name:<20}: {val_mean:.4f} ± {val_std:.4f}")
        print("=" * 80 + "\n")


def main() -> None:
    """Parses command line arguments and runs cross-validation folds."""
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    runner = FgadrCrossValidationRunner()
    for fold in range(start_fold, end_fold + 1):
        status_code = runner.execute_single_fold(fold)
        if status_code != 0:
            print(f"Training failed on fold {fold} with exit code {status_code}.")
            break
        runner.aggregate_metrics()


if __name__ == "__main__":
    main()
