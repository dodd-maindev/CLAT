"""Automated runner for 10-fold cross-validation of MIL-VT Baseline."""

import glob
import json
import os
import subprocess
import sys
import pandas as pd


class BaselineCrossValidationRunner:
    """Orchestrates training and testing for the baseline MIL-VT model."""

    def __init__(
        self,
        config_path: str = "configs/milvt_baseline.yaml",
        data_config_path: str = "configs/data/DDR.yaml",
    ) -> None:
        """Initializes configuration paths for baseline model and dataset."""
        self.config_path = config_path
        self.data_config_path = data_config_path

    def execute_fold(self, fold_index: int) -> int:
        """Executes fit and test for a single fold index."""
        print(f"\n{'#' * 30} STARTING BASELINE FOLD {fold_index} {'#' * 30}\n")
        command = [
            sys.executable,
            "src/main.py",
            "fit_and_test",
            "--config",
            self.config_path,
            "--data",
            self.data_config_path,
            f"--data.fold_num={fold_index}",
            f"--trainer.logger.init_args.name=milvt_baseline_fold_{fold_index}",
        ]
        return subprocess.run(command).returncode

    def summarize_results(self) -> None:
        """Aggregates all fold test metrics into a CSV and prints the summary."""
        records = []
        for fold_index in range(10):
            pattern = f"log/milvt_baseline_fold_{fold_index}/**/test_metrics.json"
            metric_paths = glob.glob(pattern, recursive=True)
            if metric_paths:
                latest_path = sorted(metric_paths, key=os.path.getmtime)[-1]
                with open(latest_path, "r", encoding="utf-8") as file_handle:
                    entry = json.load(file_handle)
                    entry["Fold"] = fold_index
                    records.append(entry)

        if not records:
            print("No baseline fold metrics found to summarize.")
            return

        dataframe = pd.DataFrame(records)
        columns = ["Fold"] + [c for c in dataframe.columns if c != "Fold"]
        dataframe = dataframe[columns]
        dataframe.to_csv("log/10folds_results_baseline.csv", index=False)

        print("\n" + "=" * 80)
        print("          BASELINE MIL-VT 10-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        print(dataframe.to_string(index=False))

        mean_series = dataframe.mean(numeric_only=True)
        std_series = dataframe.std(numeric_only=True)
        print("-" * 80)
        print("MEAN ± STD:")
        for metric_name in mean_series.index:
            if metric_name != "Fold":
                print(
                    f"  {metric_name:<20}: "
                    f"{mean_series[metric_name]:.4f} ± {std_series[metric_name]:.4f}"
                )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    runner = BaselineCrossValidationRunner()
    for fold in range(start_fold, end_fold + 1):
        exit_code = runner.execute_fold(fold)
        if exit_code != 0:
            print(f"Error encountered in Baseline Fold {fold}, code {exit_code}")
            break
        runner.summarize_results()
