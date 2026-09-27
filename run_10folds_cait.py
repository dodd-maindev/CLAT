"""Automated runner for 10-fold cross validation using CaiT backbone with OLAP."""

import glob
import json
import os
import subprocess
import sys
import pandas as pd


class CaiTCrossValidationRunner:
    """Class responsible for executing 10-fold CV on CaiT-xs24 backbone."""

    def __init__(self, config_path: str, data_config_path: str) -> None:
        self.config_path = config_path
        self.data_config_path = data_config_path

    def execute_fold(self, fold_idx: int) -> int:
        """Executes training and testing pipeline for a specific fold index."""
        print(f"\n{'#'*30} STARTING CAIT-OLAP FOLD {fold_idx} {'#'*30}\n")
        cmd = [
            sys.executable,
            "src/main.py",
            "fit_and_test",
            "--config",
            self.config_path,
            "--data",
            self.data_config_path,
            f"--data.fold_num={fold_idx}",
            f"--trainer.logger.init_args.name=cait_olap_fold_{fold_idx}",
        ]
        return subprocess.run(cmd).returncode

    def summarize(self) -> None:
        """Aggregates all fold test metrics into a CSV and prints the summary."""
        records = []
        for f_idx in range(10):
            pattern = f"log/cait_olap_fold_{f_idx}/**/test_metrics.json"
            metric_paths = glob.glob(pattern, recursive=True)
            if metric_paths:
                latest = sorted(metric_paths, key=os.path.getmtime)[-1]
                with open(latest, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    entry["Fold"] = f_idx
                    records.append(entry)

        if not records:
            print("No fold metrics found to summarize.")
            return

        df = pd.DataFrame(records)
        cols = ["Fold"] + [c for c in df.columns if c != "Fold"]
        df = df[cols]
        df.to_csv("log/10folds_results_cait_olap.csv", index=False)

        print("\n" + "=" * 80)
        print("          10-FOLD CAIT-XS24 + OLAP FINAL SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))

        mean_s = df.mean(numeric_only=True)
        std_s = df.std(numeric_only=True)
        print("-" * 80)
        print("MEAN ± STD:")
        for k in mean_s.index:
            if k != "Fold":
                print(f"  {k:<20}: {mean_s[k]:.4f} ± {std_s[k]:.4f}")
        print("=" * 80 + "\n")


def main() -> None:
    """Entry point for running CaiT-xs24 10-fold cross validation."""
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    runner = CaiTCrossValidationRunner(
        config_path="configs/cait_olap.yaml",
        data_config_path="configs/data/DDR.yaml",
    )

    for fold in range(start_fold, end_fold + 1):
        exit_code = runner.execute_fold(fold)
        if exit_code != 0:
            print(f"Error in CaiT-OLAP fold {fold}, exit code: {exit_code}")
            break
        runner.summarize()

    runner.summarize()


if __name__ == "__main__":
    main()
