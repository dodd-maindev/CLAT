"""Automated runner for 10-fold cross validation of Improved Medical-CLAT."""

import glob
import json
import os
import subprocess
import sys
import pandas as pd


class ImprovedCaiTCrossValidationRunner:
    """Executes 10-fold cross-validation pipeline with all three clinical improvements."""

    def __init__(
        self,
        config_path: str = "configs/cait_improved_clat.yaml",
        data_config_path: str = "configs/data/FGADDR.yaml",
    ) -> None:
        """Initializes model and dataset configuration paths."""
        self.config_path = config_path
        self.data_config_path = data_config_path

    def execute_fold(self, fold_idx: int) -> int:
        """Executes training and testing for a specified fold index."""
        print(f"\n{'#'*30} STARTING IMPROVED CLAT FOLD {fold_idx} {'#'*30}\n")
        cmd = [
            sys.executable,
            "src/main.py",
            "fit_and_test",
            "--config",
            self.config_path,
            "--data",
            self.data_config_path,
            f"--data.fold_num={fold_idx}",
            f"--trainer.logger.init_args.name=cait_improved_fold_{fold_idx}",
        ]
        return subprocess.run(cmd).returncode

    def run_all_folds(self) -> None:
        """Iterates sequentially through all 10 folds and prints aggregate summary."""
        for fold in range(10):
            return_code = self.execute_fold(fold)
            if return_code != 0:
                print(f"Warning: Fold {fold} exited with code {return_code}")
        self.summarize()

    def summarize(self) -> None:
        """Aggregates all fold test metrics into a CSV and logs the mean and standard deviation."""
        records = []
        for f_idx in range(10):
            pattern = f"log/cait_improved_fold_{f_idx}/**/test_metrics.json"
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
        os.makedirs("log", exist_ok=True)
        df.to_csv("log/10folds_results_cait_improved.csv", index=False)

        print("\n" + "=" * 80)
        print("     10-FOLD IMPROVED CAIT-XS24 + MSLR + ICDR-CRG + OFL FINAL SUMMARY")
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


if __name__ == "__main__":
    runner = ImprovedCaiTCrossValidationRunner()
    runner.run_all_folds()

