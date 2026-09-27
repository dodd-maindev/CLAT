"""Automated runner for 10-fold cross validation with full monitoring and logging."""

import glob
import json
import os
import subprocess
import sys
import pandas as pd


def execute_fold(fold_idx: int) -> int:
    """Executes fit and test for a single fold (0-9)."""
    print(f"\n{'#'*30} STARTING FOLD {fold_idx} (Index 0-9) {'#'*30}\n")
    command = [
        sys.executable,
        "src/main.py",
        "fit_and_test",
        "--config",
        "configs/milvt_olap.yaml",
        "--data",
        "configs/data/DDR.yaml",
        f"--data.fold_num={fold_idx}",
        f"--trainer.logger.init_args.name=milvt_olap_fold_{fold_idx}",
    ]
    return subprocess.run(command).returncode


def summarize_10folds() -> None:
    """Aggregates test metrics from all folds into CSV and Markdown summary."""
    records = []
    for f_idx in range(0, 10):
        metric_paths = glob.glob(f"log/milvt_olap_fold_{f_idx}/**/test_metrics.json", recursive=True)
        if metric_paths:
            latest_path = sorted(metric_paths, key=os.path.getmtime)[-1]
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["Fold"] = f_idx
                records.append(data)

    if not records:
        print("No fold metrics found to summarize.")
        return

    df = pd.DataFrame(records)
    cols = ["Fold"] + [c for c in df.columns if c != "Fold"]
    df = df[cols]
    df.to_csv("log/10folds_results_olap.csv", index=False)

    print("\n" + "=" * 80)
    print("                      10-FOLD CROSS-VALIDATION FINAL SUMMARY")
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
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    for fold in range(start_fold, end_fold + 1):
        ret = execute_fold(fold)
        if ret != 0:
            print(f"Error in Fold {fold}, exit code {ret}")
            break
        summarize_10folds()

    summarize_10folds()
