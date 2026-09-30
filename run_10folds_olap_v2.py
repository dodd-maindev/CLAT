"""Automated runner for 10-fold cross validation with OLAP v2 refinement.

Trains MIL_VT_Concept architecture across 10 folds on DDR dataset (20 epochs each).
Includes automatic resume, instant Google Drive backup per fold, and metric aggregation.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from typing import List, Optional
import pandas as pd


DRIVE_BACKUP_ROOT = "/content/drive/MyDrive/CLAT_10FOLDS_V2"
LOCAL_LOG_DIR = "log"


def is_fold_completed(fold_idx: int) -> bool:
    """Checks whether a fold has already finished training and testing."""
    metric_paths = glob.glob(
        f"{LOCAL_LOG_DIR}/milvt_olap_v2_fold_{fold_idx}/**/test_metrics.json",
        recursive=True,
    )
    if metric_paths:
        return True
    # Also check Drive backup
    if os.path.exists(DRIVE_BACKUP_ROOT):
        drive_metrics = glob.glob(
            f"{DRIVE_BACKUP_ROOT}/milvt_olap_v2_fold_{fold_idx}/**/test_metrics.json",
            recursive=True,
        )
        if drive_metrics:
            return True
    return False


def backup_fold_to_drive(fold_idx: int) -> None:
    """Copies checkpoints and logs of completed fold directly to Google Drive."""
    if not os.path.exists("/content/drive/MyDrive"):
        return

    os.makedirs(DRIVE_BACKUP_ROOT, exist_ok=True)
    src_dir = os.path.join(LOCAL_LOG_DIR, f"milvt_olap_v2_fold_{fold_idx}")
    dst_dir = os.path.join(DRIVE_BACKUP_ROOT, f"milvt_olap_v2_fold_{fold_idx}")

    if os.path.exists(src_dir):
        print(f"  [BACKUP] Syncing Fold {fold_idx} to Google Drive: {dst_dir} ...")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"  [BACKUP] Completed backup for Fold {fold_idx}.")


def execute_fold(fold_idx: int) -> int:
    """Executes fit and test for a single fold (0-9)."""
    if is_fold_completed(fold_idx):
        print(f"\n{'#'*25} FOLD {fold_idx} ALREADY COMPLETED -> SKIPPING {'#'*25}\n")
        return 0

    print(f"\n{'#'*25} STARTING OLAP v2 — FOLD {fold_idx} (20 EPOCHS) {'#'*25}\n")
    command = [
        sys.executable,
        "src/main.py",
        "fit_and_test",
        "--config",
        "configs/milvt_olap.yaml",
        "--data",
        "configs/data/DDR.yaml",
        f"--data.fold_num={fold_idx}",
        f"--trainer.logger.init_args.name=milvt_olap_v2_fold_{fold_idx}",
    ]
    ret = subprocess.run(command).returncode
    if ret == 0:
        backup_fold_to_drive(fold_idx)
    return ret


def summarize_10folds() -> None:
    """Aggregates test metrics from all folds into CSV and prints summary."""
    records = []
    for f_idx in range(10):
        search_dirs = [
            f"{LOCAL_LOG_DIR}/milvt_olap_v2_fold_{f_idx}",
            f"{DRIVE_BACKUP_ROOT}/milvt_olap_v2_fold_{f_idx}",
        ]
        found_metric = None
        for d in search_dirs:
            paths = glob.glob(f"{d}/**/test_metrics.json", recursive=True)
            if paths:
                found_metric = sorted(paths, key=os.path.getmtime)[-1]
                break

        if found_metric and os.path.exists(found_metric):
            try:
                with open(found_metric, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["Fold"] = f_idx
                    records.append(data)
            except Exception as e:
                print(f"  Warning reading metrics for Fold {f_idx}: {e}")

    if not records:
        print("No completed fold metrics found yet.")
        return

    df = pd.DataFrame(records)
    cols = ["Fold"] + [c for c in df.columns if c != "Fold"]
    df = df[cols]
    
    out_csv = os.path.join(LOCAL_LOG_DIR, "10folds_results_olap_v2.csv")
    df.to_csv(out_csv, index=False)

    if os.path.exists(DRIVE_BACKUP_ROOT):
        df.to_csv(os.path.join(DRIVE_BACKUP_ROOT, "10folds_results_olap_v2.csv"), index=False)

    print("\n" + "=" * 80)
    print("               10-FOLD CROSS-VALIDATION SUMMARY (OLAP v2)")
    print("=" * 80)
    print(df.to_string(index=False))

    mean_s = df.mean(numeric_only=True)
    std_s = df.std(numeric_only=True)
    print("-" * 80)
    print("MEAN ± STD:")
    for k in mean_s.index:
        if k != "Fold":
            print(f"  {k:<25}: {mean_s[k]:.4f} ± {std_s[k]:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    start_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_fold = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    print("=================================================================")
    print("  LAUNCHING OLAP v2 FULL 10-FOLD CROSS-VALIDATION (MIL_VT)")
    print(f"  Range: Fold {start_fold} to Fold {end_fold} (20 epochs each)")
    print("=================================================================")

    for fold in range(start_fold, end_fold + 1):
        status = execute_fold(fold)
        if status != 0:
            print(f"\n[ERROR] Fold {fold} failed with exit code {status}. Stopping pipeline.\n")
            break
        summarize_10folds()

    summarize_10folds()
    print("ALL REQUESTED FOLDS COMPLETED SUCCESSFULLY.")
