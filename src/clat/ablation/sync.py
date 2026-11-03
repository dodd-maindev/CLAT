"""Synchronization service between local ablation logs and Google Drive."""

import os
import shutil


class DriveSyncManager:
    """Manages real-time sync of fold results and summary tables to Google Drive."""

    def __init__(self, drive_root: str = "/content/drive/MyDrive/CLAT_Ablation_Backup") -> None:
        """Initializes sync manager and detects Google Drive availability."""
        self.drive_root = drive_root
        self.is_drive_available = os.path.exists("/content/drive/MyDrive")

    def sync_fold_results(self, experiment_name: str, fold_idx: int) -> None:
        """Copies fold metrics, logs, and checkpoints to Google Drive immediately."""
        if not self.is_drive_available:
            return
        local_dir = f"log/{experiment_name}_fold_{fold_idx}"
        if not os.path.exists(local_dir):
            return
        dest_dir = os.path.join(self.drive_root, local_dir)
        os.makedirs(dest_dir, exist_ok=True)
        for root, _, files in os.walk(local_dir):
            rel_dir = os.path.relpath(root, local_dir)
            target_sub_dir = os.path.join(dest_dir, rel_dir) if rel_dir != "." else dest_dir
            os.makedirs(target_sub_dir, exist_ok=True)
            for f in files:
                src_file = os.path.join(root, f)
                dst_file = os.path.join(target_sub_dir, f)
                if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                    shutil.copy2(src_file, dst_file)
        print(f"[Drive Sync] Real-time synced Fold {fold_idx} to Google Drive: {dest_dir}")

    def sync_summary_csv(self, csv_path: str) -> None:
        """Copies summary CSV to Google Drive root backup directory."""
        if not self.is_drive_available or not os.path.exists(csv_path):
            return
        os.makedirs(self.drive_root, exist_ok=True)
        dest_file = os.path.join(self.drive_root, os.path.basename(csv_path))
        shutil.copy2(csv_path, dest_file)
        print(f"[Drive Sync] Real-time updated summary on Drive: {dest_file}")

    def restore_from_drive_if_missing(self, experiment_name: str, fold_idx: int) -> bool:
        """Restores fold metrics from Drive if missing locally after session restart."""
        local_dir = f"log/{experiment_name}_fold_{fold_idx}"
        drive_dir = os.path.join(self.drive_root, local_dir)
        if not os.path.exists(local_dir) and os.path.exists(drive_dir):
            shutil.copytree(drive_dir, local_dir)
            print(f"[Drive Sync] Restored completed Fold {fold_idx} from Drive -> {local_dir}")
            return True
        return False

