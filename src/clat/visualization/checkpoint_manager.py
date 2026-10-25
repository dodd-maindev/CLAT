"""Manages discovery and archive extraction for cross-validation model checkpoints."""

import glob
import os
import zipfile
from typing import Dict, List


class CheckpointManager:
    """Discovers and prepares model checkpoints across all validation folds."""

    def __init__(self, search_paths: List[str]) -> None:
        """Initializes search directories."""
        self.search_paths = search_paths

    def extract_archive_if_needed(self, archive_name: str = "FGADR_MILVT_Full_Logs.zip") -> None:
        """Extracts checkpoint zip archive if present and folds are missing."""
        for path in self.search_paths:
            zip_candidate = os.path.join(path, archive_name)
            if os.path.exists(zip_candidate):
                missing_folds = any(
                    not glob.glob(os.path.join(path, f"*fold_{f}*", "**", "*.ckpt"), recursive=True)
                    for f in [7, 4, 3, 1]
                )
                if missing_folds:
                    print(f"📦 [CheckpointManager] Extracting {zip_candidate} into {path}...")
                    with zipfile.ZipFile(zip_candidate, "r") as archive:
                        archive.extractall(path)

    def find_all_folds(self) -> Dict[int, str]:
        """Finds highest epoch checkpoint for each available fold (0 through 9)."""
        self.extract_archive_if_needed()
        fold_map: Dict[int, str] = {}
        for fold_idx in range(10):
            fold_ckpts: List[str] = []
            for root_dir in self.search_paths:
                pattern = os.path.join(root_dir, f"*fold_{fold_idx}*", "**", "*.ckpt")
                matches = glob.glob(pattern, recursive=True)
                fold_ckpts.extend(matches)
            if fold_ckpts:
                fold_map[fold_idx] = sorted(fold_ckpts)[-1]
        return fold_map
