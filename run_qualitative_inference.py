"""Executes single-row qualitative inference pipeline on top Fold 7 model across 10 cases."""

import glob
import os
import sys
from typing import Optional
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))
from clat.model import CLAT  # noqa: E402
from clat.visualization import CheckpointManager, QualitativeRenderer  # noqa: E402


class FoldSevenQualitativePipeline:
    """Coordinates qualitative visualization for top-performing Fold 7 model."""

    def __init__(self, checkpoint_path: str, out_dir: str = "qualitative_results/fold_7") -> None:
        """Initializes device, model, and renderer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLAT.load_from_checkpoint(checkpoint_path, strict=False).to(self.device).eval()
        self.renderer = QualitativeRenderer()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def extract_cams(self, img_384: np.ndarray) -> np.ndarray:
        """Runs forward pass with attention extraction and returns 4-channel CAMs."""
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor, return_attn=True)
            return output.cams[0].cpu().numpy()

    def process_case(self, img_path: str) -> None:
        """Processes a single retinal image and saves 1-row 5-column qualitative figure."""
        case_id = os.path.splitext(os.path.basename(img_path))[0]
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            return
        img_384 = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (384, 384))
        cams = self.extract_cams(img_384)
        save_path = os.path.join(self.out_dir, f"case_{case_id}.png")
        self.renderer.render_single_row(img_384, case_id, cams, "OLAP (Fold 7)", save_path)


def find_fold_checkpoint(target_fold: int = 7) -> Optional[str]:
    """Finds highest epoch checkpoint for specified fold."""
    search_dirs = ["/content/drive/MyDrive/FGADR_MILVT_Checkpoints", "/content/CLAT/log", "log"]
    manager = CheckpointManager(search_dirs)
    all_folds = manager.find_all_folds()
    if target_fold in all_folds:
        return all_folds[target_fold]
    for fallback in [4, 3, 1, 0, 2, 6, 8, 9]:
        if fallback in all_folds:
            return all_folds[fallback]
    return None


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else find_fold_checkpoint(7)
    if not ckpt or not os.path.exists(ckpt):
        print("[Error] No valid Fold 7 checkpoint found.")
        sys.exit(1)

    print(f"🚀 [Qualitative Pipeline] Using Fold 7 checkpoint: {ckpt}")
    pipeline = FoldSevenQualitativePipeline(ckpt)

    all_images = glob.glob("data/FGADR/fundus_384/*.png")
    if not all_images:
        all_images = glob.glob("data/**/*.png", recursive=True) + glob.glob("data/**/*.jpg", recursive=True)

    selected_images = all_images[:10]
    print(f"📸 Generating qualitative panels for {len(selected_images)} patient cases...")

    for path in selected_images:
        pipeline.process_case(path)
    print("\n🎉 Qualitative inference for Fold 7 completed successfully!")
