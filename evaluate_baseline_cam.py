"""Visualizes and compares baseline CAM with Doctor Ground Truth."""

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from clat.model import CLAT


class BaselineCamEvaluator:
    """Evaluates and renders CAM heatmaps against doctor ground truth."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Loads baseline CLAT model from checkpoint."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLAT.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device).eval()

    def find_file(self, pattern: str) -> str:
        """Locates file matching glob pattern or returns empty string."""
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else ""

    def process(self, img_id: str, out_path: str, thresh: float = 0.6) -> None:
        """Runs inference and generates side-by-side comparison figure."""
        img_file = self.find_file(f"data/**/{img_id}.jpg")
        se_gt_file = self.find_file(f"data/**/SE/{img_id}.tif")
        if not img_file:
            print(f"Error: image {img_id}.jpg not found.")
            return

        raw_bgr = cv2.imread(img_file)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        img_384 = cv2.resize(raw_rgb, (384, 384))

        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor, return_attn=True)
            cams = outputs.cams.squeeze(0).cpu().numpy()  # shape (4, H, W)

        se_cam = cv2.resize(cams[3], (384, 384))
        se_cam = (se_cam - se_cam.min()) / (se_cam.max() - se_cam.min() + 1e-8)
        se_thresh = np.where(se_cam >= thresh, se_cam, 0.0)

        jet = cv2.applyColorMap(np.uint8(255 * se_thresh), cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
        active = (se_thresh > 0)[..., None]
        cam_overlay = np.where(active, (0.4 * img_384 + 0.6 * jet_rgb).astype(np.uint8), img_384)

        gt_view = img_384.copy()
        if se_gt_file and os.path.exists(se_gt_file):
            gt_mask = cv2.imread(se_gt_file, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (384, 384), interpolation=cv2.INTER_NEAREST)
            gt_view[gt_mask > 0] = [0, 255, 0]
            cnts, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(gt_view, (x - 12, y - 12), (x + w + 12, y + h + 12), (0, 0, 0), 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(gt_view)
        axes[0].set_title("Doctor GT: SE (Baseline Target)", color="green", fontsize=13, weight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_overlay)
        axes[1].set_title(f"Baseline CAM (Threshold={thresh})", color="blue", fontsize=13, weight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Comparison successfully saved to: {out_path}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "log/milvt_baseline_fold_0/version_0/checkpoints/epoch=11-step=1020.ckpt"
    target_id = sys.argv[2] if len(sys.argv) > 2 else "007-5470-300"
    out_file = sys.argv[3] if len(sys.argv) > 3 else f"baseline_cam_{target_id}.png"
    evaluator = BaselineCamEvaluator(ckpt)
    evaluator.process(target_id, out_file)
