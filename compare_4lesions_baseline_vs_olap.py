"""Compares 4 lesion concept CAMs across Doctor GT, Baseline, and OLAP."""

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from clat.model import CLAT


class ThreeTierLesionVisualizer:
    """Renders 3x4 comparative grid: Doctor GT, Baseline, and OLAP models."""

    LESION_CODES = ["EX", "HE", "MA", "SE"]
    LESION_TITLES = ["Hard Exudate (EX)", "Haemorrhage (HE)", "Microaneurysm (MA)", "Soft Exudate (SE)"]

    def __init__(self, baseline_ckpt: str, olap_ckpt: str, device: str = "cuda") -> None:
        """Loads both neural networks onto available computation device."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.baseline_model = CLAT.load_from_checkpoint(baseline_ckpt).to(self.device).eval()
        self.olap_model = CLAT.load_from_checkpoint(olap_ckpt).to(self.device).eval()

    def find_file(self, pattern: str) -> str:
        """Resolves file path matching glob pattern."""
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else ""

    def render_gt(self, base_rgb: np.ndarray, gt_file: str) -> np.ndarray:
        """Renders green ground truth mask with black bounding rectangles."""
        view = base_rgb.copy()
        if gt_file and os.path.exists(gt_file):
            gt_mask = cv2.resize(cv2.imread(gt_file, 0), (384, 384), interpolation=cv2.INTER_NEAREST)
            view[gt_mask > 0] = [0, 255, 0]
            cnts, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(view, (max(0, x - 10), max(0, y - 10)), (min(384, x + w + 10), min(384, y + h + 10)), (0, 0, 0), 2)
        return view

    def render_cam(self, base_rgb: np.ndarray, raw_cam: np.ndarray, thresh: float) -> np.ndarray:
        """Overlays JET heatmap for tokens surpassing confidence threshold."""
        cam_res = cv2.resize(raw_cam, (384, 384))
        norm_cam = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        mask = norm_cam >= thresh
        jet = cv2.applyColorMap(np.uint8(255 * np.where(mask, norm_cam, 0.0)), cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
        return np.where(mask[..., None], (0.4 * base_rgb + 0.6 * jet_rgb).astype(np.uint8), base_rgb)

    def execute(self, image_id: str, out_file: str, threshold: float = 0.6) -> None:
        """Processes image and outputs 3-row grid figure."""
        img_path = self.find_file(f"data/**/{image_id}.jpg")
        if not img_path:
            print(f"Image {image_id}.jpg not found.")
            return

        img_384 = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (384, 384))
        norm_t = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm_t).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            bl_cams = self.baseline_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()
            olap_cams = self.olap_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        row_labels = ["Doctor GT", "Baseline (MIL-VT)", "OLAP (Version 4)"]
        row_colors = ["darkgreen", "navy", "darkred"]

        for col, (code, title) in enumerate(zip(self.LESION_CODES, self.LESION_TITLES)):
            gt_path = self.find_file(f"data/**/{code}/{image_id}.tif")
            axes[0, col].imshow(self.render_gt(img_384, gt_path))
            axes[1, col].imshow(self.render_cam(img_384, bl_cams[col], threshold))
            axes[2, col].imshow(self.render_cam(img_384, olap_cams[col], threshold))

            for row in range(3):
                axes[row, col].set_title(f"{row_labels[row]}: {title}", fontsize=11, weight="bold", color=row_colors[row])
                axes[row, col].axis("off")

        plt.suptitle(f"Tri-Tier 4-Lesion Concept Comparison (Case: {image_id}, Threshold={threshold})", fontsize=16, weight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved 3x4 figure: {out_file}")


if __name__ == "__main__":
    bl = sys.argv[1] if len(sys.argv) > 1 else "log/milvt_baseline_fold_0/version_0/checkpoints/epoch=11-step=1020.ckpt"
    ol = sys.argv[2] if len(sys.argv) > 2 else "log/milvt_olap_fold_0/version_4/checkpoints/epoch=12-step=1105.ckpt"
    cid = sys.argv[3] if len(sys.argv) > 3 else "007-5470-300"
    out = sys.argv[4] if len(sys.argv) > 4 else f"tri_tier_comparison_{cid}.png"
    ThreeTierLesionVisualizer(bl, ol).execute(cid, out)
