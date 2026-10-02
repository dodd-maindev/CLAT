"""Compares 4 lesion concept CAMs between Baseline and OLAP checkpoints."""

import glob
import os
import sys
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from clat.model import CLAT


class LesionComparisonVisualizer:
    """Renders side-by-side 2x4 CAM comparison across all 4 lesion concepts."""

    LESION_NAMES = ["EX", "HE", "MA", "SE"]
    LESION_FULL_NAMES = ["Hard Exudate (EX)", "Haemorrhage (HE)", "Microaneurysm (MA)", "Soft Exudate (SE)"]

    def __init__(self, baseline_ckpt: str, olap_ckpt: str, device: str = "cuda") -> None:
        """Initializes both models on specified compute device."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.baseline_model = CLAT.load_from_checkpoint(baseline_ckpt).to(self.device).eval()
        self.olap_model = CLAT.load_from_checkpoint(olap_ckpt).to(self.device).eval()

    def find_file(self, pattern: str) -> str:
        """Finds first matching filepath for given glob pattern."""
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else ""

    def render_overlay(self, img_rgb: np.ndarray, cam_2d: np.ndarray, gt_file: str, thresh: float) -> np.ndarray:
        """Overlays thresholded CAM and optional doctor ground truth bounding box."""
        cam_res = cv2.resize(cam_2d, (384, 384))
        cam_norm = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        active_mask = cam_norm >= thresh
        jet_map = cv2.applyColorMap(np.uint8(255 * np.where(active_mask, cam_norm, 0.0)), cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet_map, cv2.COLOR_BGR2RGB)
        overlay = np.where(active_mask[..., None], (0.4 * img_rgb + 0.6 * jet_rgb).astype(np.uint8), img_rgb.copy())

        if gt_file and os.path.exists(gt_file):
            gt_gray = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            gt_gray = cv2.resize(gt_gray, (384, 384), interpolation=cv2.INTER_NEAREST)
            overlay[gt_gray > 0] = [0, 255, 0]
            contours, _ = cv2.findContours((gt_gray > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (max(0, x - 10), max(0, y - 10)), (min(384, x + w + 10), min(384, y + h + 10)), (0, 0, 0), 2)
        return overlay

    def execute(self, image_id: str, output_path: str, threshold: float = 0.6) -> None:
        """Executes inference for both models and exports 2x4 comparison figure."""
        img_file = self.find_file(f"data/**/{image_id}.jpg")
        if not img_file:
            print(f"Error: image {image_id}.jpg not found.")
            return

        raw_rgb = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        img_384 = cv2.resize(raw_rgb, (384, 384))
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            bl_cams = self.baseline_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()
            olap_cams = self.olap_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for col_idx, (code, title) in enumerate(zip(self.LESION_NAMES, self.LESION_FULL_NAMES)):
            gt_path = self.find_file(f"data/**/{code}/{image_id}.tif")
            bl_view = self.render_overlay(img_384, bl_cams[col_idx], gt_path, threshold)
            olap_view = self.render_overlay(img_384, olap_cams[col_idx], gt_path, threshold)

            axes[0, col_idx].imshow(bl_view)
            axes[0, col_idx].set_title(f"Baseline: {title}", fontsize=12, weight="bold", color="navy")
            axes[0, col_idx].axis("off")

            axes[1, col_idx].imshow(olap_view)
            axes[1, col_idx].set_title(f"OLAP (Previous): {title}", fontsize=12, weight="bold", color="darkred")
            axes[1, col_idx].axis("off")

        plt.suptitle(f"4-Lesion Concept Comparison (Case: {image_id}, Threshold={threshold})", fontsize=16, weight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"2x4 comparison figure successfully saved to: {output_path}")


if __name__ == "__main__":
    bl_path = sys.argv[1] if len(sys.argv) > 1 else "log/milvt_baseline_fold_0/version_0/checkpoints/epoch=11-step=1020.ckpt"
    olap_path = sys.argv[2] if len(sys.argv) > 2 else "log/milvt_olap_fold_0/version_6/checkpoints/epoch=13-step=1190.ckpt"
    target_case = sys.argv[3] if len(sys.argv) > 3 else "007-5470-300"
    out_file = sys.argv[4] if len(sys.argv) > 4 else f"comparison_4lesions_{target_case}.png"
    visualizer = LesionComparisonVisualizer(bl_path, olap_path)
    visualizer.execute(target_case, out_file)
