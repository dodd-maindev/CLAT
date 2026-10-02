"""Renders publication-style 3x5 comparison grid across GT, Baseline, and OLAP."""

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from clat.model import CLAT


class ClinicalHeatmapRenderer:
    """Renders standardized 3x5 clinical heatmap grid with smooth JET blending."""

    LESION_NAMES = ["EX", "HE", "MA", "SE"]

    def __init__(self, baseline_checkpoint: str, olap_checkpoint: str, device: str = "cuda") -> None:
        """Loads both neural networks onto specified device."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.baseline_model = CLAT.load_from_checkpoint(baseline_checkpoint).to(self.device).eval()
        self.olap_model = CLAT.load_from_checkpoint(olap_checkpoint).to(self.device).eval()

    def find_file(self, pattern: str) -> str:
        """Finds first matching filepath for given glob pattern."""
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else ""

    def render_ground_truth(self, raw_rgb: np.ndarray, mask_path: str) -> np.ndarray:
        """Overlays raw green mask pixels directly on original fundus without bounding boxes."""
        canvas = raw_rgb.copy()
        if mask_path and os.path.exists(mask_path):
            gt_mask = cv2.resize(cv2.imread(mask_path, 0), (384, 384), interpolation=cv2.INTER_NEAREST)
            canvas[gt_mask > 0] = [0, 255, 0]
        return canvas

    def blend_jet_heatmap(self, raw_rgb: np.ndarray, cam_2d: np.ndarray) -> np.ndarray:
        """Blends raw fundus with continuous JET colormap using publication ratio."""
        cam_res = cv2.resize(cam_2d, (384, 384))
        cam_norm = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        jet_map = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        jet_rgb = cv2.cvtColor(jet_map, cv2.COLOR_BGR2RGB)
        blended = 0.5 * raw_rgb.astype(np.float32) + 0.5 * jet_rgb.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def execute(self, image_id: str, output_path: str) -> None:
        """Executes dual inference and saves 3x5 clinical comparison figure."""
        img_path = self.find_file(f"data/**/{image_id}.jpg")
        if not img_path:
            print(f"Error: image {image_id}.jpg not found.")
            return

        raw_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_384 = cv2.resize(raw_rgb, (384, 384))
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            bl_cams = self.baseline_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()
            olap_cams = self.olap_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        axes[0, 0].imshow(img_384)
        axes[0, 0].set_title(f"Original: {image_id}", fontsize=11, weight="bold")
        axes[1, 0].imshow(img_384)
        axes[1, 0].set_title("Baseline (MIL-VT)", fontsize=11, weight="bold", color="red")
        axes[2, 0].imshow(img_384)
        axes[2, 0].set_title("V4: Unsync (Old)", fontsize=11, weight="bold", color="blue")

        for col, code in enumerate(self.LESION_NAMES, start=1):
            gt_file = self.find_file(f"data/**/{code}/{image_id}.tif")
            axes[0, col].imshow(self.render_ground_truth(img_384, gt_file))
            axes[0, col].set_title(f"GT: {code} (Active)", fontsize=11, weight="bold", color="lime")
            axes[1, col].imshow(self.blend_jet_heatmap(img_384, bl_cams[col - 1]))
            axes[1, col].set_title(f"Baseline: {code}", fontsize=11, weight="bold")
            axes[2, col].imshow(self.blend_jet_heatmap(img_384, olap_cams[col - 1]))
            axes[2, col].set_title(f"V4: {code}", fontsize=11, weight="bold")

        for r in range(3):
            for c in range(5):
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Publication-style 3x5 figure saved to: {output_path}")


if __name__ == "__main__":
    bl_ckpt = sys.argv[1] if len(sys.argv) > 1 else "log/milvt_baseline_fold_0/version_0/checkpoints/epoch=11-step=1020.ckpt"
    olap_ckpt = sys.argv[2] if len(sys.argv) > 2 else "log/milvt_olap_fold_0/version_4/checkpoints/epoch=12-step=1105.ckpt"
    case_id = sys.argv[3] if len(sys.argv) > 3 else "007-5470-300"
    target_out = sys.argv[4] if len(sys.argv) > 4 else f"comparison_3x5_{case_id}.png"
    ClinicalHeatmapRenderer(bl_ckpt, olap_ckpt).execute(case_id, target_out)
