"""Renders clear clinical comparison grids explicitly distinguishing CLAT vs OLAP."""

import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from clat.model import CLAT


class ClinicalHeatmapRenderer:
    """Renders 3x5 clinical comparison grids explicitly labeling CLAT and OLAP."""

    LESION_NAMES = ["EX", "HE", "MA", "SE"]

    def __init__(self, clat_checkpoint: str, olap_checkpoint: str, device: str = "cuda") -> None:
        """Loads both CLAT baseline and OLAP models into memory."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clat_model = CLAT.load_from_checkpoint(clat_checkpoint).to(self.device).eval()
        self.olap_model = CLAT.load_from_checkpoint(olap_checkpoint).to(self.device).eval()

    def find_file(self, pattern: str) -> str:
        """Finds first matching filepath for given glob pattern."""
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else ""

    def render_ground_truth(self, raw_rgb: np.ndarray, mask_path: str) -> np.ndarray:
        """Renders vivid green GT mask on black canvas with retina outline."""
        canvas = np.zeros_like(raw_rgb)
        fundus_mask = (raw_rgb.mean(axis=-1) > 15).astype(np.uint8)
        cnts, _ = cv2.findContours(fundus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (45, 45, 65), 1)
        if mask_path and os.path.exists(mask_path):
            gt = cv2.resize(cv2.imread(mask_path, 0), (384, 384), interpolation=cv2.INTER_NEAREST)
            if gt.max() > 0:
                dilated = cv2.dilate((gt > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                canvas[dilated > 0] = [0, 255, 0]
        return canvas

    def blend_jet(self, raw_rgb: np.ndarray, cam_2d: np.ndarray) -> np.ndarray:
        """Blends continuous JET colormap over raw fundus using 50-50 ratio."""
        cam_res = cv2.resize(cam_2d, (384, 384))
        norm = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        jet = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        return np.clip(0.5 * raw_rgb.astype(np.float32) + 0.5 * jet.astype(np.float32), 0, 255).astype(np.uint8)

    def execute_single(self, image_id: str, out_path: str) -> None:
        """Generates 3x5 comparison figure for a single case."""
        img_path = self.find_file(f"data/**/{image_id}.jpg")
        if not img_path:
            print(f"Skipping {image_id}: image not found.")
            return

        raw_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_384 = cv2.resize(raw_rgb, (384, 384))
        norm_t = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm_t).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            clat_cams = self.clat_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()
            olap_cams = self.olap_model(tensor, return_attn=True).cams.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(3, 5, figsize=(22, 13))
        axes[0, 0].imshow(img_384)
        axes[0, 0].set_title(f"Original Fundus\nID: {image_id}", fontsize=11, weight="bold")
        axes[1, 0].imshow(img_384)
        axes[1, 0].set_title("MODEL: CLAT\n(Baseline MIL-VT)", fontsize=11, weight="bold", color="darkred")
        axes[2, 0].imshow(img_384)
        axes[2, 0].set_title("MODEL: OLAP\n(Proposed V7)", fontsize=11, weight="bold", color="blue")

        for col, code in enumerate(self.LESION_NAMES, start=1):
            gt_file = self.find_file(f"data/**/{code}/{image_id}.tif")
            axes[0, col].imshow(self.render_ground_truth(img_384, gt_file))
            axes[0, col].set_title(f"Doctor GT: {code}", fontsize=11, weight="bold", color="lime")
            axes[1, col].imshow(self.blend_jet(img_384, clat_cams[col - 1]))
            axes[1, col].set_title(f"CLAT (Baseline): {code}", fontsize=11, weight="bold", color="maroon")
            axes[2, col].imshow(self.blend_jet(img_384, olap_cams[col - 1]))
            axes[2, col].set_title(f"OLAP (Proposed): {code}", fontsize=11, weight="bold", color="navy")

        for r in range(3):
            for c in range(5):
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    clat_cp = sys.argv[1] if len(sys.argv) > 1 else "log/milvt_baseline_fold_0/version_0/checkpoints/epoch=11-step=1020.ckpt"
    olap_cp = sys.argv[2] if len(sys.argv) > 2 else "log/milvt_olap_fold_0/version_7/checkpoints/epoch=11-step=1020.ckpt"
    cases_arg = sys.argv[3] if len(sys.argv) > 3 else "007-2852-100,007-4250-200,007-7235-400"
    renderer = ClinicalHeatmapRenderer(clat_cp, olap_cp)
    for cid in [c.strip() for c in cases_arg.split(",") if c.strip()]:
        renderer.execute_single(cid, f"comparison_clat_vs_olap_{cid}.png")
