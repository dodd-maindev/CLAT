"""Renders publication-grade clinical heatmap grids matching the standard DDR format."""

from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np


class QualitativeRenderer:
    """Renders clinical comparison grids with neon-green GT masks and JET CAM heatmaps."""

    LESION_NAMES = ["EX", "HE", "MA", "SE"]

    def render_ground_truth(self, raw_rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Renders vivid neon-green GT mask on dark canvas with subtle retina contour."""
        canvas = np.zeros_like(raw_rgb)
        fundus_mask = (raw_rgb.mean(axis=-1) > 15).astype(np.uint8)
        cnts, _ = cv2.findContours(fundus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (45, 45, 65), 1)

        if mask is not None and mask.max() > 0:
            dilated = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            canvas[dilated > 0] = [0, 255, 0]
        return canvas

    def blend_jet(self, raw_rgb: np.ndarray, cam_2d: np.ndarray) -> np.ndarray:
        """Blends continuous JET colormap over raw fundus using 50-50 ratio."""
        cam_res = cv2.resize(cam_2d, (384, 384))
        norm = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        jet = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        return np.clip(0.5 * raw_rgb.astype(np.float32) + 0.5 * jet.astype(np.float32), 0, 255).astype(np.uint8)

    def render_case_grid(
        self,
        raw_rgb: np.ndarray,
        case_id: str,
        gt_masks: Dict[str, Optional[np.ndarray]],
        model_runs: List[Dict[str, any]],
        save_path: str,
    ) -> None:
        """Composes publication-grade N-row x 5-column grid across GT and model predictions."""
        num_rows = 1 + len(model_runs)
        fig, axes = plt.subplots(num_rows, 5, figsize=(22, 4.4 * num_rows))
        if num_rows == 1:
            axes = np.expand_dims(axes, 0)

        # Row 0: Doctor Ground Truth
        axes[0, 0].imshow(raw_rgb)
        axes[0, 0].set_title(f"Original Fundus\nID: {case_id}", fontsize=11, weight="bold")
        for col, code in enumerate(self.LESION_NAMES, start=1):
            mask = gt_masks.get(code)
            axes[0, col].imshow(self.render_ground_truth(raw_rgb, mask))
            axes[0, col].set_title(f"Doctor GT: {code}", fontsize=11, weight="bold", color="lime")

        # Rows 1..N: Model predictions
        for row_idx, run in enumerate(model_runs, start=1):
            label = run.get("label", f"Model {row_idx}")
            color = run.get("color", "navy")
            cams = run["cams"]
            axes[row_idx, 0].imshow(raw_rgb)
            axes[row_idx, 0].set_title(f"MODEL: {label}", fontsize=11, weight="bold", color=color)

            for col, code in enumerate(self.LESION_NAMES, start=1):
                cam = cams[col - 1] if len(cams) >= col else np.zeros((384, 384))
                axes[row_idx, col].imshow(self.blend_jet(raw_rgb, cam))
                axes[row_idx, col].set_title(f"{label}: {code}", fontsize=11, weight="bold", color=color)

        for r in range(num_rows):
            for c in range(5):
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Renderer] Saved publication grid to: {save_path}")
