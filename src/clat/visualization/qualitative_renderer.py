"""Renders publication-grade single-row clinical heatmap panels."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


class QualitativeRenderer:
    """Renders clean 1-row by 5-column clinical heatmap panels."""

    LESION_NAMES = ["EX", "HE", "MA", "SE"]

    def blend_jet(self, raw_rgb: np.ndarray, cam_2d: np.ndarray) -> np.ndarray:
        """Blends continuous JET colormap over raw fundus using 50-50 ratio."""
        cam_res = cv2.resize(cam_2d, (384, 384))
        norm = (cam_res - cam_res.min()) / (cam_res.max() - cam_res.min() + 1e-8)
        jet = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        return np.clip(0.5 * raw_rgb.astype(np.float32) + 0.5 * jet.astype(np.float32), 0, 255).astype(np.uint8)

    def render_single_row(
        self,
        raw_rgb: np.ndarray,
        case_id: str,
        cams: np.ndarray,
        model_label: str,
        save_path: str,
    ) -> None:
        """Composes clean 1-row by 5-column figure displaying fundus and 4 lesion CAMs."""
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        axes[0].imshow(raw_rgb)
        axes[0].set_title(f"Original Fundus\nID: {case_id}", fontsize=11, weight="bold", color="navy")
        axes[0].axis("off")

        for col, code in enumerate(self.LESION_NAMES, start=1):
            cam = cams[col - 1] if len(cams) >= col else np.zeros((384, 384))
            axes[col].imshow(self.blend_jet(raw_rgb, cam))
            axes[col].set_title(f"{model_label}: {code}", fontsize=11, weight="bold", color="navy")
            axes[col].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Renderer] Saved qualitative panel: {save_path}")
