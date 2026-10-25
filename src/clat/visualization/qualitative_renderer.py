"""Qualitative interpretability renderer for retinal fundus disease and lesion heatmaps."""

from typing import List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np


class QualitativeRenderer:
    """Renders side-by-side qualitative figures aligning ground truth with concept heatmaps."""

    def __init__(self, colormap: int = cv2.COLORMAP_JET, alpha: float = 0.5) -> None:
        """Initializes colormap and blending alpha factor."""
        self.colormap = colormap
        self.alpha = alpha

    def blend_heatmap(self, image_rgb: np.ndarray, cam_2d: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """Blends 2D normalized activation map with original retinal image."""
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam_2d, (w, h))
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        cam_masked = np.where(cam_norm >= threshold, cam_norm, 0.0)
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_masked), self.colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        active = (cam_masked > 0)[..., None]
        blended = np.where(active, (1.0 - self.alpha) * image_rgb + self.alpha * heatmap_rgb, image_rgb)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def render_ground_truth(self, image_rgb: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Overlays doctor ground-truth lesion masks with distinctive color palette."""
        if mask is None or mask.sum() == 0:
            return image_rgb.copy()
        canvas = image_rgb.copy()
        colors = [(255, 230, 0), (255, 40, 40), (0, 240, 255), (240, 50, 240)]
        for i, color in enumerate(colors):
            if mask.shape[-1] > i and mask[..., i].max() > 0:
                bin_m = cv2.dilate((mask[..., i] > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                for c in range(3):
                    canvas[..., c] = np.where(bin_m > 0, int(0.3 * canvas[..., c] + 0.7 * color[c]), canvas[..., c])
        return canvas

    def render_patient_case(
        self,
        image_rgb: np.ndarray,
        cams: np.ndarray,
        lesion_names: List[str],
        lesion_probs: List[float],
        true_disease: str,
        pred_disease: str,
        disease_probs: List[float],
        disease_names: List[str],
        save_path: str,
        gt_mask: Optional[np.ndarray] = None,
        gt_lesions: Optional[List[str]] = None,
    ) -> None:
        """Composes publication-grade 6-panel qualitative figure for a single patient."""
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
        gt_img = self.render_ground_truth(image_rgb, gt_mask)
        axes[0].imshow(gt_img)
        les_str = f"\nLesions: {', '.join(gt_lesions)}" if gt_lesions else ""
        axes[0].set_title(f"Ground Truth: {true_disease}{les_str}", fontsize=11, fontweight="bold", color="darkgreen")
        axes[0].axis("off")

        colors = ["#2ecc71" if d == pred_disease else "#95a5a6" for d in disease_names]
        axes[1].barh(range(len(disease_names)), disease_probs, color=colors)
        axes[1].set_yticks(range(len(disease_names)))
        axes[1].set_yticklabels(disease_names, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.0)
        axes[1].set_title(f"Prediction: {pred_disease}\n(Conf: {max(disease_probs):.1%})", fontsize=11, fontweight="bold")
        axes[1].grid(axis="x", linestyle="--", alpha=0.5)

        for idx, (name, prob) in enumerate(zip(lesion_names, lesion_probs)):
            ax = axes[idx + 2]
            overlay = self.blend_heatmap(image_rgb, cams[idx])
            ax.imshow(overlay)
            status_color = "red" if prob >= 0.5 else "black"
            ax.set_title(f"Concept: {name}\nProb: {prob:.1%}", fontsize=11, fontweight="bold", color=status_color)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
