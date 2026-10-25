"""Qualitative interpretability renderer for retinal fundus disease and lesion heatmaps."""

from typing import List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np


class QualitativeRenderer:
    """Renders side-by-side qualitative figures aligning original fundus with concept heatmaps."""

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
    ) -> None:
        """Composes publication-grade 6-panel qualitative figure for a single patient."""
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"Original Fundus\nGT: {true_disease}", fontsize=11, fontweight="bold", color="darkgreen")
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
        print(f"[Renderer] Qualitative figure successfully saved to: {save_path}")
