"""Runs qualitative inference pipeline on top-performing cross-validation folds."""

import glob
import os
import sys
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from clat.model import CLAT
from clat.visualization.qualitative_renderer import QualitativeRenderer


class QualitativeInferencePipeline:
    """Coordinates qualitative visualization across high-performing pretrain model folds."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Loads trained CLAT model onto active compute device."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLAT.load_from_checkpoint(checkpoint_path).to(self.device).eval()
        self.renderer = QualitativeRenderer(alpha=0.45)
        self.disease_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        self.lesion_names = ["EX", "HE", "MA", "SE"]

    def infer_and_visualize(self, image_path: str, output_path: str, true_disease: str = "Unknown") -> None:
        """Executes forward pass with attention extraction and renders qualitative comparison."""
        raw_bgr = cv2.imread(image_path)
        if raw_bgr is None:
            print(f"[Inference] Cannot read image: {image_path}")
            return
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        img_384 = cv2.resize(raw_rgb, (384, 384))
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor, return_attn=True)
            d_probs = output.disease_logits.softmax(dim=-1)[0].cpu().numpy().tolist()
            l_probs = torch.sigmoid(output.lesion_logits)[0].cpu().numpy().tolist()
            cams = output.cams[0].cpu().numpy()

        pred_disease = self.disease_names[int(np.argmax(d_probs))]
        self.renderer.render_patient_case(
            img_384, cams, self.lesion_names, l_probs,
            true_disease, pred_disease, d_probs, self.disease_names, output_path
        )


def find_top_checkpoint() -> Optional[str]:
    """Scans Drive and local directories to find checkpoints from top-performing folds."""
    priority_folds = [7, 4, 3, 1]
    search_roots = ["/content/drive/MyDrive/FGADR_MILVT_Checkpoints", "/content/CLAT/log", "log"]
    for root in search_roots:
        for f in priority_folds:
            matches = glob.glob(f"{root}/*fold_{f}*/**/*.ckpt", recursive=True)
            if matches:
                return sorted(matches)[-1]
    all_ckpts = glob.glob("**/*.ckpt", recursive=True)
    return all_ckpts[0] if all_ckpts else None


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else find_top_checkpoint()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "qualitative_results"
    os.makedirs(out_dir, exist_ok=True)
    if not ckpt or not os.path.exists(ckpt):
        print("[Error] No valid model checkpoint found.")
        sys.exit(1)
    print(f"🚀 [Qualitative Inference] Using top checkpoint: {ckpt}")
    pipeline = QualitativeInferencePipeline(ckpt)
    sample_images = glob.glob("data/FGADR/fundus_384/*.png")[:8]
    if not sample_images:
        sample_images = glob.glob("data/**/*.png", recursive=True) + glob.glob("data/**/*.jpg", recursive=True)
    for idx, path in enumerate(sample_images[:4]):
        b_name = os.path.splitext(os.path.basename(path))[0]
        out_file = os.path.join(out_dir, f"qualitative_case_{idx + 1}_{b_name}.png")
        pipeline.infer_and_visualize(path, out_file, true_disease=f"Case #{idx + 1}")
