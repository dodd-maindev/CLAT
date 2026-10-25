"""Runs qualitative inference pipeline on top-performing cross-validation folds."""

import glob
import os
import sys
import zipfile
from typing import Optional
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("src"))
from clat.model import CLAT  # noqa: E402
from clat.visualization.qualitative_renderer import QualitativeRenderer  # noqa: E402


class QualitativeInferencePipeline:
    """Coordinates qualitative visualization across high-performing pretrain model folds."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Loads trained CLAT model onto active compute device."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLAT.load_from_checkpoint(checkpoint_path).to(self.device).eval()
        self.renderer = QualitativeRenderer(alpha=0.45)
        self.diseases = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        self.lesions = ["EX", "HE", "MA", "SE"]

    def infer_and_visualize(self, img_path: str, out_path: str, gt_label: str = "Unknown") -> None:
        """Executes forward pass with attention extraction and renders qualitative comparison."""
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
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

        pred_label = self.diseases[int(np.argmax(d_probs))]
        self.renderer.render_patient_case(
            img_384, cams, self.lesions, l_probs,
            gt_label, pred_label, d_probs, self.diseases, out_path
        )


def find_top_checkpoint() -> Optional[str]:
    """Scans Drive and local directories to find checkpoints from top-performing folds."""
    drive_dir = "/content/drive/MyDrive/FGADR_MILVT_Checkpoints"
    zip_path = os.path.join(drive_dir, "FGADR_MILVT_Full_Logs.zip")
    if os.path.exists(zip_path) and not glob.glob(f"{drive_dir}/*fold_7*/**/*.ckpt", recursive=True):
        print(f"📦 [Archive] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(drive_dir)

    for root in [drive_dir, "/content/CLAT/log", "log"]:
        for f in [7, 4, 3, 1, 2, 6, 8, 9]:
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

    gt_map = {}
    if os.path.exists("data/annotation_FGADR_disease.csv"):
        df = pd.read_csv("data/annotation_FGADR_disease.csv")
        cols = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        for _, r in df.iterrows():
            gt_map[str(r["ID"])] = cols[int(np.argmax(r[cols].values))]

    imgs = glob.glob("data/FGADR/fundus_384/*.png")
    for idx, path in enumerate(imgs[:6]):
        case_id = os.path.splitext(os.path.basename(path))[0]
        out_file = os.path.join(out_dir, f"case_{idx + 1}_{case_id}.png")
        pipeline.infer_and_visualize(path, out_file, gt_label=gt_map.get(case_id, "Unknown"))

