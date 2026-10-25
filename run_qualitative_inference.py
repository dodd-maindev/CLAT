"""Executes qualitative clinical grid inference across models and doctor ground truth."""

import glob
import os
import sys
from typing import List
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))
from clat.model import CLAT  # noqa: E402
from clat.visualization import CheckpointManager, GroundTruthProvider, QualitativeRenderer  # noqa: E402


class ClinicalGridInferencePipeline:
    """Orchestrates generation of publication-style clinical comparison grids."""

    def __init__(self, out_dir: str = "qualitative_results") -> None:
        """Initializes device, renderer, and ground-truth provider."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = out_dir
        self.renderer = QualitativeRenderer()
        self.gt_provider = GroundTruthProvider()
        os.makedirs(self.out_dir, exist_ok=True)

    def extract_cams(self, model: CLAT, img_384: np.ndarray) -> np.ndarray:
        """Runs forward pass with attention extraction and returns 4-channel CAMs."""
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(tensor, return_attn=True)
            return output.cams[0].cpu().numpy()

    def process_case(self, case_id: str, img_path: str, model_runs: List[dict], out_path: str) -> None:
        """Renders multi-row clinical grid comparing Doctor GT against model CAMs."""
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            return
        img_384 = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (384, 384))
        gt_masks = self.gt_provider.get_lesion_masks(case_id, 384, 384)

        for run in model_runs:
            if "cams" not in run:
                run["cams"] = self.extract_cams(run["model"], img_384)

        self.renderer.render_case_grid(img_384, case_id, gt_masks, model_runs, out_path)


def find_target_cases() -> List[dict]:
    """Finds available clinical cases with ground truth across DDR and FGADR."""
    priority_ids = ["007-4250-200", "007-2852-100", "007-7235-400", "0000_1", "0003_3", "0007_2"]
    found = []
    for cid in priority_ids:
        matches = glob.glob(f"data/**/{cid}.jpg", recursive=True) + glob.glob(f"data/**/{cid}.png", recursive=True)
        if matches:
            found.append({"id": cid, "path": matches[0]})
    if not found:
        all_imgs = glob.glob("data/**/*.png", recursive=True) + glob.glob("data/**/*.jpg", recursive=True)
        for p in all_imgs[:4]:
            found.append({"id": os.path.splitext(os.path.basename(p))[0], "path": p})
    return found


if __name__ == "__main__":
    search_dirs = ["/content/drive/MyDrive/FGADR_MILVT_Checkpoints", "/content/CLAT/log", "log"]
    manager = CheckpointManager(search_dirs)
    all_folds = manager.find_all_folds()
    if not all_folds:
        print("[Error] No fold checkpoints found.")
        sys.exit(1)

    pipeline = ClinicalGridInferencePipeline("qualitative_results")
    cases = find_target_cases()
    print(f"🎯 Evaluating {len(cases)} clinical cases across {len(all_folds)} folds.")

    for fold_num, ckpt_path in sorted(all_folds.items()):
        print(f"\n🔬 [Fold {fold_num}] Loading: {ckpt_path}")
        model = CLAT.load_from_checkpoint(ckpt_path, strict=False).to(pipeline.device).eval()
        fold_dir = os.path.join(pipeline.out_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        for case in cases:
            runs = [{"label": f"OLAP (Fold {fold_num})", "color": "navy", "model": model}]
            out_file = os.path.join(fold_dir, f"clinical_grid_{case['id']}.png")
            pipeline.process_case(case["id"], case["path"], runs, out_file)
    print("\n🎉 Clinical grid generation completed successfully!")
