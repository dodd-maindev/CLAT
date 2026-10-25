"""Executes multi-fold qualitative inference pipeline across all validation folds."""

import os
import sys
from typing import List
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))
from clat.model import CLAT  # noqa: E402
from clat.visualization import CheckpointManager, GroundTruthProvider, QualitativeRenderer  # noqa: E402


class MultiFoldInferenceRunner:
    """Orchestrates qualitative inference and visual explanation across model folds."""

    def __init__(self, out_dir: str = "qualitative_results") -> None:
        """Initializes hardware device, renderer, and ground-truth provider."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = out_dir
        self.renderer = QualitativeRenderer(alpha=0.45)
        self.gt_provider = GroundTruthProvider()
        self.diseases = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        self.lesions = ["EX", "HE", "MA", "SE"]
        os.makedirs(self.out_dir, exist_ok=True)

    def run_image(self, model: CLAT, case: dict, out_path: str) -> None:
        """Runs forward pass with attention extraction and renders case figure."""
        raw_bgr = cv2.imread(case["path"])
        if raw_bgr is None:
            return
        img_384 = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (384, 384))
        norm = (img_384 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(tensor, return_attn=True)
            d_probs = output.disease_logits.softmax(dim=-1)[0].cpu().numpy().tolist()
            l_probs = torch.sigmoid(output.lesion_logits)[0].cpu().numpy().tolist()
            cams = output.cams[0].cpu().numpy()

        pred_label = self.diseases[int(np.argmax(d_probs))]
        gt_mask = self.gt_provider.get_raw_mask(case["id"])
        gt_les = self.gt_provider.get_lesions(case["id"])
        self.renderer.render_patient_case(
            img_384, cams, self.lesions, l_probs,
            case["grade"], pred_label, d_probs, self.diseases, out_path,
            gt_mask=gt_mask, gt_lesions=gt_les
        )

    def evaluate_fold(self, fold_idx: int, ckpt_path: str, cases: List[dict]) -> None:
        """Executes qualitative inference for a single fold across selected patient cases."""
        print(f"\n🔬 [Fold {fold_idx}] Loading checkpoint: {ckpt_path}")
        model = CLAT.load_from_checkpoint(ckpt_path, strict=False).to(self.device).eval()
        fold_out_dir = os.path.join(self.out_dir, f"fold_{fold_idx}")
        os.makedirs(fold_out_dir, exist_ok=True)

        for case in cases:
            save_path = os.path.join(fold_out_dir, f"case_{case['id']}_{case['grade']}.png")
            self.run_image(model, case, save_path)
        print(f"✅ [Fold {fold_idx}] Completed {len(cases)} qualitative figures in {fold_out_dir}")


def select_representative_cases(max_cases: int = 4) -> List[dict]:
    """Selects diverse patient cases spanning distinct clinical retinopathy grades."""
    gt_prov = GroundTruthProvider()
    grades = ["Mild", "Moderate", "Severe", "Proliferative"]
    selected = []
    for g in grades:
        for img_id, dis in gt_prov.disease_map.items():
            if dis == g:
                p = os.path.join("data", "FGADR", "fundus_384", f"{img_id}.png")
                if os.path.exists(p):
                    selected.append({"id": img_id, "path": p, "grade": g})
                    break
    return selected[:max_cases]


if __name__ == "__main__":
    search_dirs = ["/content/drive/MyDrive/FGADR_MILVT_Checkpoints", "/content/CLAT/log", "log"]
    manager = CheckpointManager(search_dirs)
    all_folds = manager.find_all_folds()
    if not all_folds:
        print("[Error] No fold checkpoints found.")
        sys.exit(1)

    print(f"📊 Discovered {len(all_folds)} folds: {list(all_folds.keys())}")
    test_cases = select_representative_cases(4)
    runner = MultiFoldInferenceRunner("qualitative_results")

    for fold_num, checkpoint in sorted(all_folds.items()):
        runner.evaluate_fold(fold_num, checkpoint, test_cases)
    print("\n🎉 Multi-fold qualitative inference finished successfully!")
