"""CLI runner to extract failure cases and generate visualizations from checkpoints."""

import argparse
import glob
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("src"))
from clat.ablation.extract_failure_cases import FailureCaseExtractor
from clat.dataset import FGADDRDataModule
from clat.model import CLAT


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Extract failure cases from a fold checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt).")
    parser.add_argument("--data_config", type=str, default="configs/data/FGADDR.yaml", help="Data yaml.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_extraction() -> None:
    """Executes failure analysis on the validation/test dataloader."""
    args = parse_args()
    print(f"Loading checkpoint: {args.ckpt}")
    model = CLAT.load_from_checkpoint(args.ckpt).to(args.device)
    model.eval()

    dm = FGADDRDataModule(config_path=args.data_config, fold_num=args.fold)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    all_y_true, all_y_pred_probs = [], []
    all_cpt_true, all_cpt_probs = [], []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating Fold {args.fold}"):
            imgs = batch["image"].to(args.device)
            labels = batch["disease_label"].to(args.device)
            cpts = batch["lesion_label"].to(args.device)
            ids = batch.get("id", [str(i) for i in range(len(imgs))])

            outputs = model(imgs)
            disease_logits = outputs.get("disease_logits", outputs.get("logits"))
            cpt_logits = outputs.get("lesion_logits", outputs.get("concept_logits"))

            all_y_true.append(labels.cpu())
            all_y_pred_probs.append(torch.softmax(disease_logits, dim=-1).cpu())
            all_cpt_true.append(cpts.cpu())
            all_cpt_probs.append(torch.sigmoid(cpt_logits).cpu() if cpt_logits is not None else torch.zeros_like(cpts.cpu()))
            all_ids.extend(ids)

    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred_probs, dim=0)
    cpt_true = torch.cat(all_cpt_true, dim=0)
    cpt_pred = torch.cat(all_cpt_probs, dim=0)

    extractor = FailureCaseExtractor()
    failures = extractor.compute_discrepancies(y_true, y_pred, cpt_true, cpt_pred, all_ids)
    df = extractor.save_categorized_report(failures, filename=f"failure_cases_fold_{args.fold}.csv")

    print("\n--- Failure Category Summary ---")
    print(df["Category"].value_counts())
    print(f"\nTop 5 representative error cases for paper visualization:")
    print(df[["Image_ID", "True_Grade", "Pred_Grade", "Confidence", "Category"]].head(5).to_string(index=False))


if __name__ == "__main__":
    run_extraction()
