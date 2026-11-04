"""Failure case extraction tool for analyzing model misclassifications."""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class MisclassifiedSample:
    """Represents a misclassified test instance."""
    image_id: str
    true_grade: int
    pred_grade: int
    confidence: float
    grade_error_magnitude: int
    true_concepts: List[int]
    pred_concepts: List[float]


class FailureCaseExtractor:
    """Extracts and categorizes misclassified samples from trained checkpoints."""

    def __init__(self, output_dir: str = "log/failure_analysis") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def compute_discrepancies(
        y_true: torch.Tensor,
        y_pred_probs: torch.Tensor,
        cpt_true: torch.Tensor,
        cpt_probs: torch.Tensor,
        image_ids: List[str],
    ) -> List[MisclassifiedSample]:
        """Identifies misclassified cases from batched predictions."""
        preds = torch.argmax(y_pred_probs, dim=-1)
        failures: List[MisclassifiedSample] = []

        for idx, img_id in enumerate(image_ids):
            t_grade = int(y_true[idx].item())
            p_grade = int(preds[idx].item())
            if t_grade != p_grade:
                conf = float(y_pred_probs[idx, p_grade].item())
                err = abs(t_grade - p_grade)
                failures.append(
                    MisclassifiedSample(
                        image_id=str(img_id),
                        true_grade=t_grade,
                        pred_grade=p_grade,
                        confidence=round(conf, 4),
                        grade_error_magnitude=err,
                        true_concepts=[int(v) for v in cpt_true[idx].tolist()],
                        pred_concepts=[round(float(v), 3) for v in cpt_probs[idx].tolist()],
                    )
                )
        return failures

    def save_categorized_report(
        self, samples: List[MisclassifiedSample], filename: str = "failure_cases.csv"
    ) -> pd.DataFrame:
        """Saves extracted failure cases into structured CSV categorized by clinical error types."""
        records = [
            {
                "Image_ID": s.image_id,
                "True_Grade": s.true_grade,
                "Pred_Grade": s.pred_grade,
                "Confidence": s.confidence,
                "Error_Delta": s.grade_error_magnitude,
                "True_Concepts": str(s.true_concepts),
                "Pred_Concepts": str(s.pred_concepts),
                "Category": self._classify_error_type(s.true_grade, s.pred_grade),
            }
            for s in samples
        ]
        df = pd.DataFrame(records)
        df.sort_values(by=["Error_Delta", "Confidence"], ascending=[False, False], inplace=True)
        out_path = os.path.join(self.output_dir, filename)
        df.to_csv(out_path, index=False)
        print(f"[Done] Saved {len(df)} failure cases to {out_path}")
        return df

    @staticmethod
    def _classify_error_type(true_grade: int, pred_grade: int) -> str:
        """Maps misclassification to clinical diagnostic error type."""
        delta = pred_grade - true_grade
        if delta == -1 and true_grade == 1:
            return "Under-diagnosis_Mild_Missed_MA"
        if delta == -1 and true_grade in [2, 3]:
            return "Borderline_NPDR_Under-grading"
        if delta == 1 and true_grade == 0:
            return "Over-diagnosis_Artifact_False_Positive"
        if delta == 1 and true_grade in [1, 2]:
            return "Borderline_NPDR_Over-grading"
        if abs(delta) >= 2:
            return "Severe_Cross-Grade_Error"
        return "Adjacent_Grade_Discrepancy"
