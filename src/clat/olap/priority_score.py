"""Priority scoring and ordinal threshold calibration for clinical evaluation."""

from typing import Any, Dict, Optional, Sequence
import numpy as np
import torch


class PriorityScoreCalculator:
    """Calculates prioritized composite score for medical checkpoint selection."""

    def __init__(
        self,
        weight_kappa: float = 0.35,
        weight_auc: float = 0.25,
        weight_sensitivity: float = 0.15,
        weight_cpt_f1: float = 0.09,
        weight_cpt_auc: float = 0.06,
        weight_cpt_acc: float = 0.05,
        weight_specificity: float = 0.05,
    ) -> None:
        """Initializes weights reflecting strict clinical diagnostic hierarchy."""
        self.weights = {
            "kappa": weight_kappa,
            "auc": weight_auc,
            "sensitivity": weight_sensitivity,
            "cpt_f1": weight_cpt_f1,
            "cpt_auc": weight_cpt_auc,
            "cpt_acc": weight_cpt_acc,
            "specificity": weight_specificity,
        }

    def compute(self, metrics: Dict[str, Any]) -> torch.Tensor:
        """Computes weighted priority score across available validation metrics."""
        score = torch.tensor(0.0)
        total_weight = 0.0
        for name, weight in self.weights.items():
            if name in metrics:
                val = metrics[name]
                tensor_val = val if isinstance(val, torch.Tensor) else torch.tensor(val)
                score = score + weight * tensor_val.cpu()
                total_weight += weight
        return score / max(total_weight, 1e-6)


class ContinuousSeverityCalibrator:
    """Calculates expected disease severity grade from class probabilities."""

    def __init__(self, num_classes: int = 5) -> None:
        """Initializes class ranks [0, 1, ..., num_classes - 1]."""
        self.num_classes = num_classes
        self.ranks = torch.arange(num_classes, dtype=torch.float32)

    def compute_expected_severity(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Computes expected continuous grade s = sum(c * p_c)."""
        ranks = self.ranks.to(probabilities.device)
        return torch.sum(probabilities * ranks, dim=-1)

    def discretize_by_thresholds(
        self, severity: torch.Tensor, thresholds: Optional[Sequence[float]] = None
    ) -> torch.Tensor:
        """Discretizes continuous severity into ordinal classes via threshold cutoffs."""
        if thresholds is None:
            thresholds = [0.5, 1.5, 2.5, 3.5]
        classes = torch.zeros_like(severity, dtype=torch.long)
        for idx, cutoff in enumerate(thresholds):
            classes = torch.where(severity >= cutoff, idx + 1, classes)
        return classes
