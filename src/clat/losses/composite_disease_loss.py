"""Composite disease loss orchestrator unifying cross entropy, ordinal focal, and QWK."""

import torch
import torch.nn as nn
from .ordinal_focal_loss import OrdinalFocalLoss
from .quadratic_weighted_kappa_loss import QuadraticWeightedKappaLoss


class CompositeDiseaseLoss(nn.Module):
    """Combines smoothed cross entropy, ordinal focal distance, and differentiable QWK."""

    def __init__(
        self,
        number_of_classes: int = 5,
        label_smoothing: float = 0.05,
        ordinal_weight: float = 0.5,
        kappa_weight: float = 0.3,
    ) -> None:
        """Initializes constituent losses and their balancing weights."""
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ordinal_loss = OrdinalFocalLoss(number_of_classes=number_of_classes)
        self.kappa_loss = QuadraticWeightedKappaLoss(number_of_classes=number_of_classes)
        self.ordinal_weight = ordinal_weight
        self.kappa_weight = kappa_weight

    def forward(
        self, prediction_logits: torch.Tensor, ground_truth_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculates combined multi-objective disease classification loss.

        Args:
            prediction_logits: Stage logits of shape (batch_size, num_classes).
            ground_truth_labels: Ground truth stage labels of shape (batch_size,).

        Returns:
            Weighted aggregate scalar loss tensor.
        """
        ce_loss = self.cross_entropy(prediction_logits, ground_truth_labels)
        ord_loss = self.ordinal_loss(prediction_logits, ground_truth_labels)
        qwk_loss = self.kappa_loss(prediction_logits, ground_truth_labels)

        return ce_loss + self.ordinal_weight * ord_loss + self.kappa_weight * qwk_loss

