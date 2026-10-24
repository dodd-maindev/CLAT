"""Hybrid Focal Ordinal Disease Loss combining weighted focal cross-entropy and Soft-QWK."""

from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from clat.losses.soft_qwk_loss import SoftQuadraticWeightedKappaLoss


class OrdinalDiseaseLoss(nn.Module):
    """Focal-weighted ordinal loss optimizing both hard-sample cross-entropy and QWK."""

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[Sequence[float]] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 1.5,
        qwk_weight: float = 1.0,
    ) -> None:
        """Initializes focal ordinal disease loss with class weights and QWK module."""
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.qwk_weight = qwk_weight
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights_tensor)
        else:
            self.class_weights = None
        self.soft_qwk = SoftQuadraticWeightedKappaLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal cross-entropy and soft quadratic weighted kappa loss."""
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probabilities)

        if targets.dim() == 1:
            target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
            targets_one_hot = F.one_hot(targets, self.num_classes).float()
        else:
            target_probabilities = (probabilities * targets).sum(dim=-1)
            targets_one_hot = targets.float()

        focal_modulators = torch.pow(1.0 - target_probabilities.clamp(0.0, 1.0), self.focal_gamma)

        smoothed_targets = targets_one_hot * (1.0 - self.label_smoothing) + (
            self.label_smoothing / self.num_classes
        )
        cross_entropy_per_sample = -(smoothed_targets * log_probabilities).sum(dim=-1)

        if self.class_weights is not None and targets.dim() == 1:
            sample_class_weights = self.class_weights[targets]
            focal_loss = (sample_class_weights * focal_modulators * cross_entropy_per_sample).mean()
        else:
            focal_loss = (focal_modulators * cross_entropy_per_sample).mean()

        qwk_loss = self.soft_qwk(logits, targets)
        return focal_loss + self.qwk_weight * qwk_loss
