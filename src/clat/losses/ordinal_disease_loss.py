"""Hybrid Focal Ordinal Disease Loss combining weighted focal cross-entropy and Expected Grade distance."""

from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from clat.losses.soft_qwk_loss import SoftQuadraticWeightedKappaLoss


class OrdinalDiseaseLoss(nn.Module):
    """Focal-weighted ordinal loss optimizing both class cross-entropy and expected grade distance."""

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[Sequence[float]] = None,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.5,
        qwk_weight: float = 0.2,
        ordinal_weight: float = 1.0,
    ) -> None:
        """Initializes focal ordinal disease loss with class weights and expected grade loss."""
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.qwk_weight = qwk_weight
        self.ordinal_weight = ordinal_weight
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights_tensor)
        else:
            self.class_weights = None
        self.soft_qwk = SoftQuadraticWeightedKappaLoss(num_classes=num_classes)
        self.register_buffer("class_ranks", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal cross-entropy, continuous expected grade, and soft QWK loss."""
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probabilities)

        if targets.dim() == 1:
            target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
            targets_one_hot = F.one_hot(targets, self.num_classes).float()
            target_ranks = targets.float()
        else:
            target_probabilities = (probabilities * targets).sum(dim=-1)
            targets_one_hot = targets.float()
            target_ranks = (targets_one_hot * self.class_ranks).sum(dim=-1)

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

        expected_grade = (probabilities * self.class_ranks).sum(dim=-1)
        ord_loss = F.smooth_l1_loss(expected_grade, target_ranks, beta=0.5)

        total_loss = focal_loss + self.ordinal_weight * ord_loss
        if self.qwk_weight > 0.0:
            qwk_loss = self.soft_qwk(logits, targets)
            total_loss = total_loss + self.qwk_weight * qwk_loss

        return total_loss
