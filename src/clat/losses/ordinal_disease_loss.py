"""Hybrid Ordinal Disease Loss combining weighted cross entropy and Soft-QWK."""

from typing import Optional, Sequence
import torch
import torch.nn as nn
from clat.losses.soft_qwk_loss import SoftQuadraticWeightedKappaLoss


class OrdinalDiseaseLoss(nn.Module):
    """Hybrid loss optimizing both cross-entropy and quadratic weighted kappa."""

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[Sequence[float]] = None,
        label_smoothing: float = 0.05,
        qwk_weight: float = 0.5,
    ) -> None:
        """Initializes balanced cross entropy and soft quadratic weighted kappa."""
        super().__init__()
        self.label_smoothing = label_smoothing
        self.qwk_weight = qwk_weight
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights_tensor)
        else:
            self.class_weights = None
        self.soft_qwk = SoftQuadraticWeightedKappaLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes joint cross-entropy and ordinal quadratic kappa loss."""
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        qwk_loss = self.soft_qwk(logits, targets)
        return ce_loss + self.qwk_weight * qwk_loss
