"""Differentiable Soft Quadratic Weighted Kappa (QWK) Loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftQuadraticWeightedKappaLoss(nn.Module):
    """Calculates differentiable Quadratic Weighted Kappa loss for ordinal targets."""

    def __init__(self, num_classes: int = 5, epsilon: float = 1e-7) -> None:
        """Initializes class count, quadratic weight matrix, and numerical epsilon."""
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        indices = torch.arange(num_classes, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(indices, indices, indexing="ij")
        weight_matrix = ((row_grid - col_grid) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("weight_matrix", weight_matrix)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes soft quadratic weighted kappa loss between logits and targets."""
        probabilities = F.softmax(logits, dim=-1)
        batch_size = targets.size(0)

        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, self.num_classes).float()
        else:
            targets_one_hot = targets.float()

        observed_matrix = torch.matmul(probabilities.t(), targets_one_hot) / batch_size

        pred_hist = probabilities.sum(dim=0, keepdim=True) / batch_size
        target_hist = targets_one_hot.sum(dim=0, keepdim=True) / batch_size
        expected_matrix = torch.matmul(pred_hist.t(), target_hist)

        numerator = torch.sum(self.weight_matrix * observed_matrix)
        denominator = torch.sum(self.weight_matrix * expected_matrix) + self.epsilon

        return numerator / denominator
