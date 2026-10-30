"""Ordinal Focal Loss module with distance-weighted penalty for graded disease stages."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalFocalLoss(nn.Module):
    """Computes ordinal focal loss penalizing distance between predicted and true stages."""

    def __init__(
        self,
        number_of_classes: int = 5,
        focal_gamma: float = 2.0,
        distance_power: float = 2.0,
        distance_loss_weight: float = 0.5,
        epsilon: float = 1e-8,
    ) -> None:
        """Initializes class count, focal gamma, and precomputed distance cost matrix."""
        super().__init__()
        self.number_of_classes = number_of_classes
        self.focal_gamma = focal_gamma
        self.distance_loss_weight = distance_loss_weight
        self.epsilon = epsilon

        ranks = torch.arange(number_of_classes, dtype=torch.float32)
        distance_matrix = torch.abs(ranks.unsqueeze(0) - ranks.unsqueeze(1)) ** distance_power
        distance_matrix = distance_matrix / ((number_of_classes - 1.0) ** distance_power)
        self.register_buffer("distance_matrix", distance_matrix)

    def forward(
        self,
        prediction_logits: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates combined focal and ordinal distance penalty loss.

        Args:
            prediction_logits: Raw unnormalized class scores of shape (batch_size, num_classes).
            ground_truth_labels: Integer class indices of shape (batch_size,).
            sample_weights: Optional per-sample weighting factor.

        Returns:
            Scalar tensor representing ordinal focal loss.
        """
        probabilities = F.softmax(prediction_logits, dim=-1)
        clipped_probabilities = torch.clamp(probabilities, self.epsilon, 1.0 - self.epsilon)

        one_hot_targets = F.one_hot(
            ground_truth_labels, num_classes=self.number_of_classes
        ).float()
        probability_at_target = (clipped_probabilities * one_hot_targets).sum(dim=-1)
        focal_modulation = torch.pow(1.0 - probability_at_target, self.focal_gamma)
        cross_entropy = -torch.log(probability_at_target)
        focal_loss = focal_modulation * cross_entropy

        batch_distance_cost = self.distance_matrix[ground_truth_labels]  # (B, C)
        distance_penalty = (probabilities * batch_distance_cost).sum(dim=-1)

        total_sample_loss = focal_loss + self.distance_loss_weight * distance_penalty
        if sample_weights is not None:
            total_sample_loss = total_sample_loss * sample_weights

        return total_sample_loss.mean()

