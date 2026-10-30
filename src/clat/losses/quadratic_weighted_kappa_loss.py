"""Differentiable Quadratic Weighted Kappa Loss for ordinal classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadraticWeightedKappaLoss(nn.Module):
    """Computes differentiable continuous approximation of Quadratic Weighted Kappa."""

    def __init__(self, number_of_classes: int = 5, epsilon: float = 1e-7) -> None:
        """Initializes class dimension and quadratic weight penalty matrix."""
        super().__init__()
        self.number_of_classes = number_of_classes
        self.epsilon = epsilon

        class_indices = torch.arange(number_of_classes, dtype=torch.float32)
        weight_matrix = (class_indices.unsqueeze(0) - class_indices.unsqueeze(1)) ** 2
        weight_matrix = weight_matrix / ((number_of_classes - 1.0) ** 2)
        self.register_buffer("weight_matrix", weight_matrix)

    def forward(
        self, prediction_logits: torch.Tensor, ground_truth_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculates differentiable QWK loss from batch predictions.

        Args:
            prediction_logits: Prediction logits of shape (batch_size, num_classes).
            ground_truth_labels: Target class labels of shape (batch_size,).

        Returns:
            Scalar tensor representing 1.0 - soft quadratic weighted kappa.
        """
        batch_size = prediction_logits.shape[0]
        predicted_probabilities = F.softmax(prediction_logits, dim=-1)
        one_hot_targets = F.one_hot(
            ground_truth_labels, num_classes=self.number_of_classes
        ).float()

        # Compute observed agreement matrix O
        observed_matrix = torch.matmul(
            predicted_probabilities.transpose(0, 1), one_hot_targets
        )

        # Compute expected agreement matrix E
        predicted_histograms = predicted_probabilities.sum(dim=0, keepdim=True)
        target_histograms = one_hot_targets.sum(dim=0, keepdim=True)
        expected_matrix = torch.matmul(
            predicted_histograms.transpose(0, 1), target_histograms
        ) / float(batch_size)

        # Normalize matrices to probability distributions
        observed_normalized = observed_matrix / float(batch_size)
        expected_normalized = expected_matrix / float(batch_size)

        numerator = torch.sum(self.weight_matrix * observed_normalized)
        denominator = torch.sum(self.weight_matrix * expected_normalized)

        return numerator / (denominator + self.epsilon)

