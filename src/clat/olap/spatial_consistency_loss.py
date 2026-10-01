"""Spatial consistency regularization between OLAP spatial maps and attention maps."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Encourages OLAP spatial maps to align with transformer attention patterns."""

    def __init__(self, loss_weight: float = 0.02, eps: float = 1e-8) -> None:
        """Initializes the spatial consistency loss.

        Args:
            loss_weight: Multiplier for the consistency penalty.
            eps: Epsilon for numerical stability in normalization.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self, spatial_maps: torch.Tensor, attention_maps: torch.Tensor
    ) -> torch.Tensor:
        """Computes cosine alignment loss between spatial and attention maps.

        Args:
            spatial_maps: OLAP spatial conv output of shape (B, C, H, W).
            attention_maps: Reshaped attention weights of shape (B, C, H, W).

        Returns:
            Scalar consistency penalty encouraging spatial-attention alignment.
        """
        spatial_activated = F.relu(spatial_maps)

        # Flatten spatial dims: (B, C, H*W)
        spatial_flat = spatial_activated.flatten(start_dim=2)
        attention_flat = attention_maps.flatten(start_dim=2)

        # L2 normalize along spatial dimension
        spatial_norm = F.normalize(spatial_flat, p=2, dim=-1, eps=self.eps)
        attention_norm = F.normalize(attention_flat, p=2, dim=-1, eps=self.eps)

        # Cosine similarity per concept per batch: (B, C)
        cosine_similarity = (spatial_norm * attention_norm).sum(dim=-1)

        # Loss = 1 - mean cosine similarity (maximize alignment)
        alignment_loss = 1.0 - cosine_similarity.mean()

        return self.loss_weight * alignment_loss
