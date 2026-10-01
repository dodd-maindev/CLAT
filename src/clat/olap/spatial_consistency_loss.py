"""Spatial consistency regularization with label masking and diversity."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Aligns OLAP spatial features with attention maps using label masking."""

    def __init__(
        self,
        loss_weight: float = 0.02,
        diversity_weight: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """Initializes the spatial consistency regularization module.

        Args:
            loss_weight: Scaling factor for spatial alignment loss.
            diversity_weight: Scaling factor for spatial inter-concept separation.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.diversity_weight = diversity_weight
        self.eps = eps

    def forward(
        self,
        spatial_maps: torch.Tensor,
        attention_maps: torch.Tensor,
        lesion_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes label-masked cosine alignment and concept separation.

        Args:
            spatial_maps: OLAP spatial activations of shape (B, C, H, W).
            attention_maps: Reshaped attention maps of shape (B, C, H, W).
            lesion_labels: Ground truth lesion presence of shape (B, C).

        Returns:
            Scalar spatial consistency loss.
        """
        batch_size, num_concepts = spatial_maps.shape[:2]
        spatial_activated = F.relu(spatial_maps)

        spatial_flat = spatial_activated.flatten(start_dim=2)
        attention_flat = attention_maps.flatten(start_dim=2)

        spatial_norm = F.normalize(spatial_flat, p=2, dim=-1, eps=self.eps)
        attention_norm = F.normalize(attention_flat, p=2, dim=-1, eps=self.eps)

        # Cosine similarity per concept per sample: (B, C)
        cosine_similarity = (spatial_norm * attention_norm).sum(dim=-1)
        per_concept_error = 1.0 - cosine_similarity

        if lesion_labels is not None:
            valid_mask = lesion_labels.float()
            total_active = valid_mask.sum()
            if total_active > 0:
                alignment_loss = (per_concept_error * valid_mask).sum() / (
                    total_active + self.eps
                )
            else:
                alignment_loss = torch.tensor(0.0, device=spatial_maps.device)
        else:
            alignment_loss = per_concept_error.mean()

        # Inter-concept spatial separation penalty
        if num_concepts > 1:
            gram_matrix = torch.bmm(spatial_norm, spatial_norm.transpose(1, 2))
            identity = torch.eye(
                num_concepts, device=spatial_maps.device
            ).unsqueeze(0)
            off_diagonal = gram_matrix - identity
            diversity_loss = (off_diagonal**2).sum(dim=(-1, -2)).mean() / (
                num_concepts * (num_concepts - 1)
            )
        else:
            diversity_loss = torch.tensor(0.0, device=spatial_maps.device)

        total_loss = (
            self.loss_weight * alignment_loss
            + self.diversity_weight * diversity_loss
        )
        return total_loss
