"""Unified OLAP concept head integrating GeM pooling, projection, and dynamic gating."""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .concept_projector import CalibratedConceptProjector
from .convex_gate import ConvexDynamicGate
from .gem_pooling import GeneralizedMeanPooling2d
from .orthogonal_loss import OrthogonalSubspaceLoss


class OrthogonalAdaptivePoolingConceptHead(nn.Module):
    """OLAP concept head combining GeM spatial pooling, projection, and orthogonal loss."""

    def __init__(
        self,
        embed_dim: int,
        num_concepts: int,
        initial_power: float = 3.0,
        ortho_weight: float = 0.1,
    ) -> None:
        """Initializes the OLAP concept head.

        Args:
            embed_dim: Token embedding dimension.
            num_concepts: Number of concept categories.
            initial_power: Initial exponent for GeM pooling.
            ortho_weight: Weight for orthogonal subspace penalty.
        """
        super().__init__()
        self.spatial_conv = nn.Conv2d(embed_dim, num_concepts, kernel_size=1)
        self.gem_pooling = GeneralizedMeanPooling2d(num_concepts, initial_power=initial_power)
        self.concept_projector = CalibratedConceptProjector(embed_dim, num_concepts)
        self.dynamic_gate = ConvexDynamicGate(num_concepts)
        self.orthogonal_regularizer = OrthogonalSubspaceLoss(loss_weight=ortho_weight)

    def forward(
        self, patch_tokens: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward concept logit computation and regularizer estimation.

        Args:
            patch_tokens: Spatial features of shape (B, embed_dim, H, W).
            lesion_tokens: Output tokens of shape (B, num_concepts, embed_dim).

        Returns:
            Tuple of (fused_logits, local_logits, global_logits, ortho_loss).
        """
        # Local spatial concept branch via GeM pooling
        spatial_maps = self.spatial_conv(patch_tokens)
        local_logits = self.gem_pooling(spatial_maps)

        # Global concept token branch via calibrated projection
        global_logits = self.concept_projector(lesion_tokens)

        # Convex dynamic fusion
        fused_logits = self.dynamic_gate(local_logits, global_logits)

        # Representation disentanglement penalty
        ortho_loss = self.orthogonal_regularizer(lesion_tokens)

        return fused_logits, local_logits, global_logits, ortho_loss
