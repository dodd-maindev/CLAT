"""Calibrated concept fusion head preserving deep Transformer attention maps."""

from typing import Tuple
import torch
import torch.nn as nn

from .concept_projector import CalibratedConceptProjector
from .convex_gate import ConvexDynamicGate
from .gem_pooling import GeneralizedMeanPooling2d


class OrthogonalAdaptivePoolingConceptHead(nn.Module):
    """Calibrated concept head fusing local spatial pooling and global tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_concepts: int,
        initial_power: float = 3.0,
        ortho_weight: float = 0.0,
        spatial_weight: float = 0.0,
    ) -> None:
        """Initializes the calibrated concept head.

        Args:
            embed_dim: Token embedding dimension.
            num_concepts: Number of concept categories.
            initial_power: Initial exponent for GeM pooling.
            ortho_weight: Unused parameter kept for interface compatibility.
            spatial_weight: Unused parameter kept for interface compatibility.
        """
        super().__init__()
        self.gem_pooling = GeneralizedMeanPooling2d(num_concepts, initial_power=initial_power)
        self.concept_projector = CalibratedConceptProjector(embed_dim, num_concepts)
        self.dynamic_gate = ConvexDynamicGate(num_concepts)
        self.spatial_consistency = None

    def forward(
        self, local_spatial_maps: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs calibrated concept logit fusion.

        Args:
            local_spatial_maps: Spatial features from backbone head (B, num_concepts, H, W).
            lesion_tokens: Output concept tokens (B, num_concepts, embed_dim).

        Returns:
            Tuple of (fused_logits, local_logits, global_logits, zero_loss, spatial_maps).
        """
        # Local spatial concept branch via GeM pooling on backbone features
        local_logits = self.gem_pooling(local_spatial_maps)

        # Global concept token branch via calibrated projection
        global_logits = self.concept_projector(lesion_tokens)

        # Convex dynamic fusion of local and global branches
        fused_logits = self.dynamic_gate(local_logits, global_logits)

        # Zero auxiliary loss to prevent attention distortion
        zero_loss = torch.tensor(0.0, device=lesion_tokens.device)

        return fused_logits, local_logits, global_logits, zero_loss, local_spatial_maps
