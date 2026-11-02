"""Calibrated concept fusion head preserving deep Transformer attention maps."""

from typing import Tuple
import torch
import torch.nn as nn

from .concept_projector import CalibratedConceptProjector
from .convex_gate import ConvexDynamicGate
from .gem_pooling import GeneralizedMeanPooling2d
from .orthogonal_loss import OrthogonalSubspaceLoss


class OrthogonalAdaptivePoolingConceptHead(nn.Module):
    """Calibrated concept head fusing local spatial pooling and global tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_concepts: int,
        initial_power: float = 3.0,
        ortho_weight: float = 0.05,
        pooling_type: str = "lse",
        use_projector_layernorm: bool = True,
        use_dynamic_gate: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the calibrated concept head.

        Args:
            embed_dim: Token embedding dimension.
            num_concepts: Number of concept categories.
            initial_power: Initial exponent for GeM pooling.
            ortho_weight: Weight for orthogonal subspace penalty.
            pooling_type: Spatial pooling method ("lse" vs "gap").
            use_projector_layernorm: Whether to apply LayerNorm in concept projector.
            use_dynamic_gate: Whether to use learnable gating weights.
        """
        super().__init__()
        self.gem_pooling = GeneralizedMeanPooling2d(
            num_concepts, initial_power=initial_power, pooling_type=pooling_type
        )
        self.concept_projector = CalibratedConceptProjector(
            embed_dim, num_concepts, use_norm=use_projector_layernorm
        )
        self.dynamic_gate = ConvexDynamicGate(
            num_concepts, use_dynamic_gate=use_dynamic_gate
        )
        self.orthogonal_regularizer = OrthogonalSubspaceLoss(loss_weight=ortho_weight)

    def forward(
        self, local_spatial_maps: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs calibrated concept logit fusion on spatial features and global tokens.

        Args:
            local_spatial_maps: Spatial features (B, num_concepts, H, W).
            lesion_tokens: Output concept tokens (B, num_concepts, embed_dim).

        Returns:
            Tuple of (fused_logits, local_logits, global_logits, ortho_loss, spatial_maps).
        """
        local_logits = self.gem_pooling(local_spatial_maps)
        global_logits = self.concept_projector(lesion_tokens)
        fused_logits = self.dynamic_gate(local_logits, global_logits)
        ortho_loss = self.orthogonal_regularizer(lesion_tokens)

        return fused_logits, local_logits, global_logits, ortho_loss, local_spatial_maps
