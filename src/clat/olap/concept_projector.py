"""Calibrated global concept projection module for lesion tokens."""

import torch
import torch.nn as nn


class CalibratedConceptProjector(nn.Module):
    """Calibrates and projects high-dimensional lesion tokens into logit scalars."""

    def __init__(self, embed_dim: int, num_concepts: int) -> None:
        """Initializes the calibrated concept projector.

        Args:
            embed_dim: Feature dimension of the token embeddings.
            num_concepts: Number of distinct concept classes.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        # Learnable projection vector per concept: shape (num_concepts, embed_dim)
        self.projection_weights = nn.Parameter(
            torch.randn(num_concepts, embed_dim) * (1.0 / (embed_dim ** 0.5))
        )
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, lesion_tokens: torch.Tensor) -> torch.Tensor:
        """Projects lesion tokens to concept logit values.

        Args:
            lesion_tokens: Output tokens of shape (batch_size, num_concepts, embed_dim).

        Returns:
            Global logits of shape (batch_size, num_concepts).
        """
        normed_tokens = self.norm(lesion_tokens)
        # Pointwise dot-product per concept: (B, C, D) * (1, C, D) -> sum over D
        logits = (normed_tokens * self.projection_weights.unsqueeze(0)).sum(dim=-1)
        return logits + self.bias.unsqueeze(0)
