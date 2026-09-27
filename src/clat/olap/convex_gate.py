"""Convex dynamic gating module for blending local and global concept logits."""

import torch
import torch.nn as nn


class ConvexDynamicGate(nn.Module):
    """Dynamically balances local and global concept logits via learned convex combination."""

    def __init__(self, num_concepts: int, initial_logit: float = 0.0) -> None:
        """Initializes the convex dynamic gate.

        Args:
            num_concepts: Number of concept categories (e.g. 4 for DR lesions).
            initial_logit: Initial value for logits (0.0 corresponds to 0.5 balanced weight).
        """
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((1, num_concepts), float(initial_logit)))

    @property
    def local_weights(self) -> torch.Tensor:
        """Returns the current sigmoid gating weights for local logits."""
        return torch.sigmoid(self.gate_logits)

    def forward(self, local_logits: torch.Tensor, global_logits: torch.Tensor) -> torch.Tensor:
        """Fuses local and global logits using learned convex weights.

        Args:
            local_logits: Local spatial logits of shape (batch_size, num_concepts).
            global_logits: Global token-derived logits of shape (batch_size, num_concepts).

        Returns:
            Fused logits of shape (batch_size, num_concepts).
        """
        weight_local = torch.sigmoid(self.gate_logits)
        return weight_local * local_logits + (1.0 - weight_local) * global_logits
