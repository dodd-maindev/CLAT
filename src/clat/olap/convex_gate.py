"""Convex dynamic gating module for blending local and global concept logits."""

import torch
import torch.nn as nn


class ConvexDynamicGate(nn.Module):
    """Balances local and global concept logits via learned or fixed convex fusion."""

    def __init__(
        self,
        num_concepts: int,
        initial_logit: float = 0.0,
        use_dynamic_gate: bool = True,
    ) -> None:
        """Initializes the convex dynamic gate.

        Args:
            num_concepts: Number of concept categories (e.g. 4 for DR lesions).
            initial_logit: Initial logit value (0.0 corresponds to 0.5 weight).
            use_dynamic_gate: Whether to learn dynamic weights or keep 50/50 balance.
        """
        super().__init__()
        self.use_dynamic_gate = use_dynamic_gate
        self.gate_logits = nn.Parameter(torch.full((1, num_concepts), float(initial_logit)))

    @property
    def local_weights(self) -> torch.Tensor:
        """Returns the current gating weights for local logits."""
        if not self.use_dynamic_gate:
            return torch.full_like(self.gate_logits, 0.5)
        return torch.sigmoid(self.gate_logits)

    def forward(self, local_logits: torch.Tensor, global_logits: torch.Tensor) -> torch.Tensor:
        """Fuses local and global logits using convex weights.

        Args:
            local_logits: Local spatial logits of shape (batch_size, num_concepts).
            global_logits: Global token-derived logits of shape (batch_size, num_concepts).

        Returns:
            Fused logits of shape (batch_size, num_concepts).
        """
        if not self.use_dynamic_gate:
            return 0.5 * local_logits + 0.5 * global_logits

        weight_local = torch.sigmoid(self.gate_logits)
        return weight_local * local_logits + (1.0 - weight_local) * global_logits
