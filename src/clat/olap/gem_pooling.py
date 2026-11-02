"""Generalized Mean Pooling layer for spatial concept extraction."""

import torch
import torch.nn as nn


class GeneralizedMeanPooling2d(nn.Module):
    """Smooth Generalized Pooling across spatial dimensions with learnable temperature or GAP."""

    def __init__(
        self,
        num_channels: int,
        initial_power: float = 3.0,
        eps: float = 1e-6,
        pooling_type: str = "lse",
    ) -> None:
        """Initializes the spatial pooling layer.

        Args:
            num_channels: Number of input channels (one parameter per channel).
            initial_power: Initial scaling factor for smooth maximum.
            eps: Epsilon constant for stability.
            pooling_type: Type of pooling ("lse" for LogSumExp, "gap" for average).
        """
        super().__init__()
        self.eps = eps
        self.pooling_type = pooling_type.lower()
        self.power = nn.Parameter(torch.full((1, num_channels, 1, 1), float(initial_power)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies spatial pooling over spatial dimensions (H, W).

        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Pooled tensor of shape (batch_size, num_channels).
        """
        if self.pooling_type == "gap":
            return torch.flatten(x.mean(dim=(-2, -1)), start_dim=1)

        tau = torch.clamp(self.power, min=0.1, max=10.0)
        x_scaled = x * tau
        lse = torch.logsumexp(x_scaled, dim=(-2, -1), keepdim=True)
        pooled = lse / tau
        return torch.flatten(pooled, start_dim=1)
