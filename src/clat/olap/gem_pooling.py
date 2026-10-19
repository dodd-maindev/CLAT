"""Generalized Mean Pooling layer for spatial concept extraction."""

import torch
import torch.nn as nn


class GeneralizedMeanPooling2d(nn.Module):
    """Smooth Generalized Pooling across spatial dimensions with learnable temperature."""

    def __init__(self, num_channels: int, initial_power: float = 3.0, eps: float = 1e-6) -> None:
        """Initializes the smooth generalized pooling layer.

        Args:
            num_channels: Number of input channels (one parameter per channel).
            initial_power: Initial scaling factor for smooth maximum.
            eps: Epsilon constant for stability.
        """
        super().__init__()
        self.eps = eps
        self.power = nn.Parameter(torch.full((1, num_channels, 1, 1), float(initial_power)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies smooth generalized pooling over spatial dimensions (H, W).

        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Pooled tensor of shape (batch_size, num_channels).
        """
        tau = torch.clamp(self.power, min=0.1, max=10.0)
        x_scaled = x * tau
        lse = torch.logsumexp(x_scaled, dim=(-2, -1), keepdim=True)
        pooled = lse / tau
        return torch.flatten(pooled, start_dim=1)

