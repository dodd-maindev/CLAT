"""Generalized Mean Pooling (GeM) layer for spatial concept extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedMeanPooling2d(nn.Module):
    """Generalized Mean Pooling across spatial dimensions with learnable p-norm."""

    def __init__(self, num_channels: int, initial_power: float = 3.0, eps: float = 1e-6) -> None:
        """Initializes the GeM pooling layer.

        Args:
            num_channels: Number of input channels (one power parameter per channel).
            initial_power: Initial p-norm exponent value (default 3.0).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.eps = eps
        # Learnable power parameter p per channel
        self.power = nn.Parameter(torch.full((1, num_channels, 1, 1), float(initial_power)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies GeM pooling over spatial dimensions (H, W).

        Args:
            x: Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Pooled tensor of shape (batch_size, num_channels).
        """
        # Clamp power parameter to be strictly >= 1.0 for valid norms
        clamped_power = torch.clamp(self.power, min=1.0)
        clamped_x = torch.clamp(x, min=self.eps)
        powered_x = clamped_x.pow(clamped_power)
        mean_powered_x = F.adaptive_avg_pool2d(powered_x, (1, 1))
        pooled = mean_powered_x.pow(1.0 / clamped_power)
        return torch.flatten(pooled, start_dim=1)
