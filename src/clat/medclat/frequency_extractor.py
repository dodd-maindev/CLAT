"""Frequency-guided detail enhancement module for retinal micro-lesions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFrequencyDetailExtractor(nn.Module):
    """Extracts high-frequency retinal lesion textures using Laplacian filtering."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 384) -> None:
        """Initializes fixed Laplacian kernel and trainable projection head."""
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        self.register_buffer("kernel", laplacian_kernel.repeat(in_channels, 1, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts high-frequency residual feature map from raw fundus images."""
        high_freq = F.conv2d(images, self.kernel, padding=1, groups=images.shape[1])
        return self.conv(high_freq)
