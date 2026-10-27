"""Estimator for retinal anatomical landmarks: Optic Disc and Fovea."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalLandmarkEstimator(nn.Module):
    """Predicts coarse center locations and regional embeddings for optic disc and fovea."""

    def __init__(self, embed_dim: int = 384) -> None:
        """Initializes spatial landmark projection heads."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.landmark_embed_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns landmark heatmaps [B, 2, H, W] and spatial context tokens [B, 2, D]."""
        heatmaps = self.conv(patch_tokens)
        b, _, h, w = heatmaps.shape
        probs = F.softmax(heatmaps.flatten(2), dim=-1).view(b, 2, h, w)
        pooled = torch.einsum("bchw,bdhw->bcd", probs, patch_tokens)
        landmark_tokens = self.landmark_embed_proj(pooled)
        return heatmaps, landmark_tokens
