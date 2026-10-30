"""Multi-scale micro-lesion refiner injecting fine spatial details into patch tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLesionRefiner(nn.Module):
    """Enriches coarse transformer patch tokens with high-resolution micro-lesion cues."""

    def __init__(self, embed_dim: int = 288, in_channels: int = 3) -> None:
        """Initializes multi-scale convolutional branches and feature injection gate."""
        super().__init__()
        # Fine-scale branch capturing tiny microaneurysms (3px - 8px)
        self.fine_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Dilated contextual branch capturing hemorrhages and exudate clusters
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
        )
        # Learnable gating scalar initialized close to zero for stable training
        self.refinement_gate = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, patch_tokens: torch.Tensor, raw_images: torch.Tensor
    ) -> torch.Tensor:
        """Fuses multi-scale micro-lesion features into spatial patch representation.

        Args:
            patch_tokens: Transformer patch tokens of shape (B, embed_dim, H_p, W_p).
            raw_images: Original high-resolution fundus images of shape (B, 3, H, W).

        Returns:
            Detail-enhanced patch tokens of identical shape (B, embed_dim, H_p, W_p).
        """
        fine_features = self.fine_branch(raw_images)
        lesion_cues = self.dilated_branch(fine_features)

        # Resample fine-grained cues to align exactly with patch token spatial grid
        target_height, target_width = patch_tokens.shape[2], patch_tokens.shape[3]
        aligned_cues = F.adaptive_avg_pool2d(
            lesion_cues, (target_height, target_width)
        )

        fused = patch_tokens + self.refinement_gate * aligned_cues
        # Apply layer normalization across channel dimension
        permuted = fused.permute(0, 2, 3, 1)
        normalized = self.norm(permuted)
        return normalized.permute(0, 3, 1, 2).contiguous()

