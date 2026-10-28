"""Injects high-frequency spatial details into lesion tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionDetailInjector(nn.Module):
    """Adapts high-frequency spatial features and injects them into lesion tokens."""

    def __init__(self, embed_dim: int = 384, num_lesions: int = 4) -> None:
        """Initializes cross-attention pooling for lesion-specific details."""
        super().__init__()
        self.num_lesions = num_lesions
        self.proj_key = nn.Linear(embed_dim, embed_dim)
        self.proj_val = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.zeros(1, num_lesions, 1))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, lesion_tokens: torch.Tensor, high_freq_map: torch.Tensor
    ) -> torch.Tensor:
        """Fuses high-frequency details into lesion query tokens."""
        batch_size = lesion_tokens.shape[0]
        flattened = high_freq_map.flatten(2).permute(0, 2, 1)
        keys = self.proj_key(flattened)
        values = self.proj_val(flattened)
        scores = torch.bmm(lesion_tokens, keys.transpose(1, 2))
        scale = 1.0 / (lesion_tokens.shape[-1] ** 0.5)
        attn = F.softmax(scores * scale, dim=-1)
        detail_ctx = torch.bmm(attn, values)
        enhanced = lesion_tokens + torch.tanh(self.gate) * detail_ctx
        return self.norm(enhanced)
