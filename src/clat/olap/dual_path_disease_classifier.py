"""Dual-path classifier combining lesion cross-attention and global context."""

import torch
import torch.nn as nn


class DualPathDiseaseClassifier(nn.Module):
    """Fuses concept-level attention tokens with global retinal patch context."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        initial_global_weight: float = 0.1,
    ) -> None:
        """Initializes projection heads and residual fusion parameter."""
        super().__init__()
        self.concept_projector = nn.Linear(embed_dim, 1)
        self.global_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.global_gate = nn.Parameter(torch.tensor(initial_global_weight))

    def forward(
        self,
        disease_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Computes fused disease logits from concept attention and global patches."""
        concept_logits = self.concept_projector(disease_tokens).squeeze(-1)
        if patch_tokens.dim() == 4:
            global_context = patch_tokens.mean(dim=(-2, -1))
        elif patch_tokens.dim() == 3:
            global_context = patch_tokens.mean(dim=1)
        else:
            global_context = patch_tokens
        global_logits = self.global_projector(global_context)
        return concept_logits + self.global_gate * global_logits
