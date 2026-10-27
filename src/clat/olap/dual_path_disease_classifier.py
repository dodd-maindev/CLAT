"""Dual-path classifier combining lesion cross-attention, direct concepts, and global context."""

from typing import Optional
import torch
import torch.nn as nn


class DualPathDiseaseClassifier(nn.Module):
    """Fuses concept-level attention tokens, direct concept probabilities, and global retinal patch context."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        num_concepts: int = 4,
        initial_global_weight: float = 0.1,
        initial_concept_weight: float = 0.1,
    ) -> None:
        """Initializes projection heads and residual fusion parameters."""
        super().__init__()
        self.concept_projector = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.concept_projector.weight)
        nn.init.zeros_(self.concept_projector.bias)
        self.global_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.direct_concept_head = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes),
        )
        self.global_gate = nn.Parameter(torch.tensor(initial_global_weight))
        self.concept_gate = nn.Parameter(torch.tensor(initial_concept_weight))

    def forward(
        self,
        disease_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        lesion_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes fused disease logits from concept attention, global patches, and direct concepts."""
        base_logits = disease_tokens.mean(dim=-1)
        learned_logits = self.concept_projector(disease_tokens).squeeze(-1)
        concept_logits = base_logits + learned_logits
        if patch_tokens.dim() == 4:
            global_context = patch_tokens.mean(dim=(-2, -1))
        elif patch_tokens.dim() == 3:
            global_context = patch_tokens.mean(dim=1)
        else:
            global_context = patch_tokens
        global_logits = self.global_projector(global_context)
        out = concept_logits + self.global_gate * global_logits
        if lesion_logits is not None:
            concept_probs = torch.sigmoid(lesion_logits)
            concept_disease_logits = self.direct_concept_head(concept_probs)
            out = out + self.concept_gate * concept_disease_logits
        return out
