"""Dual-path classifier combining lesion cross-attention, direct concepts, and global context."""

from typing import Optional
import torch
import torch.nn as nn
from clat.medclat.clinical_rule_concept_reasoner import ClinicalRuleConceptReasoner


class DualPathDiseaseClassifier(nn.Module):
    """Fuses concept-level attention tokens, ICDR-reasoned concepts, and global context."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        num_concepts: int = 4,
        initial_global_weight: float = 0.1,
        initial_concept_weight: float = 0.1,
    ) -> None:
        """Initializes projection heads, ICDR concept reasoner, and residual fusion gates."""
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
        self.direct_concept_head = ClinicalRuleConceptReasoner(
            number_of_concepts=num_concepts,
            number_of_classes=num_classes,
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
        concept_logits = disease_tokens.mean(dim=-1)
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
