"""Anatomical landmark-aware disease classifier for retinal grading."""

from typing import Optional
import torch
import torch.nn as nn
from clat.olap.dual_path_disease_classifier import DualPathDiseaseClassifier
from .anatomical_estimator import AnatomicalLandmarkEstimator


class AnatomicalAwareDiseaseClassifier(nn.Module):
    """Integrates anatomical context tokens into dual-path disease diagnosis."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        num_concepts: int = 4,
        initial_global_weight: float = 0.1,
        initial_concept_weight: float = 0.1,
    ) -> None:
        """Initializes landmark estimator, base classifier, and landmark projection."""
        super().__init__()
        self.estimator = AnatomicalLandmarkEstimator(embed_dim)
        self.base_classifier = DualPathDiseaseClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_concepts=num_concepts,
            initial_global_weight=initial_global_weight,
            initial_concept_weight=initial_concept_weight,
        )
        self.landmark_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self.landmark_gate = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        disease_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        lesion_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuses base dual-path predictions with anatomical landmark features."""
        base_logits = self.base_classifier(disease_tokens, patch_tokens, lesion_logits)
        if patch_tokens.dim() == 4:
            _, landmark_tokens = self.estimator(patch_tokens)
            flat_landmarks = landmark_tokens.flatten(1)
            landmark_logits = self.landmark_head(flat_landmarks)
            return base_logits + self.landmark_gate * landmark_logits
        return base_logits
