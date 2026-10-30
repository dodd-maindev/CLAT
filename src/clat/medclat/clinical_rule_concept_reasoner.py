"""Clinical rule-guided concept reasoning module mapping lesions to stages."""

import torch
import torch.nn as nn
from .clinical_icdr_matrix import ClinicalIcdrPriorMatrix


class ClinicalRuleConceptReasoner(nn.Module):
    """Reasons from lesion concepts to diagnostic stages using ICDR clinical priors."""

    def __init__(
        self,
        number_of_concepts: int = 4,
        number_of_classes: int = 5,
        hidden_dimension: int = 32,
    ) -> None:
        """Initializes learned neural pathway, clinical prior matrix, and adaptive gate."""
        super().__init__()
        self.prior_provider = ClinicalIcdrPriorMatrix(
            number_of_classes=number_of_classes,
            number_of_concepts=number_of_concepts,
        )
        self.learned_pathway = nn.Sequential(
            nn.Linear(number_of_concepts, hidden_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, number_of_classes),
        )
        self.clinical_prior_scale = nn.Parameter(torch.tensor(0.5))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, lesion_probabilities: torch.Tensor) -> torch.Tensor:
        """Fuses data-driven concept mapping with ICDR clinical rule matrix.

        Args:
            lesion_probabilities: Sigmoidal concept activations of shape (batch_size, 4).

        Returns:
            Reasoned disease stage logits of shape (batch_size, 5).
        """
        learned_logits = self.learned_pathway(lesion_probabilities)

        prior_matrix = self.prior_provider.get_matrix()  # (5, 4)
        clinical_score = torch.matmul(lesion_probabilities, prior_matrix.t())  # (B, 5)

        calibrated_prior = clinical_score / torch.clamp(self.temperature, min=0.1)
        return learned_logits + self.clinical_prior_scale * calibrated_prior

