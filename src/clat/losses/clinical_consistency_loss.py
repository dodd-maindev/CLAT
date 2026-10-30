"""Clinical consistency loss enforcing ICDR diagnostic guidelines on concept predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalConsistencyLoss(nn.Module):
    """Penalizes clinical rule contradictions between predicted stage and lesion concepts."""

    def __init__(
        self,
        microaneurysm_index: int = 2,
        hemorrhage_index: int = 1,
        hard_exudate_index: int = 0,
        soft_exudate_index: int = 3,
    ) -> None:
        """Initializes canonical indices for standard retinal lesion concepts."""
        super().__init__()
        self.ma_idx = microaneurysm_index
        self.he_idx = hemorrhage_index
        self.ex_idx = hard_exudate_index
        self.se_idx = soft_exudate_index

    def forward(
        self, disease_logits: torch.Tensor, lesion_logits: torch.Tensor
    ) -> torch.Tensor:
        """Computes penalty for clinically inconsistent stage-lesion co-occurrences.

        Args:
            disease_logits: Predicted disease stage logits of shape (batch_size, 5).
            lesion_logits: Predicted concept logits of shape (batch_size, 4).

        Returns:
            Scalar tensor representing total clinical inconsistency penalty.
        """
        disease_probs = F.softmax(disease_logits, dim=-1)
        lesion_probs = torch.sigmoid(lesion_logits)

        # Rule 1: Grade 0 (No DR) cannot exhibit prominent lesions
        prob_grade_0 = disease_probs[:, 0]
        total_lesion_burden = lesion_probs.mean(dim=-1)
        penalty_grade_0 = prob_grade_0 * total_lesion_burden

        # Rule 2: Grade 1 (Mild NPDR) consists of microaneurysms only (no EX, HE, SE)
        prob_grade_1 = disease_probs[:, 1]
        advanced_lesions = (
            lesion_probs[:, self.ex_idx]
            + lesion_probs[:, self.he_idx]
            + lesion_probs[:, self.se_idx]
        ) / 3.0
        penalty_grade_1 = prob_grade_1 * advanced_lesions

        # Rule 3: Grade 3 & 4 (Severe & PDR) require high lesion presence
        severe_disease_probs = disease_probs[:, 3] + disease_probs[:, 4]
        max_lesion_intensity = lesion_probs.max(dim=-1).values
        penalty_severe_without_lesions = severe_disease_probs * (1.0 - max_lesion_intensity)

        total_penalty = penalty_grade_0 + penalty_grade_1 + penalty_severe_without_lesions
        return total_penalty.mean()

