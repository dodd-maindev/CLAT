"""Spatial consistency regularization with semi-supervised doctor mask guidance."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """Aligns OLAP spatial features with doctor masks or attention maps."""

    def __init__(
        self,
        loss_weight: float = 0.02,
        diversity_weight: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """Initializes the spatial consistency module."""
        super().__init__()
        self.loss_weight = loss_weight
        self.diversity_weight = diversity_weight
        self.eps = eps

    def _compute_alignment(
        self,
        spatial_norm: torch.Tensor,
        attention_norm: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes weakly-supervised attention alignment error."""
        cosine_sim = (spatial_norm * attention_norm).sum(dim=-1)
        error = 1.0 - cosine_sim
        if labels is None:
            return error.mean()
        valid = labels.float()
        total = valid.sum()
        if total > 0:
            return (error * valid).sum() / (total + self.eps)
        return torch.tensor(0.0, device=error.device)

    def forward(
        self,
        spatial_maps: torch.Tensor,
        attention_maps: torch.Tensor,
        lesion_labels: Optional[torch.Tensor] = None,
        doctor_masks: Optional[torch.Tensor] = None,
        has_mask_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes spatial consistency loss with semi-supervised mask guidance."""
        _, c = spatial_maps.shape[:2]
        spatial_act = F.relu(spatial_maps)
        s_norm = F.normalize(spatial_act.flatten(2), p=2, dim=-1, eps=self.eps)
        a_norm = F.normalize(attention_maps.flatten(2), p=2, dim=-1, eps=self.eps)

        # 1. Spatial supervision: Doctor ground-truth mask or Attention alignment
        if doctor_masks is not None and has_mask_flags is not None and has_mask_flags.any():
            sup_loss = F.binary_cross_entropy_with_logits(
                spatial_maps[has_mask_flags], doctor_masks[has_mask_flags]
            )
            unsup_flags = ~has_mask_flags
            if unsup_flags.any():
                u_lbls = lesion_labels[unsup_flags] if lesion_labels is not None else None
                weak_loss = self._compute_alignment(s_norm[unsup_flags], a_norm[unsup_flags], u_lbls)
                alignment_loss = 0.7 * sup_loss + 0.3 * weak_loss
            else:
                alignment_loss = sup_loss
        else:
            alignment_loss = self._compute_alignment(s_norm, a_norm, lesion_labels)

        # 2. Inter-concept spatial separation penalty
        if c > 1:
            gram = torch.bmm(s_norm, s_norm.transpose(1, 2))
            identity = torch.eye(c, device=spatial_maps.device).unsqueeze(0)
            off_diag = gram - identity
            diversity_loss = (off_diag**2).sum(dim=(-1, -2)).mean() / (c * (c - 1))
        else:
            diversity_loss = torch.tensor(0.0, device=spatial_maps.device)

        return self.loss_weight * alignment_loss + self.diversity_weight * diversity_loss
