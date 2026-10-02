"""Direct Attention Guidance module aligning Transformer attention with doctor masks."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAttentionGuidanceLoss(nn.Module):
    """Aligns multi-head attention maps directly with doctor segmentation masks."""

    def __init__(self, loss_weight: float = 0.02, pos_weight: float = 10.0) -> None:
        """Initializes the direct attention guidance module.

        Args:
            loss_weight: Scaling coefficient for the attention guidance penalty.
            pos_weight: Importance weighting for sparse positive lesion patches.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight

    def _normalize_map(self, attention_slice: torch.Tensor) -> torch.Tensor:
        """Normalizes an individual 2D attention map into [0, 1] range."""
        min_val = attention_slice.min()
        max_val = attention_slice.max()
        return (attention_slice - min_val) / (max_val - min_val + 1e-8)

    def forward(
        self,
        multi_head_attention: torch.Tensor,
        doctor_masks: Optional[torch.Tensor],
        lesion_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes weighted BCE alignment loss between attention maps and masks.

        Args:
            multi_head_attention: Tensor of shape (B, num_concepts, H, W).
            doctor_masks: Ground truth masks of shape (B, num_concepts, H_orig, W_orig).
            lesion_labels: Binary concept presence of shape (B, num_concepts).

        Returns:
            Scalar guidance loss tensor.
        """
        if not self.training or doctor_masks is None or self.loss_weight <= 0.0:
            return torch.tensor(0.0, device=multi_head_attention.device)

        batch_size, num_concepts, grid_h, grid_w = multi_head_attention.shape
        downsampled_masks = F.adaptive_max_pool2d(doctor_masks.to(multi_head_attention.device), (grid_h, grid_w))

        # Identify training samples that have non-empty doctor mask annotations
        has_mask_per_sample = downsampled_masks.sum(dim=(1, 2, 3)) > 0
        if not has_mask_per_sample.any():
            return torch.tensor(0.0, device=multi_head_attention.device)

        total_loss = torch.tensor(0.0, device=multi_head_attention.device)
        valid_count = 0

        for b in range(batch_size):
            if not has_mask_per_sample[b]:
                continue
            for k in range(num_concepts):
                # Only supervise when the concept label is positive and mask contains lesions
                target_k = downsampled_masks[b, k]
                is_active = target_k.sum() > 0
                if lesion_labels is not None and lesion_labels[b, k] == 0:
                    continue

                if is_active:
                    norm_attn = self._normalize_map(multi_head_attention[b, k])
                    clamped_attn = torch.clamp(norm_attn, 1e-4, 1.0 - 1e-4)
                    weights = 1.0 + (self.pos_weight - 1.0) * target_k
                    loss_k = F.binary_cross_entropy(clamped_attn, target_k, weight=weights)
                    total_loss = total_loss + loss_k
                    valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=multi_head_attention.device)

        return self.loss_weight * (total_loss / valid_count)
