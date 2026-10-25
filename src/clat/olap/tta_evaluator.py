"""Test-Time Augmentation (TTA) evaluator for rotational-invariant retinal image inference."""

from typing import Optional
import torch
import torch.nn as nn
from clat.utils import CLATOutput


class TestTimeAugmentationEvaluator:
    """Evaluates multi-view augmented retinal images to eliminate borderline classification noise."""

    def __init__(self, include_vertical_flip: bool = True) -> None:
        """Initializes multi-view flip settings."""
        self.include_vertical_flip = include_vertical_flip

    def evaluate(
        self,
        model_forward_fn,
        images: torch.Tensor,
        lesion_labels: Optional[torch.Tensor] = None,
    ) -> CLATOutput:
        """Averages predictions across original and flipped views."""
        views = [images, torch.flip(images, dims=[-1])]
        if self.include_vertical_flip:
            views.append(torch.flip(images, dims=[-2]))

        out_first = model_forward_fn(views[0], return_attn=False, lesion_lbls=lesion_labels)
        sum_disease = out_first.disease_logits
        sum_lesion = out_first.lesion_logits

        for view in views[1:]:
            out_view = model_forward_fn(view, return_attn=False, lesion_lbls=lesion_labels)
            sum_disease = sum_disease + out_view.disease_logits
            sum_lesion = sum_lesion + out_view.lesion_logits

        num_views = float(len(views))
        return CLATOutput(
            disease_logits=sum_disease / num_views,
            lesion_logits=sum_lesion / num_views,
            lesion_tokens=out_first.lesion_tokens,
            concept_masks=None,
            attn_weights=None,
            cross_attn_maps=None,
        )
