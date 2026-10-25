"""Test-Time Augmentation (TTA) evaluator for rotational-invariant retinal image inference."""

from typing import Optional
import torch
import torch.nn as nn
from clat.utils import CLATOutput


class TestTimeAugmentationEvaluator:
    """Evaluates multi-view augmented retinal images under full D4 dihedral group transformations."""

    def __init__(self, use_full_dihedral: bool = True) -> None:
        """Initializes multi-view settings with full 8-fold rotational/reflection symmetry."""
        self.use_full_dihedral = use_full_dihedral

    def generate_views(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Generates 8 invariant views: 4 orthogonal rotations combined with horizontal flip."""
        views: list[torch.Tensor] = []
        for k in (0, 1, 2, 3):
            rotated = torch.rot90(images, k=k, dims=[-2, -1])
            views.append(rotated)
            if self.use_full_dihedral:
                views.append(torch.flip(rotated, dims=[-1]))
        return views

    def evaluate(
        self,
        model_forward_fn,
        images: torch.Tensor,
        lesion_labels: Optional[torch.Tensor] = None,
    ) -> CLATOutput:
        """Averages softmax probabilities and lesion activations across all dihedral views."""
        views = self.generate_views(images)

        out_first = model_forward_fn(views[0], return_attn=False, lesion_lbls=lesion_labels)
        sum_disease_prob = out_first.disease_logits.softmax(dim=-1)
        sum_lesion_prob = out_first.lesion_logits.sigmoid()

        for view in views[1:]:
            out_view = model_forward_fn(view, return_attn=False, lesion_lbls=lesion_labels)
            sum_disease_prob = sum_disease_prob + out_view.disease_logits.softmax(dim=-1)
            sum_lesion_prob = sum_lesion_prob + out_view.lesion_logits.sigmoid()

        num_views = float(len(views))
        avg_disease_prob = sum_disease_prob / num_views
        avg_lesion_prob = sum_lesion_prob / num_views

        # Safe logit reconstruction from calibrated ensemble probabilities
        safe_disease_logits = torch.log(avg_disease_prob.clamp(min=1e-7))
        safe_lesion_logits = torch.logit(avg_lesion_prob.clamp(min=1e-7, max=1.0 - 1e-7))

        return CLATOutput(
            disease_logits=safe_disease_logits,
            lesion_logits=safe_lesion_logits,
            lesion_tokens=out_first.lesion_tokens,
            concept_masks=None,
            attn_weights=None,
            cross_attn_maps=None,
        )
