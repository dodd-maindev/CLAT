"""Loss functions package for clinical and concept-aware retinal grading."""

from .clinical_consistency_loss import ClinicalConsistencyLoss
from .composite_disease_loss import CompositeDiseaseLoss
from .ordinal_focal_loss import OrdinalFocalLoss
from .quadratic_weighted_kappa_loss import QuadraticWeightedKappaLoss

__all__ = [
    "ClinicalConsistencyLoss",
    "CompositeDiseaseLoss",
    "OrdinalFocalLoss",
    "QuadraticWeightedKappaLoss",
]

