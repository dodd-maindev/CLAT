"""Loss modules for medical ordinal classification and concept awareness."""

from clat.losses.ordinal_disease_loss import OrdinalDiseaseLoss
from clat.losses.soft_qwk_loss import SoftQuadraticWeightedKappaLoss

__all__ = ["SoftQuadraticWeightedKappaLoss", "OrdinalDiseaseLoss"]
