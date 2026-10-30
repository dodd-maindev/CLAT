"""Clinical prior specification defining ICDR stage-to-lesion correlation matrix."""

import torch
import torch.nn as nn


class ClinicalIcdrPriorMatrix(nn.Module):
    """Encapsulates the International Clinical Diabetic Retinopathy guideline prior matrix."""

    def __init__(self, number_of_classes: int = 5, number_of_concepts: int = 4) -> None:
        """Initializes normalized prior matrix according to ICDR clinical definitions.

        Columns correspond to standard lesions: [EX (0), HE (1), MA (2), SE (3)].
        Rows correspond to stages: [0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: PDR].
        """
        super().__init__()
        # Rows: Stages 0..4, Columns: EX, HE, MA, SE
        icdr_matrix = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0],  # Grade 0: Absence of all lesion types
                [-0.6, -0.6,  1.0, -0.8],  # Grade 1: Microaneurysms only
                [ 0.6,  0.6,  0.8,  0.3],  # Grade 2: Mild-moderate hemorrhages / exudates
                [ 0.8,  1.0,  0.9,  0.8],  # Grade 3: Severe 4-2-1 intraretinal hemorrhages
                [ 0.9,  1.0,  1.0,  0.9],  # Grade 4: High proliferative lesion saturation
            ],
            dtype=torch.float32,
        )
        self.register_buffer("prior_matrix", icdr_matrix)

    def get_matrix(self) -> torch.Tensor:
        """Returns the registered ICDR prior matrix tensor."""
        return self.prior_matrix

