"""Pathological knowledge graph module modeling clinical lesion correlations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathologicalKnowledgeGraph(nn.Module):
    """Refines concept embedding space via medical co-occurrence graph reasoning."""

    def __init__(self, embed_dim: int = 512, num_lesions: int = 4) -> None:
        """Initializes clinical co-occurrence prior and zero-initialized residual gate."""
        super().__init__()
        # DR progression prior:
        # 0: MA (Microaneurysms), 1: HE (Hemorrhages), 2: EX (Hard Exudates), 3: SE (Cotton Wool Spots)
        prior_adj = torch.tensor(
            [
                [1.0, 0.8, 0.7, 0.3],  # MA
                [0.8, 1.0, 0.6, 0.8],  # HE
                [0.7, 0.6, 1.0, 0.5],  # EX
                [0.3, 0.8, 0.5, 1.0],  # SE
            ],
            dtype=torch.float32,
        )
        self.register_buffer("prior_adj", prior_adj)
        self.learnable_adj = nn.Parameter(torch.zeros(num_lesions, num_lesions))
        self.proj = nn.Linear(embed_dim, embed_dim)
        # Zero-gate initialization guarantees strict baseline equivalence at step 0
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, concept_features: torch.Tensor) -> torch.Tensor:
        """Propagates contextual messages across lesion concepts."""
        adj = self.prior_adj + torch.tanh(self.learnable_adj)
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(dim=-1).clamp(min=1e-6)))
        norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
        transformed = self.proj(concept_features)
        if transformed.dim() == 2:
            graph_out = norm_adj @ transformed
        else:
            graph_out = torch.einsum("ij,bjk->bik", norm_adj, transformed)
        refined = concept_features + self.gate * F.gelu(graph_out)
        return self.norm(refined)
