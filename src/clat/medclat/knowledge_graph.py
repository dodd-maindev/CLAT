"""Knowledge graph module modeling co-occurrence relationships among retinal lesions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalLesionKnowledgeGraph(nn.Module):
    """Refines lesion concept representations via pathological graph reasoning."""

    def __init__(self, embed_dim: int = 512, num_lesions: int = 4) -> None:
        """Initializes clinical co-occurrence adjacency and message passing layers."""
        super().__init__()
        # DR pathology progression prior: MA -> HE, MA -> EX, HE -> SE
        # Indices: 0: MA, 1: HE, 2: EX, 3: SE
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
        self.weight = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, concept_embeds: torch.Tensor) -> torch.Tensor:
        """Propagates contextual lesion messages across the medical graph."""
        adj = self.prior_adj + torch.tanh(self.learnable_adj)
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(adj.sum(dim=-1).clamp(min=1e-6)))
        norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
        transformed = self.weight(concept_embeds)
        if transformed.dim() == 2:
            graph_out = norm_adj @ transformed
        else:
            graph_out = torch.einsum("ij,bjk->bik", norm_adj, transformed)
        refined = concept_embeds + self.graph_gate * F.gelu(graph_out)
        return self.norm(refined)
