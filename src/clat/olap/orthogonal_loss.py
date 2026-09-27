"""Orthogonal subspace regularization loss for concept token disentanglement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalSubspaceLoss(nn.Module):
    """Enforces mathematical orthogonality between distinct concept token representations."""

    def __init__(self, loss_weight: float = 0.1, eps: float = 1e-6) -> None:
        """Initializes the orthogonal subspace loss.

        Args:
            loss_weight: Multiplier weight for the orthogonality penalty.
            eps: Epsilon constant to prevent division by zero during normalization.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, concept_tokens: torch.Tensor) -> torch.Tensor:
        """Computes Frobenius norm discrepancy between token Gram matrix and identity.

        Args:
            concept_tokens: Tensor of shape (batch_size, num_concepts, embed_dim).

        Returns:
            Scalar penalty encouraging mutual orthogonality among concepts.
        """
        batch_size, num_concepts, _ = concept_tokens.shape
        if num_concepts <= 1:
            return torch.tensor(0.0, device=concept_tokens.device)

        # L2-normalize tokens across the embedding dimension: (B, C, D)
        normalized_tokens = F.normalize(concept_tokens, p=2, dim=-1, eps=self.eps)

        # Gram matrix of cosine similarities: (B, C, C)
        gram_matrix = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))

        # Identity matrix: (C, C)
        identity = torch.eye(num_concepts, device=concept_tokens.device).unsqueeze(0)

        # Off-diagonal penalty (squared Frobenius norm of off-diagonal elements)
        difference = gram_matrix - identity
        off_diagonal_penalty = difference.pow(2).sum(dim=(-2, -1)) / (num_concepts * (num_concepts - 1))

        return self.loss_weight * off_diagonal_penalty.mean()
