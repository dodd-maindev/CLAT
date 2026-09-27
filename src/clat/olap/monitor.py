"""Diagnostic and transparency monitoring callback for concept learning."""

from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer


class ConceptTransparencyMonitor(Callback):
    """Logs detailed mathematical indicators to inspect and debug concept learning."""

    def __init__(self, verbose: bool = True) -> None:
        """Initializes the transparency monitor.

        Args:
            verbose: Whether to print an ASCII dashboard at the end of each epoch.
        """
        super().__init__()
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Extracts and logs internal geometric states of the OLAP concept head."""
        model = getattr(pl_module, "model", None)
        olap_head = getattr(model, "olap_head", None)
        if olap_head is None:
            return

        lesion_names: List[str] = getattr(pl_module, "lesion_names", [])
        num_concepts = len(lesion_names) if lesion_names else 4

        # 1. GeM powers and convex gate weights
        gem_p = torch.clamp(olap_head.gem_pooling.power, min=1.0).squeeze().detach().cpu().numpy()
        local_w = olap_head.dynamic_gate.local_weights.squeeze().detach().cpu().numpy()

        for idx, name in enumerate(lesion_names if lesion_names else [f"C{i}" for i in range(num_concepts)]):
            pl_module.log(f"olap_p/{name}", float(gem_p[idx]), prog_bar=False)
            pl_module.log(f"olap_local_weight/{name}", float(local_w[idx]), prog_bar=False)

        # 2. Pairwise cosine similarity among concept tokens
        tokens = getattr(model, "lesion_tokens", None)
        if tokens is not None:
            normed = F.normalize(tokens.squeeze(0), p=2, dim=-1)
            cosine_matrix = torch.mm(normed, normed.t()).detach().cpu().numpy()
            np.fill_diagonal(cosine_matrix, 0.0)
            max_sim = float(np.max(np.abs(cosine_matrix)))
            mean_sim = float(np.mean(np.abs(cosine_matrix)))
            pl_module.log("olap/max_concept_cosine", max_sim, prog_bar=False)
            pl_module.log("olap/mean_concept_cosine", mean_sim, prog_bar=False)
        else:
            max_sim, mean_sim = 0.0, 0.0

        # 3. Print transparent diagnostic report
        if self.verbose and trainer.is_global_zero:
            self._print_epoch_summary(trainer.current_epoch, lesion_names, gem_p, local_w, max_sim)

    def _print_epoch_summary(
        self, epoch: int, names: List[str], powers: np.ndarray, weights: np.ndarray, max_sim: float
    ) -> None:
        """Prints a human-readable ASCII status report of concept dynamics."""
        print(f"\n{'='*25} [OLAP DIAGNOSTICS - Epoch {epoch:02d}] {'='*25}")
        print(f"{'Concept':<20} | {'GeM Power (p)':<14} | {'Local %':<10} | {'Global %':<10}")
        print("-" * 65)
        for idx, name in enumerate(names if names else [f"Concept {i}" for i in range(len(powers))]):
            print(f"{name:<20} | {powers[idx]:<14.3f} | {weights[idx]*100:<9.1f}% | {(1.0-weights[idx])*100:<9.1f}%")
        print("-" * 65)
        print(f"Max Off-Diagonal Cosine: {max_sim:.4f} (Disentanglement Metric)")
        print(f"{'='*74}\n")
