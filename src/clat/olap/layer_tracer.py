"""Layer dataflow tracer callback to log layer-by-layer activations and representations."""

import json
import os
from typing import Any, Dict
from lightning import Callback, LightningModule, Trainer


class LayerDataflowTracer(Callback):
    """Traces and logs exact tensor transformations through model layers on batch 0."""

    def __init__(self, trace_every_n_epochs: int = 1) -> None:
        super().__init__()
        self.trace_every_n_epochs = trace_every_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Enables layer tracing for the first batch of the epoch."""
        if trainer.current_epoch % self.trace_every_n_epochs == 0:
            if hasattr(pl_module, "model"):
                setattr(pl_module.model, "capture_trace", True)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Extracts trace from batch 0, prints summary table, and saves JSON record."""
        if batch_idx != 0 or not hasattr(pl_module, "model"):
            return

        model = pl_module.model
        setattr(model, "capture_trace", False)
        trace_data: Dict[str, Any] = getattr(model, "last_forward_trace", {})
        if not trace_data:
            return

        # 1. Print layer-by-layer dataflow report
        if trainer.is_global_zero:
            self._print_trace(trainer.current_epoch, trace_data, pl_module)

        # 2. Save JSON trace for post-hoc debugging
        if trainer.log_dir is not None:
            trace_dir = os.path.join(trainer.log_dir, "layer_traces")
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"epoch_{trainer.current_epoch:02d}_batch0_trace.json")
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)

    def _print_trace(self, epoch: int, trace: Dict[str, Any], pl_module: LightningModule) -> None:
        """Prints a human-readable dataflow summary."""
        print(f"\n{'-'*28} [LAYER DATAFLOW TRACE - Epoch {epoch:02d}] {'-'*28}")
        for layer_name, info in trace.items():
            if isinstance(info, dict):
                shape_str = str(info.get("shape", ""))
                norm_val = info.get("norm", info.get("mean", ""))
                norm_str = f"norm/mean={norm_val:.4f}" if isinstance(norm_val, (int, float)) else ""
                print(f"  -> {layer_name:<28} | Shape: {shape_str:<18} | {norm_str}")
            elif isinstance(info, list):
                vals = [f"{v:.3f}" for v in info[:5]]
                print(f"  -> {layer_name:<28} | Sample Values: {vals}")
        print(f"{'-'*76}\n")
