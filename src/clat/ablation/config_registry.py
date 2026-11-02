"""Registry of ablation study variants and configuration paths."""

from typing import Dict, List

TABLE_6_CONFIGS: Dict[str, str] = {
    "wo_detail_injector": "configs/ablation/table6_wo_detail_injector.yaml",
    "wo_olap_head": "configs/ablation/table6_wo_olap_head.yaml",
    "wo_pkg": "configs/ablation/table6_wo_pkg.yaml",
    "wo_landmark_classifier": "configs/ablation/table6_wo_landmark_classifier.yaml",
}

TABLE_7_CONFIGS: Dict[str, str] = {
    "wo_lse_pooling": "configs/ablation/table7_wo_lse_pooling.yaml",
    "wo_layernorm": "configs/ablation/table7_wo_layernorm.yaml",
    "wo_dynamic_gate": "configs/ablation/table7_wo_dynamic_gate.yaml",
    "wo_ortho_loss": "configs/ablation/table7_wo_ortho_loss.yaml",
}

TABLE_6_METRICS: List[str] = [
    "auc",
    "kappa",
    "macro_f1",
    "sensitivity",
    "specificity",
]

TABLE_7_METRICS: List[str] = [
    "auc",
    "kappa",
    "cpt_acc",
    "cpt_f1",
    "mean_concept_cosine",
]


def get_ablation_config(table_num: int, variant_name: str) -> str:
    """Retrieves config path for a specified table and variant."""
    registry = TABLE_6_CONFIGS if table_num == 6 else TABLE_7_CONFIGS
    if variant_name not in registry:
        raise KeyError(f"Variant '{variant_name}' not found in Table {table_num}.")
    return registry[variant_name]
