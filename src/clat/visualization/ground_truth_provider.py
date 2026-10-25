"""Provides clinical ground-truth disease labels, lesion tags, and segmentation masks."""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from clat.olap.mask_provider import TrainMaskProvider


class GroundTruthProvider:
    """Manages clinical ground-truth metadata and doctor segmentation masks."""

    def __init__(self, data_root: str = "data") -> None:
        """Initializes disease and lesion annotations with mask provider."""
        self.data_root = data_root
        self.disease_map: Dict[str, str] = {}
        self.lesion_map: Dict[str, List[str]] = {}
        self.mask_provider = TrainMaskProvider(target_size=384)
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parses CSV annotation files for disease and lesion ground truth."""
        d_csv = os.path.join(self.data_root, "annotation_FGADR_disease.csv")
        if os.path.exists(d_csv):
            df_d = pd.read_csv(d_csv)
            cols = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
            for _, r in df_d.iterrows():
                self.disease_map[str(r["ID"])] = cols[int(np.argmax(r[cols].values))]

        l_csv = os.path.join(self.data_root, "annotation_FGADR_lesion.csv")
        if os.path.exists(l_csv):
            df_l = pd.read_csv(l_csv)
            for _, r in df_l.iterrows():
                active = [les for les in ["EX", "HE", "MA", "SE"] if r.get(les, 0) == 1]
                self.lesion_map[str(r["ID"])] = active

    def get_disease(self, img_id: str) -> str:
        """Returns clinical ground-truth disease grade."""
        return self.disease_map.get(img_id, "Unknown")

    def get_lesions(self, img_id: str) -> List[str]:
        """Returns list of active ground-truth lesions."""
        return self.lesion_map.get(img_id, [])

    def get_raw_mask(self, img_id: str, h: int = 384, w: int = 384) -> Optional[np.ndarray]:
        """Returns 4-channel ground truth mask if available."""
        if self.mask_provider.has_mask(img_id):
            return self.mask_provider.load_raw_mask(img_id, h, w)
        return None
