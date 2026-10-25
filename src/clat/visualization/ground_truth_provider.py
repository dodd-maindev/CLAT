"""Provides clinical ground-truth disease labels, lesion tags, and segmentation masks."""

import glob
import os
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from clat.olap.mask_provider import TrainMaskProvider


class GroundTruthProvider:
    """Manages clinical ground-truth metadata and doctor segmentation masks."""

    LESION_CODES = ["EX", "HE", "MA", "SE"]

    def __init__(self, data_root: str = "data") -> None:
        """Initializes disease and lesion annotations with mask provider."""
        self.data_root = data_root
        self.disease_map: Dict[str, str] = {}
        self.mask_provider = TrainMaskProvider(target_size=384)
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parses CSV annotation files for disease and lesion ground truth."""
        cols = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        for fname in ["annotation_FGADR_disease.csv", "annotation_DDR_disease.csv"]:
            p = os.path.join(self.data_root, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                for _, r in df.iterrows():
                    self.disease_map[str(r["ID"])] = cols[int(np.argmax(r[cols].values))]

    def get_disease(self, img_id: str) -> str:
        """Returns clinical ground-truth disease grade."""
        return self.disease_map.get(img_id, "Unknown")

    def get_lesion_masks(self, img_id: str, h: int = 384, w: int = 384) -> Dict[str, Optional[np.ndarray]]:
        """Returns individual 2D masks for each lesion code EX, HE, MA, SE."""
        masks: Dict[str, Optional[np.ndarray]] = {code: None for code in self.LESION_CODES}
        if self.mask_provider.has_mask(img_id):
            raw = self.mask_provider.load_raw_mask(img_id, h, w)
            for idx, code in enumerate(self.LESION_CODES):
                masks[code] = raw[..., idx]
            return masks

        for code in self.LESION_CODES:
            for ext in [".tif", ".tiff", ".png", ".jpg"]:
                matches = glob.glob(f"data/**/{code}/{img_id}{ext}", recursive=True)
                if matches:
                    loaded = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
                    if loaded is not None:
                        masks[code] = cv2.resize(loaded, (w, h), interpolation=cv2.INTER_NEAREST)
                        break
        return masks
