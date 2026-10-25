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
    SEARCH_ROOTS = ["data", "/content/CLAT/data", "/content/drive/MyDrive"]

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

    def find_cases_with_active_masks(self) -> List[str]:
        """Scans for image IDs that actually have doctor segmentation mask files on disk."""
        active_ids = set(self.mask_provider._index.keys())
        for root in self.SEARCH_ROOTS:
            if not os.path.exists(root):
                continue
            for p in glob.glob(f"{root}/**/*.tif", recursive=True):
                active_ids.add(os.path.splitext(os.path.basename(p))[0])
        return sorted(list(active_ids))

    def get_lesion_masks(self, img_id: str, h: int = 384, w: int = 384) -> Dict[str, Optional[np.ndarray]]:
        """Returns individual 2D masks for each lesion code EX, HE, MA, SE."""
        masks: Dict[str, Optional[np.ndarray]] = {code: None for code in self.LESION_CODES}
        if self.mask_provider.has_mask(img_id):
            raw = self.mask_provider.load_raw_mask(img_id, h, w)
            for idx, code in enumerate(self.LESION_CODES):
                masks[code] = raw[..., idx]
            return masks

        for code in self.LESION_CODES:
            for root in self.SEARCH_ROOTS:
                if not os.path.exists(root):
                    continue
                matches = glob.glob(f"{root}/**/{code}/{img_id}.*", recursive=True) + glob.glob(
                    f"{root}/**/label/{code}/{img_id}.*", recursive=True
                )
                for m in matches:
                    if m.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                        loaded = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                        if loaded is not None and loaded.max() > 0:
                            masks[code] = cv2.resize(loaded, (w, h), interpolation=cv2.INTER_NEAREST)
                            break
                if masks[code] is not None:
                    break
        return masks
