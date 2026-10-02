"""Semi-supervised training mask provider strictly indexing the training split."""

import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class TrainMaskProvider:
    """Manages ground-truth lesion segmentation masks for semi-supervised training."""

    LESIONS = ("EX", "HE", "MA", "SE")
    SEARCH_ROOTS = (
        "data/DDR-dataset/lesion_segmentation/train",
        "data/DDR/lesion_segmentation/train",
        "/content/CLAT/data/DDR-dataset/lesion_segmentation/train",
        "/content/CLAT/data/DDR/lesion_segmentation/train",
    )

    def __init__(self, target_size: int = 24) -> None:
        """Initializes mask provider with spatial target size."""
        self.target_size = target_size
        self._index: Dict[str, Dict[str, str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Indexes all masks found strictly in training directories."""
        for root in self.SEARCH_ROOTS:
            if not os.path.exists(root):
                continue
            for lesion in self.LESIONS:
                for p in [os.path.join(root, "label", lesion, "*.*"), os.path.join(root, lesion, "*.*")]:
                    for path in glob.glob(p):
                        if path.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                            img_id = os.path.splitext(os.path.basename(path))[0]
                            self._index.setdefault(img_id, {})[lesion] = path

    def has_mask(self, img_id: str) -> bool:
        """Checks if a training mask exists for the given image ID."""
        return img_id in self._index

    def load_raw_mask(self, img_id: str, height: int, width: int) -> np.ndarray:
        """Loads 4-channel ground truth mask of shape (H, W, 4) in {0.0, 1.0}."""
        mask = np.zeros((height, width, 4), dtype=np.float32)
        if not self.has_mask(img_id):
            return mask
        lesion_files = self._index[img_id]
        for i, lesion in enumerate(self.LESIONS):
            if lesion in lesion_files:
                pil_mask = Image.open(lesion_files[lesion]).convert("L").resize((width, height), Image.NEAREST)
                mask[:, :, i] = (np.array(pil_mask) > 0).astype(np.float32)
        return mask

    def load_single_mask(self, img_id: str) -> Optional[torch.Tensor]:
        """Loads and resizes 4-channel ground truth mask tensor of shape (4, H, W)."""
        if not self.has_mask(img_id):
            return None
        raw = self.load_raw_mask(img_id, 384, 384)
        t = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0)
        return F.adaptive_max_pool2d(t, (self.target_size, self.target_size))[0]

    def get_batch_masks(
        self, image_ids: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assembles batch mask tensor (B, 4, H, W) and boolean mask indicator (B,)."""
        b = len(image_ids)
        masks = torch.zeros(b, 4, self.target_size, self.target_size, device=device)
        flags = torch.zeros(b, dtype=torch.bool, device=device)
        for i, img_id in enumerate(image_ids):
            single = self.load_single_mask(img_id)
            if single is not None:
                masks[i] = single.to(device)
                flags[i] = True
        return masks, flags
