"""Semi-supervised training mask provider strictly indexing the training split."""

import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class TrainMaskProvider:
    """Manages ground-truth lesion segmentation masks for semi-supervised training.

    Academic Integrity Guarantee: Strictly indexes only the 'train' directory.
    Never indexes or accesses 'valid' or 'test' directories to prevent data leakage.
    """

    LESIONS = ("EX", "HE", "MA", "SE")
    SEARCH_ROOTS = (
        "data/DDR-dataset/lesion_segmentation/train",
        "data/DDR/lesion_segmentation/train",
        "/content/CLAT/data/DDR-dataset/lesion_segmentation/train",
        "/content/CLAT/data/DDR/lesion_segmentation/train",
        "/content/DDR_extracted/DDR-dataset/lesion_segmentation/train",
    )

    def __init__(self, target_size: int = 24) -> None:
        """Initializes the mask provider.

        Args:
            target_size: Spatial grid size matching ViT patch tokens (e.g. 24 for 384x384).
        """
        self.target_size = target_size
        self._index: Dict[str, Dict[str, str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Indexes all masks found strictly in training directories."""
        for root in self.SEARCH_ROOTS:
            if not os.path.exists(root):
                continue
            for lesion in self.LESIONS:
                patterns = [
                    os.path.join(root, "label", lesion, "*.*"),
                    os.path.join(root, lesion, "*.*"),
                ]
                for pattern in patterns:
                    for path in glob.glob(pattern):
                        if path.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                            img_id = os.path.splitext(os.path.basename(path))[0]
                            if img_id not in self._index:
                                self._index[img_id] = {}
                            self._index[img_id][lesion] = path

    def has_mask(self, img_id: str) -> bool:
        """Checks if a training mask exists for the given image ID."""
        return img_id in self._index

    def load_single_mask(self, img_id: str) -> Optional[torch.Tensor]:
        """Loads and resizes 4-channel ground truth mask tensor of shape (4, H, W)."""
        if not self.has_mask(img_id):
            return None
        lesion_files = self._index[img_id]
        channel_tensors = []
        for lesion in self.LESIONS:
            if lesion in lesion_files:
                pil_mask = Image.open(lesion_files[lesion]).convert("L")
                arr = (np.array(pil_mask) > 127).astype(np.float32)
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                # Adaptive max pool preserves sparse punctate lesion signals
                pooled = F.adaptive_max_pool2d(t, (self.target_size, self.target_size))
                channel_tensors.append(pooled.squeeze(0))
            else:
                channel_tensors.append(torch.zeros(1, self.target_size, self.target_size))
        return torch.cat(channel_tensors, dim=0)

    def get_batch_masks(
        self, image_ids: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assembles batch mask tensor (B, 4, H, W) and boolean mask indicator (B,)."""
        batch_size = len(image_ids)
        masks = torch.zeros(batch_size, 4, self.target_size, self.target_size, device=device)
        has_mask_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i, img_id in enumerate(image_ids):
            single = self.load_single_mask(img_id)
            if single is not None:
                masks[i] = single.to(device)
                has_mask_flags[i] = True

        return masks, has_mask_flags
