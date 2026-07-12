"""Dataset for supervised LV segmentation from cached labeled .npz volumes.

Augmentations are SPECT-safe (see src/data/transforms.py) and are applied to
numpy arrays before the conversion to torch tensors, so scipy-based ops
(rotation, elastic, etc.) work natively without device round-trips.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_cached_labeled
from .transforms import SegAugmentConfig, apply_seg_augment


class SegVolumeDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        indices: Optional[List[int]] = None,
        augment: Optional[SegAugmentConfig] = None,
        seed: int = 0,
    ):
        cache_dir = Path(cache_dir)
        self.files: List[Path] = sorted(cache_dir.glob("*.npz"))
        if indices is not None:
            self.files = [self.files[i] for i in indices]
        if not self.files:
            raise FileNotFoundError(f"No labeled .npz files in {cache_dir}.")
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        img_np, msk_np = load_cached_labeled(self.files[idx])         # (Z,Y,X) float32 / uint8
        if self.augment is not None and self.augment.enable:
            img_np, msk_np = apply_seg_augment(img_np, msk_np, self.augment, self._rng)

        img = torch.from_numpy(np.ascontiguousarray(img_np)).unsqueeze(0).float()   # (1,Z,Y,X)
        msk = torch.from_numpy(np.ascontiguousarray(msk_np)).long()                  # (Z,Y,X)
        return {"image": img, "mask": msk, "name": self.files[idx].stem}
