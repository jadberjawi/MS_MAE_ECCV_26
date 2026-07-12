"""Dataset for self-supervised pretraining from cached unlabeled .npz volumes.

Returns two views per item:
    "image"      -> CLEAN volume (geometric aug applied) = the reconstruction target
    "image_aug"  -> input view (clean + appearance aug)  = what the encoder sees

When augment is None/disabled, "image_aug" == "image" (current behaviour). Pretext
transforms (masking, jigsaw shuffle) still live in the strategy's training_step;
this only handles the geometric/appearance augmentation split (see ssl_augment.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_cached_unlabeled
from .ssl_augment import SSLAugmentConfig, make_ssl_views


class SSLVolumeDataset(Dataset):
    def __init__(self, cache_dir: str | Path, indices: Optional[List[int]] = None,
                 augment: Optional[SSLAugmentConfig] = None, seed: int = 0):
        cache_dir = Path(cache_dir)
        self.files: List[Path] = sorted(cache_dir.glob("*.npz"))
        if indices is not None:
            self.files = [self.files[i] for i in indices]
        if not self.files:
            raise FileNotFoundError(f"No .npz files in {cache_dir}. Run scripts/prepare_data.py first.")
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        img = load_cached_unlabeled(self.files[idx])             # (Z, Y, X) float32 in [0,1]
        target, inp = make_ssl_views(img, self.augment, self._rng)
        x = torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0).float()      # (1,Z,Y,X)
        x_aug = torch.from_numpy(np.ascontiguousarray(inp)).unsqueeze(0).float()
        return {"image": x, "image_aug": x_aug, "name": self.files[idx].stem}
