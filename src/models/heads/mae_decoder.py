"""Intentionally weak decoder for MAE-style pretraining.

Why weak: when the decoder has plenty of capacity it can reconstruct the masked
voxels from low-level cues that bypass the encoder representation. By keeping
it narrow + shallow and refusing skip connections, reconstruction quality
becomes a strong proxy for "how good is the encoder's feature".
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakMAEDecoder(nn.Module):
    def __init__(
        self,
        encoder,                       # UNet3DEncoder instance (for shape info only)
        channels: int = 16,
        num_blocks: int = 2,
        use_skips: bool = False,
        out_channels: int = 1,
    ):
        super().__init__()
        self.use_skips = use_skips

        in_c = encoder.out_channels
        self.proj_in = nn.Conv3d(in_c, channels, kernel_size=1)

        body = []
        for _ in range(num_blocks):
            body.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1))
            body.append(nn.GELU())
        self.body = nn.Sequential(*body)

        # Upsample factor must match the encoder's total downsample.
        self.depth = len(encoder.stage_channels) - 1
        self.head = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, feats: List[torch.Tensor], target_shape) -> torch.Tensor:
        x = feats[-1]                              # deepest only
        x = self.proj_in(x)
        x = self.body(x)
        # Plain trilinear upsample back to input resolution.
        x = F.interpolate(x, size=tuple(target_shape), mode="trilinear", align_corners=False)
        return self.head(x)
