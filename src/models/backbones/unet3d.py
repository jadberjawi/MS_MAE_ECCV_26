"""Shared 3D U-Net encoder + matching decoder.

Design:
  - `UNet3DEncoder` is the SHARED backbone. The SAME class is built for SSL and
    for finetuning so weight transfer is trivially key-compatible.
  - `UNet3DEncoder.forward` returns a list of per-stage feature maps from shallow
    to deep. Downstream heads/decoders consume this list however they like:
        * MAE weak decoder: only the deepest feature, no skips.
        * Jigsaw head: only the deepest feature, global-pooled.
        * Segmentation decoder: all stages, with skip connections.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Building blocks ----------

def _norm(name: str, channels: int) -> nn.Module:
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if name == "batch":
        return nn.BatchNorm3d(channels)
    if name == "group":
        return nn.GroupNorm(num_groups=max(1, channels // 8), num_channels=channels)
    if name in (None, "none", ""):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, norm: str, activation: str, dropout: float):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm = _norm(norm, out_c)
        self.act = _act(activation)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class StageBlock(nn.Module):
    """A stack of conv blocks; channels go in_c -> out_c on the first block."""
    def __init__(self, in_c: int, out_c: int, num_blocks: int, norm: str, activation: str, dropout: float):
        super().__init__()
        layers = [ConvBlock(in_c, out_c, norm, activation, dropout)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_c, out_c, norm, activation, dropout))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


# ---------- Encoder ----------

class UNet3DEncoder(nn.Module):
    """3D U-Net encoder. Returns a list of feature maps from shallow to deep.

    Channel widths follow `base_channels * channel_mult[i]`. Each stage runs
    `num_res_blocks` conv blocks at its width; between stages we downsample by
    strided conv (stride=2).
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 1,
        norm: str = "instance",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channel_mult = tuple(channel_mult)
        self.stage_channels: List[int] = [base_channels * m for m in channel_mult]

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev_c = in_channels
        for i, c in enumerate(self.stage_channels):
            self.stages.append(StageBlock(prev_c, c, num_res_blocks, norm, activation, dropout))
            if i < len(self.stage_channels) - 1:
                self.downs.append(nn.Conv3d(c, c, kernel_size=2, stride=2, bias=False))
            prev_c = c

    @property
    def out_channels(self) -> int:
        return self.stage_channels[-1]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return feats   # shallow -> deep


# ---------- Segmentation decoder ----------

class UNet3DDecoder(nn.Module):
    """Symmetric U-Net decoder using the encoder's stage feature maps.

    Consumes `feats = encoder(x)` (shallow->deep) and outputs logits at the
    full input resolution with `out_channels` classes.
    """
    def __init__(
        self,
        encoder: UNet3DEncoder,
        out_channels: int,
        norm: str = "instance",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = list(encoder.stage_channels)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # From deep -> shallow: at each step, upsample, concat skip from one shallower stage.
        for i in range(len(ch) - 1, 0, -1):
            in_c = ch[i]
            skip_c = ch[i - 1]
            self.ups.append(nn.ConvTranspose3d(in_c, skip_c, kernel_size=2, stride=2, bias=False))
            self.blocks.append(StageBlock(skip_c + skip_c, skip_c, num_blocks=2,
                                          norm=norm, activation=activation, dropout=dropout))

        self.head = nn.Conv3d(ch[0], out_channels, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        # iterate deep->shallow
        for k, (up, block) in enumerate(zip(self.ups, self.blocks)):
            skip = feats[-2 - k]
            x = up(x)
            # Defensive size-match in case input shape isn't perfectly divisible by 2^depth.
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)
