"""3D patch-cube masking for (MS-)MAE.

We tile the (Z, Y, X) volume into non-overlapping cubes of `patch_size`, sample a
random subset of cubes at `mask_ratio`, and zero them out at the input. The mask
is returned as a binary tensor at full volume resolution (1 = masked, 0 = visible)
so the loss can be computed per-voxel on the masked region only.
"""
from __future__ import annotations

from typing import Sequence

import torch


def random_cube_mask(
    shape: Sequence[int],          # (Z, Y, X) volume shape
    patch_size: Sequence[int],     # (pz, py, px)
    mask_ratio: float,
    device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (Z, Y, X) binary mask (1 = masked) where masking is at patch-cube granularity."""
    Z, Y, X = shape
    pz, py, px = patch_size
    assert Z % pz == 0 and Y % py == 0 and X % px == 0, \
        f"Volume {shape} must be divisible by patch {patch_size}"

    nz, ny, nx = Z // pz, Y // py, X // px
    n_patches = nz * ny * nx
    n_masked = int(round(n_patches * mask_ratio))

    perm = torch.randperm(n_patches, generator=generator, device=device)
    masked_idx = perm[:n_masked]

    flat = torch.zeros(n_patches, device=device, dtype=torch.float32)
    flat[masked_idx] = 1.0
    grid = flat.view(nz, ny, nx)

    # Upsample to voxel resolution via repeat_interleave per axis.
    voxel_mask = grid.repeat_interleave(pz, dim=0).repeat_interleave(py, dim=1).repeat_interleave(px, dim=2)
    return voxel_mask   # (Z, Y, X) float in {0,1}


def apply_mask(image: torch.Tensor, mask: torch.Tensor, fill: float) -> torch.Tensor:
    """image: (B, 1, Z, Y, X); mask: (B, Z, Y, X)."""
    m = mask.unsqueeze(1)   # (B,1,Z,Y,X)
    return image * (1 - m) + fill * m
