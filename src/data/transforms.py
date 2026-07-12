"""SPECT-safe 3D augmentations.

Physics / anatomy constraints respected:

1. NO axis flips by default.
   The heart is chiral: LV walls (anterior/inferior/septal/lateral) sit in fixed
   anatomical positions, and apex/base are distinct along the long axis. A left-right
   or apex-base flip creates a mirror-image heart that is biologically impossible
   and will degrade the model.

2. Rotations restricted to the SHORT-AXIS plane only.
   Volumes are stored (Z, Y, X) where Z is the long (apex-base) axis. Rotation
   around Z (in the short-axis plane) is fine in small amounts — it just
   simulates patient positioning variability. Rotation around X or Y would tip
   apex into base territory — forbidden by default.

3. Intensity transforms are SCALE-only and small.
   SPECT counts are physically meaningful (radioactive decay → photon counts).
   Volumes are normalized to [0, 1] at cache time, which already removes absolute
   scale; we allow a small multiplicative jitter to simulate scanner gain /
   sensitivity variation. We do NOT add an offset — true zero must stay zero
   (background = no counts).

4. Noise model is Poisson-like, not Gaussian.
   SPECT acquisition noise is Poisson (photon counting). After [0,1] normalization
   this approximates a Gaussian with sigma proportional to sqrt(intensity).
   Adding ordinary uniform-sigma Gaussian noise is non-physical.

5. Elastic deformation kept very small.
   Useful to capture inter-patient anatomical variation, but large deformations
   warp the LV myocardium into shapes that don't occur clinically. Default alpha
   is small and the warp is smooth (large sigma).

All transforms operate on numpy arrays in canonical (Z, Y, X) order. The
SegVolumeDataset converts to torch AFTER augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate as nd_rotate, shift as nd_shift


@dataclass
class SegAugmentConfig:
    enable: bool = True

    # In-plane (short-axis) rotation only. Axis-1 is Y, axis-2 is X in our (Z,Y,X) layout.
    rotation_deg: float = 8.0           # max ± degrees around the long (Z) axis
    p_rotation: float = 0.7

    # Voxel-level translation (centering jitter). Same scale in all 3 axes is fine.
    translation_voxels: int = 3         # max ± voxels per axis
    p_translation: float = 0.7

    # Multiplicative intensity scale only (no shift — preserve true zero).
    intensity_scale: float = 0.10       # max ± fractional scale
    p_intensity: float = 0.5

    # Poisson-like multiplicative noise: sigma_voxel = strength * sqrt(intensity).
    poisson_strength: float = 0.05
    p_noise: float = 0.5

    # Elastic deformation. Keep alpha small and sigma large (smooth warps).
    elastic_alpha: float = 0.0          # default OFF — set to 2-4 to enable
    elastic_sigma: float = 6.0
    p_elastic: float = 0.3

    # Flips are OFF by default. Cardiac anatomy is chiral; flipping breaks
    # apex/base or septal/lateral semantics. Only enable per-axis if you know
    # your data has been canonicalized such that flipping is meaningful.
    flip_axes: List[int] = field(default_factory=list)   # e.g. [] (default), [1] for Y-only
    p_flip: float = 0.5


# ---------- individual transforms ----------

def _maybe(rng: np.random.Generator, p: float) -> bool:
    return rng.random() < p


def random_rotation_z(img: np.ndarray, msk: Optional[np.ndarray], max_deg: float,
                      rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rotate around the long (Z) axis by a small random angle.

    Input layout: (Z, Y, X). Rotation axes = (1, 2) (the Y/X plane).
    """
    if max_deg <= 0:
        return img, msk
    angle = float(rng.uniform(-max_deg, max_deg))
    img_r = nd_rotate(img, angle=angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
    msk_r = nd_rotate(msk, angle=angle, axes=(1, 2), reshape=False, order=0, mode="constant", cval=0) \
        if msk is not None else None
    return img_r.astype(img.dtype, copy=False), (msk_r.astype(msk.dtype, copy=False) if msk is not None else None)


def random_translation(img: np.ndarray, msk: Optional[np.ndarray], max_voxels: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if max_voxels <= 0:
        return img, msk
    shifts = [int(rng.integers(-max_voxels, max_voxels + 1)) for _ in range(3)]
    img_s = nd_shift(img, shift=shifts, order=1, mode="nearest")
    msk_s = nd_shift(msk, shift=shifts, order=0, mode="constant", cval=0) if msk is not None else None
    return img_s.astype(img.dtype, copy=False), (msk_s.astype(msk.dtype, copy=False) if msk is not None else None)


def random_intensity_scale(img: np.ndarray, max_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative scale only, clipped to [0,1] to preserve normalization.

    NO additive shift — that would move 'no counts' away from zero, which is not
    physically meaningful for SPECT.
    """
    if max_scale <= 0:
        return img
    s = 1.0 + float(rng.uniform(-max_scale, max_scale))
    return np.clip(img * s, 0.0, 1.0).astype(img.dtype, copy=False)


def poisson_like_noise(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """sigma proportional to sqrt(intensity), then clip to [0,1].

    Approximates the Poisson noise of SPECT acquisition under the normalization
    we've already applied. `strength` is a dimensionless fraction.
    """
    if strength <= 0:
        return img
    sigma = strength * np.sqrt(np.clip(img, 0.0, 1.0))
    noise = rng.standard_normal(img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0.0, 1.0).astype(img.dtype, copy=False)


def random_elastic_3d(img: np.ndarray, msk: Optional[np.ndarray], alpha: float, sigma: float,
                      rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Small, smooth 3D elastic warp. alpha controls strength, sigma controls smoothness.

    Default alpha=0 in the config — turn on only if you've validated visually
    that the resulting deformations are anatomically plausible.
    """
    if alpha <= 0:
        return img, msk
    shape = img.shape
    dz = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    coords = np.stack([z + dz, y + dy, x + dx], axis=0)
    img_w = map_coordinates(img, coords, order=1, mode="nearest").reshape(shape)
    msk_w = map_coordinates(msk, coords, order=0, mode="constant", cval=0).reshape(shape) \
        if msk is not None else None
    return img_w.astype(img.dtype, copy=False), (msk_w.astype(msk.dtype, copy=False) if msk is not None else None)


def maybe_flip(img: np.ndarray, msk: Optional[np.ndarray], axes: Sequence[int],
               p: float, rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    for ax in axes:
        if rng.random() < p:
            img = np.flip(img, axis=ax).copy()
            if msk is not None:
                msk = np.flip(msk, axis=ax).copy()
    return img, msk


# ---------- composition ----------

def apply_seg_augment(img_np: np.ndarray, msk_np: np.ndarray, cfg: SegAugmentConfig,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Compose the SPECT-safe pipeline.

    Order matters: geometric ops first (rotation, translation, elastic), then
    intensity-domain ops (scale, noise). Flips, if explicitly enabled, come
    after geometry to keep the mask aligned.
    """
    if not cfg.enable:
        return img_np, msk_np

    if _maybe(rng, cfg.p_rotation):
        img_np, msk_np = random_rotation_z(img_np, msk_np, cfg.rotation_deg, rng)
    if _maybe(rng, cfg.p_translation):
        img_np, msk_np = random_translation(img_np, msk_np, cfg.translation_voxels, rng)
    if _maybe(rng, cfg.p_elastic):
        img_np, msk_np = random_elastic_3d(img_np, msk_np, cfg.elastic_alpha, cfg.elastic_sigma, rng)
    if cfg.flip_axes:
        img_np, msk_np = maybe_flip(img_np, msk_np, cfg.flip_axes, cfg.p_flip, rng)
    if _maybe(rng, cfg.p_intensity):
        img_np = random_intensity_scale(img_np, cfg.intensity_scale, rng)
    if _maybe(rng, cfg.p_noise):
        img_np = poisson_like_noise(img_np, cfg.poisson_strength, rng)
    return img_np, msk_np
