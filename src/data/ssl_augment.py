"""Augmentation for SSL pretraining, with the denoising-SSL contract.

Two augmentation TYPES, handled differently (see the discussion in the project):

  * GEOMETRIC (rotation, translation) -> applied to the shared base volume, so it
    affects BOTH the reconstruction target and the model input IDENTICALLY. This
    just enriches the pose distribution; it does not ask the model to "undo" pose.

  * APPEARANCE (intensity scale, Poisson-like noise) -> applied to the INPUT view
    ONLY, while the target stays CLEAN. The model must therefore learn to remove
    the nuisance to reconstruct the clean target -> it learns a representation
    INVARIANT to intensity / noise, exactly the axes that differ across tracers
    and centers.

`make_ssl_views` returns (target, input):
    target = geometric(clean)                  # clean, for reconstruction
    input  = appearance(geometric(clean))      # corrupted, fed to the encoder

For non-reconstructive pretexts (jigsaw) there is no clean target; the input view
is what gets tiled/shuffled, which is the correct behaviour.

All ops reuse the SPECT-safe transforms in src/data/transforms.py so the physics
constraints (no flips, short-axis rotation only, multiplicative intensity, Poisson
noise) are identical to finetuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .transforms import (
    _maybe, random_rotation_z, random_translation,
    random_intensity_scale, poisson_like_noise,
)


@dataclass
class SSLAugmentConfig:
    enable: bool = False

    # GEOMETRIC (applied to both target and input, identically).
    rotation_deg: float = 8.0
    p_rotation: float = 0.7
    translation_voxels: int = 3
    p_translation: float = 0.7

    # APPEARANCE (applied to the INPUT view only; target stays clean).
    intensity_scale: float = 0.10
    p_intensity: float = 0.5
    poisson_strength: float = 0.05
    p_noise: float = 0.5


def make_ssl_views(img: np.ndarray, cfg: Optional[SSLAugmentConfig],
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (target, input) views of one (Z,Y,X) volume."""
    if cfg is None or not cfg.enable:
        return img, img

    # Geometric on the shared base -> defines BOTH views identically.
    if _maybe(rng, cfg.p_rotation):
        img, _ = random_rotation_z(img, None, cfg.rotation_deg, rng)
    if _maybe(rng, cfg.p_translation):
        img, _ = random_translation(img, None, cfg.translation_voxels, rng)
    target = img

    # Appearance on the INPUT view only -> target stays clean.
    inp = img
    if _maybe(rng, cfg.p_intensity):
        inp = random_intensity_scale(inp, cfg.intensity_scale, rng)
    if _maybe(rng, cfg.p_noise):
        inp = poisson_like_noise(inp, cfg.poisson_strength, rng)
    return target, inp
