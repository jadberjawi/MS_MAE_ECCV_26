"""Encoder weight transfer between SSL pretraining and supervised finetuning.

Pretraining and finetuning use the SAME encoder class (configs/backbone/unet3d.yaml).
Lightning saves the LightningModule's full state_dict, prefixed by the attribute
name (e.g. "encoder.stages.0.conv.weight"). When EMA is enabled during
pretraining we ALSO get an `ema_encoder_state_dict` key in the checkpoint; we
prefer that one because it's typically 1-3% better than the live encoder.

This module copies only the encoder weights into a freshly-built encoder,
ignoring any task-specific heads / decoders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


ENCODER_PREFIX = "encoder."
EMA_CKPT_KEY = "ema_encoder_state_dict"


def load_pretrained_encoder(
    encoder: nn.Module,
    ckpt_path: str | Path,
    map_location: str = "cpu",
    strict: bool = True,
    prefer_ema: bool = True,
) -> Tuple[int, int, str]:
    """Copy encoder weights from a Lightning .ckpt into `encoder` in-place.

    If `prefer_ema=True` and the checkpoint contains `ema_encoder_state_dict`,
    those weights are used. Otherwise we fall back to the `encoder.*` keys in
    the regular `state_dict`.

    Returns (n_loaded, n_total, source) where source is "ema" or "live".
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    source = "live"
    enc_sd = {}
    if prefer_ema and EMA_CKPT_KEY in state:
        for k, v in state[EMA_CKPT_KEY].items():
            if k.startswith(ENCODER_PREFIX):
                enc_sd[k[len(ENCODER_PREFIX):]] = v
        source = "ema"

    if not enc_sd:
        full_sd = state["state_dict"] if "state_dict" in state else state
        for k, v in full_sd.items():
            if k.startswith(ENCODER_PREFIX):
                enc_sd[k[len(ENCODER_PREFIX):]] = v

    if not enc_sd:
        raise RuntimeError(
            f"No encoder weights in {ckpt_path}. "
            "Did pretraining attach the encoder as `self.encoder`?"
        )

    missing, unexpected = encoder.load_state_dict(enc_sd, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Encoder state_dict mismatch.\nMissing: {missing}\nUnexpected: {unexpected}"
        )
    # Report state_dict-entry counts (params + buffers) for an apples-to-apples
    # number — with BatchNorm the encoder has more state_dict entries than
    # learnable parameters (running_mean/var/num_batches_tracked buffers).
    return len(enc_sd), len(encoder.state_dict()), source
