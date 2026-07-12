"""Exponential moving average (EMA) of the pretraining encoder.

Why: the encoder weights at the end of pretraining are a noisy point in
parameter space. An EMA shadow copy averages over the recent trajectory and
typically gives 1–3% better linear-probe / finetune Dice with no extra training
cost. This is the same trick used by MoCo/BYOL/DINO and by most modern MAE
implementations as a stabiliser.

Mechanics:
    after every train step:  ema_θ ← decay · ema_θ + (1 − decay) · θ
    during validation:       temporarily replace the live encoder with ema_θ
                             so val metrics + the linear probe reflect the
                             encoder we'd actually transfer.
    on checkpoint save:      attach the EMA state dict under
                             `ema_encoder_state_dict` so the finetune script
                             can pick it up.

Buffers (e.g. BatchNorm running stats) are copied — they are not parameters
and don't benefit from blending.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


EMA_CKPT_KEY = "ema_encoder_state_dict"
ENCODER_PREFIX = "encoder."


class EMAEncoderCallback(pl.Callback):
    def __init__(
        self,
        decay: float = 0.999,
        use_for_validation: bool = True,
        save_in_ckpt: bool = True,
    ):
        super().__init__()
        self.decay = float(decay)
        self.use_for_validation = bool(use_for_validation)
        self.save_in_ckpt = bool(save_in_ckpt)

        self.ema_encoder: Optional[nn.Module] = None
        self._stash: Optional[Dict[str, torch.Tensor]] = None

    # ---------- lifecycle ----------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # pl_module is already on the right device here.
        self.ema_encoder = copy.deepcopy(pl_module.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        self.ema_encoder.eval()

    # ---------- EMA update ----------

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.ema_encoder is None:
            return
        live_sd = pl_module.encoder.state_dict()
        for k, v in self.ema_encoder.state_dict().items():
            src = live_sd[k].detach()
            if v.dtype.is_floating_point and src.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.to(v.dtype), alpha=1.0 - self.decay)
            else:
                # buffers like num_batches_tracked are integer — just copy.
                v.copy_(src)

    # ---------- val swap ----------

    def on_validation_start(self, trainer, pl_module) -> None:
        if not self.use_for_validation or self.ema_encoder is None:
            return
        self._stash = {k: v.detach().clone() for k, v in pl_module.encoder.state_dict().items()}
        pl_module.encoder.load_state_dict(self.ema_encoder.state_dict())

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._stash is None:
            return
        pl_module.encoder.load_state_dict(self._stash)
        self._stash = None

    # ---------- checkpointing ----------

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        if not self.save_in_ckpt or self.ema_encoder is None:
            return
        # Save with the "encoder." prefix so the format matches the live state_dict.
        checkpoint[EMA_CKPT_KEY] = {
            f"{ENCODER_PREFIX}{k}": v.detach().cpu() for k, v in self.ema_encoder.state_dict().items()
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        ema = checkpoint.get(EMA_CKPT_KEY)
        if ema is None or self.ema_encoder is None:
            return
        sd = {k[len(ENCODER_PREFIX):]: v for k, v in ema.items() if k.startswith(ENCODER_PREFIX)}
        self.ema_encoder.load_state_dict(sd)
