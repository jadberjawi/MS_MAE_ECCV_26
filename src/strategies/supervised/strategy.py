"""Supervised LV segmentation LightningModule.

Encoder is the SHARED U-Net encoder (configs/backbone/unet3d.yaml). Decoder is a
full symmetric U-Net decoder with skip connections (configs/finetune/*.yaml).

Init modes:
  - from_scratch: pretrained_ckpt=null
  - from_pretrained: pretrained_ckpt=<path-to-pretrain-.ckpt>; encoder weights
    are loaded via src/utils/checkpoint.py and a discriminative LR applied
    (encoder_lr_mult < 1).

Metrics logged per epoch:
  train: loss
  val:   loss, dice, iou, sensitivity, precision, hd95, asd
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.utils import get_class

from ...losses.dice import DiceCELoss
from ...metrics import seg_metrics as M
from ...utils.checkpoint import load_pretrained_encoder


class SupervisedSegStrategy(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder: Dict[str, Any],
        loss: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        pretrained_ckpt: Optional[str] = None,
        freeze_encoder_epochs: int = 0,
        encoder_lr_mult: float = 1.0,
        spacing=(4.0, 4.0, 4.0),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = hydra.utils.instantiate(decoder, encoder=encoder)
        self.criterion = DiceCELoss(
            dice_weight=float(loss.get("dice_weight", 1.0)),
            ce_weight=float(loss.get("ce_weight", 1.0)),
            include_background=bool(loss.get("include_background", False)),
        )
        self._optim_cfg = optim_cfg
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_lr_mult = float(encoder_lr_mult)
        self.spacing = tuple(spacing)

        if pretrained_ckpt:
            n_loaded, n_total, source = load_pretrained_encoder(self.encoder, pretrained_ckpt, strict=True)
            print(f"[finetune] Loaded {n_loaded} encoder tensors ({source}) from {pretrained_ckpt} "
                  f"(encoder has {n_total} params).")

    # --------- training ---------
    def forward(self, x):
        feats = self.encoder(x)
        logits = self.decoder(feats)
        # Safety: ensure logits match the input spatial size so the loss/metrics
        # (computed against masks at input resolution) always align. No-op for the
        # U-Net decoder (already full-res); fixes single-scale/foundation decoders
        # whose output resolution can differ from the input.
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def on_train_epoch_start(self):
        if self.freeze_encoder_epochs <= 0:
            return
        frozen = self.current_epoch < self.freeze_encoder_epochs
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
        self.log("train/encoder_frozen", float(frozen), on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx):
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["mask"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, prefix="test")

    def _eval_step(self, batch, prefix: str):
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["mask"])
        self.log(f"{prefix}/loss", loss, on_epoch=True, prog_bar=(prefix == "val"))
        self.log(f"{prefix}/dice", M.dice_score(logits, batch["mask"]), on_epoch=True, prog_bar=(prefix == "val"))
        self.log(f"{prefix}/iou", M.iou_score(logits, batch["mask"]), on_epoch=True)
        self.log(f"{prefix}/sensitivity", M.sensitivity(logits, batch["mask"]), on_epoch=True)
        self.log(f"{prefix}/precision", M.precision(logits, batch["mask"]), on_epoch=True)
        try:
            self.log(f"{prefix}/hd95", M.hausdorff_95(logits, batch["mask"], spacing=self.spacing), on_epoch=True)
            self.log(f"{prefix}/asd", M.average_surface_distance(logits, batch["mask"], spacing=self.spacing), on_epoch=True)
        except Exception:
            pass
        return loss

    # --------- optim ---------
    def configure_optimizers(self):
        opt_cfg = dict(self._optim_cfg["optimizer"])
        OptCls = get_class(opt_cfg.pop("_target_"))
        base_lr = float(opt_cfg.get("lr", 1e-3))

        if self.encoder_lr_mult == 1.0:
            optimizer = OptCls(self.parameters(), **opt_cfg)
        else:
            enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
            dec_params = [p for p in self.decoder.parameters() if p.requires_grad]
            # Drop empty groups — a fully frozen encoder (e.g. frozen SAM-Med3D)
            # contributes no trainable params, and an empty group can error.
            param_groups = []
            if enc_params:
                param_groups.append({"params": enc_params, "lr": base_lr * self.encoder_lr_mult})
            if dec_params:
                param_groups.append({"params": dec_params, "lr": base_lr})
            if not param_groups:
                raise RuntimeError("No trainable parameters (encoder frozen and decoder has none?).")
            optimizer = OptCls(param_groups, **opt_cfg)

        sched_cfg = self._optim_cfg.get("scheduler")
        if sched_cfg is None:
            return optimizer
        sched_cfg = dict(sched_cfg)
        SchedCls = get_class(sched_cfg.pop("_target_"))
        scheduler = SchedCls(optimizer, **sched_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
