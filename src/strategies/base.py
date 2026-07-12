"""Base LightningModule for pretraining strategies.

Each strategy subclasses this and implements `pretext_step(batch)` returning a
dict with at minimum `{"loss": Tensor}` and any scalar metrics to log.

Encoder is stored as `self.encoder` so the encoder-transfer utility can locate
its weights uniformly via the "encoder." prefix.
"""
from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from hydra.utils import get_class


class BasePretrainStrategy(pl.LightningModule):
    def __init__(self, encoder, optim_cfg: Dict[str, Any]):
        super().__init__()
        self.encoder = encoder
        self._optim_cfg = optim_cfg

    # Subclasses implement this.
    def pretext_step(self, batch: dict, stage: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        out = self.pretext_step(batch, "train")
        self._log_dict(out, prefix="train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.pretext_step(batch, "val")
        self._log_dict(out, prefix="val")
        return out["loss"]

    def _log_dict(self, out: Dict[str, torch.Tensor], prefix: str) -> None:
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 0:
                self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss"))

    def configure_optimizers(self):
        opt_cfg = dict(self._optim_cfg["optimizer"])
        OptCls = get_class(opt_cfg.pop("_target_"))
        optimizer = OptCls(self.parameters(), **opt_cfg)
        sched_cfg = self._optim_cfg.get("scheduler")
        if sched_cfg is None:
            return optimizer
        sched_cfg = dict(sched_cfg)
        SchedCls = get_class(sched_cfg.pop("_target_"))
        scheduler = SchedCls(optimizer, **sched_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
