"""Shared builders: trainer + callbacks. The wandb logger is built explicitly
via src/utils/wandb_init.py (so the init line is visible at the call site)."""
from __future__ import annotations

from typing import List, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)


def build_callbacks(trainer_cfg: DictConfig, ckpt_dir: str, has_logger: bool = True) -> List[pl.Callback]:
    cbs: List[pl.Callback] = []
    cb_cfg = trainer_cfg.callbacks

    ck = cb_cfg.checkpoint
    cbs.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=str(ck.filename),
        save_top_k=int(ck.save_top_k),
        monitor=str(ck.monitor),
        mode=str(ck.mode),
        save_last=bool(ck.save_last),
        auto_insert_metric_name=False,
    ))

    es = cb_cfg.early_stopping
    if bool(es.enable):
        cbs.append(EarlyStopping(monitor=str(es.monitor), patience=int(es.patience), mode=str(es.mode)))

    if has_logger:
        cbs.append(LearningRateMonitor(logging_interval="epoch"))
    return cbs


def build_trainer(trainer_cfg: DictConfig, callbacks: List[pl.Callback],
                  logger: Optional[pl.loggers.Logger], default_root_dir: str) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=int(trainer_cfg.max_epochs),
        accelerator=str(trainer_cfg.accelerator),
        devices=trainer_cfg.devices,
        precision=str(trainer_cfg.precision),
        log_every_n_steps=int(trainer_cfg.log_every_n_steps),
        check_val_every_n_epoch=int(trainer_cfg.check_val_every_n_epoch),
        gradient_clip_val=float(trainer_cfg.gradient_clip_val),
        accumulate_grad_batches=int(trainer_cfg.accumulate_grad_batches),
        deterministic=bool(trainer_cfg.deterministic),
        num_sanity_val_steps=int(trainer_cfg.num_sanity_val_steps),
        callbacks=callbacks,
        logger=logger if logger is not None else False,
        default_root_dir=default_root_dir,
    )
