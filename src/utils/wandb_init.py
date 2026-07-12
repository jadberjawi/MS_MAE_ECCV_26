"""Explicit wandb initialization.

We deliberately call `wandb.init(...)` ourselves (instead of letting Lightning's
WandbLogger do it implicitly) so the init parameters are visible at the call
site. Lightning's WandbLogger transparently attaches to the active wandb run
when one exists, so the rest of the trainer machinery (self.log calls, etc.)
keeps working unchanged.

Auth:
    Set WANDB_API_KEY in your shell or .sh script. Do not check the key into
    version control. The .sh files include a commented `export WANDB_API_KEY=...`
    line as a reminder.

Disabling:
    `use_wandb: false` at the top of the run config, or `logger.mode=disabled`,
    or `WANDB_MODE=disabled` env var.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger, WandbLogger


def init_wandb(
    cfg: DictConfig,
    run_name: str,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Logger]:
    """Initialise wandb (explicitly) and return a Lightning WandbLogger that
    attaches to the active run. Returns None when wandb is disabled.

    The full Hydra config is logged to wandb under the run config, plus any
    `extra_config` (e.g. fold index for CV runs).
    """
    if not bool(cfg.get("use_wandb", False)):
        return None

    logger_cfg = cfg.get("logger")
    if logger_cfg is None:
        return None
    if str(logger_cfg.get("mode", "online")) == "disabled":
        return None

    save_dir = str(logger_cfg.get("save_dir", "."))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    init_kwargs = dict(
        project=str(logger_cfg.project),
        entity=logger_cfg.get("entity") or None,
        name=run_name,
        group=str(logger_cfg.get("group", "")) or None,
        tags=list(logger_cfg.get("tags", [])) or None,
        notes=str(logger_cfg.get("notes", "")) or None,
        mode=str(logger_cfg.get("mode", "online")),
        dir=save_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        reinit="finish_previous",   # finish any prior run (e.g. previous fold) cleanly
    )

    # *** THIS is the line that mirrors `if use_wandb: wandb.init(...)` ***
    wandb.init(**init_kwargs)

    if extra_config:
        wandb.config.update(dict(extra_config), allow_val_change=True)

    # Lightning's WandbLogger detects the active run and attaches to it.
    return WandbLogger(log_model=bool(logger_cfg.get("log_model", False)))


def finish_wandb() -> None:
    """End the current wandb run. Safe to call when no run is active."""
    if wandb.run is not None:
        wandb.finish()
