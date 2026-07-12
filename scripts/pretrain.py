"""Entry point: MS-MAE self-supervised pretraining.

Examples:
    python scripts/pretrain.py
    python scripts/pretrain.py pretrain.mask_ratio=0.75 trainer.max_epochs=300
    python scripts/pretrain.py use_wandb=false trainer.max_epochs=2   # quick smoke test

Outputs a Lightning checkpoint under
    outputs/pretrain/<run_name>/checkpoints/last.ckpt
which scripts/finetune.py consumes via finetune.pretrained_ckpt=...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.splits import train_val_holdout
from src.data.ssl_dataset import SSLVolumeDataset
from src.data.ssl_augment import SSLAugmentConfig
from src.engine.build import build_callbacks, build_trainer
from src.engine.ema import EMAEncoderCallback
from src.utils.hydra_utils import instantiate_drop
from src.utils.seed import seed_everything
from src.utils.wandb_init import finish_wandb, init_wandb


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="pretrain")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(int(cfg.seed))
    pl.seed_everything(int(cfg.seed), workers=True)

    # ----- Data -----
    cache_dir = Path(cfg.data.cache_dir)
    n_total = len(sorted(cache_dir.glob("*.npz")))
    if n_total == 0:
        raise FileNotFoundError(
            f"No cached unlabeled .npz at {cache_dir}. Run scripts/prepare_data.py first."
        )

    if n_total < 2:
        train_idx, val_idx = [0], [0]      # trivial single-sample smoke test
    else:
        train_idx, val_idx = train_val_holdout(n_total, float(cfg.data.val_fraction), int(cfg.seed))

    # Augmentation: TRAIN only (val is never augmented, so val loss is comparable
    # across runs). Disable via `data.augment.enable=false`.
    aug_cfg = OmegaConf.select(cfg, "data.augment", default=None)
    train_aug = None
    if aug_cfg is not None and bool(aug_cfg.get("enable", False)):
        train_aug = SSLAugmentConfig(**OmegaConf.to_container(aug_cfg, resolve=True))
        print("[augment] SSL train augmentation ON.")

    train_ds = SSLVolumeDataset(cache_dir, indices=list(train_idx), augment=train_aug, seed=int(cfg.seed))
    val_ds = SSLVolumeDataset(cache_dir, indices=list(val_idx), augment=None)

    loader_kw = dict(
        batch_size=int(cfg.data.loader.batch_size),
        num_workers=int(cfg.data.loader.num_workers),
        pin_memory=bool(cfg.data.loader.pin_memory),
        persistent_workers=bool(cfg.data.loader.persistent_workers),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ----- Model: shared encoder + MS-MAE strategy -----
    encoder = instantiate_drop(cfg.backbone)
    strategy = instantiate_drop(
        cfg.pretrain,
        encoder=encoder,
        optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
    )

    # ----- Trainer -----
    ckpt_dir = str(Path(cfg.paths.ckpt_dir))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = init_wandb(cfg, run_name=str(cfg.run_name))
    callbacks = build_callbacks(cfg.trainer, ckpt_dir=ckpt_dir, has_logger=logger is not None)

    # EMA shadow copy of the encoder. Saved under `ema_encoder_state_dict` in each
    # checkpoint; the finetune loader prefers it (typically +1-3% Dice for free).
    ema_cfg = cfg.get("ema", None)
    if ema_cfg is not None and bool(ema_cfg.get("enable", False)):
        callbacks.append(EMAEncoderCallback(
            decay=float(ema_cfg.decay),
            use_for_validation=bool(ema_cfg.use_for_validation),
            save_in_ckpt=bool(ema_cfg.save_in_ckpt),
        ))
        print(f"[ema] enabled, decay={ema_cfg.decay}")

    trainer = build_trainer(cfg.trainer, callbacks=callbacks, logger=logger,
                            default_root_dir=str(cfg.paths.output_dir))

    trainer.fit(strategy, train_loader, val_loader)
    finish_wandb()
    print(f"\nPretraining done. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
