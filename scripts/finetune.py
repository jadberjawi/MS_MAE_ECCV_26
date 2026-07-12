"""Entry point: supervised finetune with held-out test set + K-fold CV.

Split policy (see configs/finetune.yaml -> split:):
    1. test_fraction of all labeled samples is HELD OUT once (seeded).
       This set is touched only at the end, via trainer.test(ckpt=best).
    2. The remaining pool is K-folded. For each fold:
         - val = the held-out fold of the pool   (used for early stop / best-ckpt)
         - train = the rest of the pool          (used to fit)
       Validation and test sets NEVER overlap.

Examples:
    # From-scratch baseline, 5-fold CV with 20% held-out test:
    python scripts/finetune.py

    # From SSL-pretrained encoder:
    python scripts/finetune.py finetune=from_pretrained \
        finetune.pretrained_ckpt=outputs/pretrain/mae_unet3d_seed1337/checkpoints/last.ckpt

    # Single fold (e.g. fold 0) for debugging:
    python scripts/finetune.py split.fold=0

    # Change split ratios:
    python scripts/finetune.py split.test_fraction=0.15 split.n_folds=4

    # Quick smoke test:
    python scripts/finetune.py logger.mode=disabled trainer.max_epochs=2 \
        split.test_fraction=0.25 split.n_folds=2 trainer.callbacks.early_stopping.enable=false
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# PyTorch 2.6+ defaults `torch.load(weights_only=True)`, which refuses to load
# checkpoints that contain OmegaConf containers (Lightning sometimes saves these
# in hparams). Pre-register OmegaConf classes as "safe" so Lightning's internal
# torch.load calls succeed without needing weights_only=False everywhere.
try:
    import torch.serialization as _ts
    from omegaconf import DictConfig as _OCDict, ListConfig as _OCList
    from omegaconf.base import ContainerMetadata as _OCContainerMeta, Metadata as _OCMeta
    from omegaconf.nodes import (
        AnyNode as _OCAny, BooleanNode as _OCBool, FloatNode as _OCFloat,
        IntegerNode as _OCInt, StringNode as _OCStr,
    )
    _ts.add_safe_globals([
        _OCDict, _OCList, _OCContainerMeta, _OCMeta,
        _OCAny, _OCBool, _OCFloat, _OCInt, _OCStr,
    ])
except Exception:
    pass

from src.data.seg_dataset import SegVolumeDataset
from src.data.splits import holdout_then_kfold, kfold_indices, subsample_indices
from src.data.transforms import SegAugmentConfig
from src.engine.build import build_callbacks, build_trainer
from src.utils.hydra_utils import instantiate_drop
from src.utils.seed import seed_everything
from src.utils.wandb_init import finish_wandb, init_wandb


def _augment_cfg(cfg: DictConfig) -> SegAugmentConfig:
    """Build a SegAugmentConfig from the data.augment subtree."""
    a = OmegaConf.to_container(cfg.data.augment, resolve=True)
    return SegAugmentConfig(**a)


def _make_loader(cache_dir, indices, augment, batch_size, num_workers, pin_memory, seed, shuffle):
    # `indices=None` means "use every .npz in cache_dir" (e.g. external test set).
    ds = SegVolumeDataset(
        cache_dir=cache_dir,
        indices=(None if indices is None else list(indices)),
        augment=augment,
        seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory)


def _run_one_fold(cfg: DictConfig, fold_idx: int, train_idx, val_idx, test_idx) -> dict:
    # Label-efficiency subsample: shrink ONLY the train set; val and test stay
    # full so cross-fraction comparisons remain fair. Nested wrt seed.
    train_fraction = float(cfg.split.train_fraction)
    full_train_idx = train_idx
    train_idx = subsample_indices(
        train_idx,
        fraction=train_fraction,
        seed=int(cfg.split.seed) + fold_idx,
    )

    # External test set?  test_idx == None signals "use cfg.data.test_cache_dir".
    use_external_test = test_idx is None
    test_cache = str(cfg.data.test_cache_dir if use_external_test else cfg.data.cache_dir)
    test_indices_log = "ALL of external test dir" if use_external_test else list(test_idx)

    print(f"\n========== Fold {fold_idx} "
          f"| train={list(train_idx)} ({len(train_idx)}/{len(full_train_idx)}, "
          f"tf={train_fraction}) "
          f"| val={list(val_idx)} | test={test_indices_log} ==========")

    aug = _augment_cfg(cfg)

    train_loader = _make_loader(cfg.data.cache_dir, train_idx, augment=aug,
                                batch_size=int(cfg.data.loader.batch_size),
                                num_workers=int(cfg.data.loader.num_workers),
                                pin_memory=bool(cfg.data.loader.pin_memory),
                                seed=int(cfg.seed) + fold_idx, shuffle=True)
    val_loader = _make_loader(cfg.data.cache_dir, val_idx, augment=None,
                              batch_size=int(cfg.data.loader.batch_size),
                              num_workers=int(cfg.data.loader.num_workers),
                              pin_memory=bool(cfg.data.loader.pin_memory),
                              seed=int(cfg.seed) + fold_idx, shuffle=False)
    test_loader = _make_loader(test_cache, None if use_external_test else test_idx, augment=None,
                               batch_size=int(cfg.data.loader.batch_size),
                               num_workers=int(cfg.data.loader.num_workers),
                               pin_memory=bool(cfg.data.loader.pin_memory),
                               seed=int(cfg.seed) + fold_idx, shuffle=False)

    encoder = instantiate_drop(cfg.backbone)
    strategy = instantiate_drop(
        cfg.finetune,
        encoder=encoder,
        optim_cfg=OmegaConf.to_container(cfg.optim, resolve=True),
        spacing=tuple(cfg.data.preprocess.target_spacing),
    )

    fold_ckpt_dir = str(Path(cfg.paths.ckpt_dir) / f"fold{fold_idx}")
    Path(fold_ckpt_dir).mkdir(parents=True, exist_ok=True)

    run_name_fold = f"{cfg.run_name}_fold{fold_idx}"
    logger = init_wandb(cfg, run_name=run_name_fold, extra_config={"fold": fold_idx})
    callbacks = build_callbacks(cfg.trainer, ckpt_dir=fold_ckpt_dir, has_logger=logger is not None)

    trainer = build_trainer(cfg.trainer, callbacks=callbacks, logger=logger,
                            default_root_dir=str(Path(cfg.paths.output_dir) / f"fold{fold_idx}"))
    trainer.fit(strategy, train_loader, val_loader)

    # Test on the held-out set using the BEST checkpoint from this fold.
    best_ckpt = None
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_path:
            best_ckpt = cb.best_model_path
            break
    # weights_only=False: our own checkpoints contain Lightning-saved hparams that
    # may include OmegaConf containers. PyTorch 2.6+ defaults `weights_only=True`
    # which refuses to unpickle these. We trust our own ckpts.
    test_metrics = trainer.test(strategy, dataloaders=test_loader,
                                ckpt_path=best_ckpt or "last",
                                verbose=False, weights_only=False)
    test_metrics_flat = test_metrics[0] if test_metrics else {}

    # Collect val (final) + test metrics for fold summary.
    summary = {
        "train_size": int(len(train_idx)),
        "train_size_full": int(len(full_train_idx)),
        "train_fraction": train_fraction,
    }
    for k, v in trainer.callback_metrics.items():
        if torch.is_tensor(v):
            summary[k] = float(v)
        elif isinstance(v, (int, float)):
            summary[k] = float(v)
    summary.update({k: float(v) for k, v in test_metrics_flat.items()})

    finish_wandb()
    return summary


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="finetune")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(int(cfg.seed))
    pl.seed_everything(int(cfg.seed), workers=True)

    cache_dir = Path(cfg.data.cache_dir)
    files = sorted(cache_dir.glob("*.npz"))
    n_items = len(files)
    if n_items == 0:
        raise FileNotFoundError(
            f"No cached labeled .npz at {cache_dir}. Run scripts/prepare_data.py first."
        )

    # Save the fully-resolved config alongside cv_summary.json. This makes the
    # aggregator (and scripts/evaluate_all.py) work without needing Hydra's
    # `hydra:` resolver to be registered.
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.paths.output_dir) / "resolved_config.yaml", resolve=True)

    # ----- External patient-disjoint test set (if prepared) -----
    # If cache/labeled_test/ exists and is non-empty, we use it as the FIXED test
    # set: no test_fraction holdout, K-fold over the FULL labeled cache.
    # This is the right thing to do when patients have multiple scans and you
    # want zero leakage at the test boundary.
    test_cache_dir = Path(str(OmegaConf.select(cfg, "data.test_cache_dir", default="")))
    external_test_files = sorted(test_cache_dir.glob("*.npz")) if test_cache_dir.exists() else []
    if external_test_files:
        # KFold over ALL labeled files; external set is the test.
        folds_local = kfold_indices(n_items, n_folds=int(cfg.split.n_folds), seed=int(cfg.split.seed))
        folds = [(np.asarray(tr), np.asarray(va)) for tr, va in folds_local]
        test_idx = None   # signal: external test set
        print(f"[split] EXTERNAL test set found at {test_cache_dir} "
              f"({len(external_test_files)} samples). "
              f"K-folding all {n_items} train+val files. test_fraction is IGNORED.")
    else:
        # Backward-compatible random test holdout.
        test_idx, folds = holdout_then_kfold(
            n_items=n_items,
            test_fraction=float(cfg.split.test_fraction),
            n_folds=int(cfg.split.n_folds),
            seed=int(cfg.split.seed),
        )
        print(f"[split] no external test set at {test_cache_dir} — falling back to "
              f"random holdout. test={list(test_idx)} ({len(test_idx)} samples)  "
              f"pool_size={n_items - len(test_idx)}  n_folds={len(folds)}")
        print("[split] WARNING: if your patients have multiple scans, this random "
              "holdout will leak. Prefer the external test set (data/labeled_test/).")

    if int(cfg.split.fold) >= 0:
        chosen = [(int(cfg.split.fold), folds[int(cfg.split.fold)])]
    else:
        chosen = list(enumerate(folds))

    all_metrics = []
    for i, (tr, va) in chosen:
        m = _run_one_fold(cfg, fold_idx=i, train_idx=tr, val_idx=va, test_idx=test_idx)
        m["_fold"] = i
        all_metrics.append(m)

    summary = {
        "split": {
            "n_items": int(n_items),
            "test_fraction": float(cfg.split.test_fraction),
            "n_folds": int(cfg.split.n_folds),
            "train_fraction": float(cfg.split.train_fraction),
            "test_idx": ([int(i) for i in test_idx]
                         if test_idx is not None else "EXTERNAL"),
            "test_source": (str(cfg.data.test_cache_dir)
                            if test_idx is None else str(cfg.data.cache_dir)),
            "files_total": [p.name for p in files],
        },
        "per_fold": all_metrics,
    }
    if all_metrics:
        keys = sorted({k for m in all_metrics for k in m if k != "_fold"})
        agg = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if k in m]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        summary["aggregate"] = agg

    summary_path = Path(cfg.paths.output_dir) / "cv_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote CV summary (val + test aggregates) to {summary_path}")


if __name__ == "__main__":
    main()
