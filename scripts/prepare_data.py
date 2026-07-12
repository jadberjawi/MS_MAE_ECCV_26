"""One-time preprocessing: DICOM (+ NIfTI mask) -> normalized .npz cache.

Every raw volume is resampled to 4 mm isotropic, LV-centred cropped (label-free),
and intensity-normalized to [0, 1], then written as a single .npz. The SAME
preprocessing is used for the unlabeled (pretraining) and labeled (finetuning)
cohorts so the encoder sees consistent geometry/scale across both stages.

Usage:
    python scripts/prepare_data.py                  # unlabeled + labeled + (labeled_test if present)
    python scripts/prepare_data.py --only unlabeled
    python scripts/prepare_data.py --only labeled
    python scripts/prepare_data.py --only labeled_test
    python scripts/prepare_data.py --force          # overwrite existing cache

Reads paths / preprocessing parameters from the SAME data configs training uses:
    configs/data/ssl.yaml   (unlabeled)
    configs/data/seg.yaml   (labeled train+val pool AND optional labeled_test held-out set)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.data.preprocessing import (
    PreprocessConfig, preproc_from_cfg,
    preprocess_image, preprocess_image_mask,
    save_npz_unlabeled, save_npz_labeled,
)


def _load_data_cfg(name: str):
    """Load one data config standalone and inject the `paths` block it interpolates."""
    cfg = OmegaConf.create({
        "paths": {
            "project_root": str(ROOT),
            "cache_dir": str(ROOT / "cache"),
        },
        "seed": 0,
    })
    data_cfg = OmegaConf.load(ROOT / "configs" / "data" / f"{name}.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.data = data_cfg
    return cfg.data


def _make_preproc(cfg) -> PreprocessConfig:
    return preproc_from_cfg(cfg.preprocess)


def prepare_unlabeled(force: bool) -> int:
    cfg = _load_data_cfg("ssl")
    raw_dir = Path(str(cfg.raw_dir))
    out_dir = Path(str(cfg.cache_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    preproc = _make_preproc(cfg)

    n = 0
    for dcm in sorted(raw_dir.glob("*.dcm")):
        out_path = out_dir / f"{dcm.stem}.npz"
        if out_path.exists() and not force:
            continue
        img, spacing = preprocess_image(dcm, preproc)
        save_npz_unlabeled(out_path, img, spacing)
        print(f"  [unlabeled] {dcm.name} -> {out_path.name}  shape={img.shape}")
        n += 1
    return n


def _prepare_labeled_from_dirs(img_dir: Path, msk_dir: Path, out_dir: Path,
                               preproc: PreprocessConfig, tag: str, force: bool) -> int:
    """Shared helper used by both `labeled` and `labeled_test`."""
    if not img_dir.exists():
        print(f"  [{tag}] no source dir at {img_dir} — skipping.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for dcm in sorted(img_dir.glob("*.dcm")):
        # Accept either extension and with/without the `_mask` suffix, so masks
        # saved as .nii or without the suffix still pair correctly.
        candidates = [
            msk_dir / f"{dcm.stem}_mask.nii.gz",
            msk_dir / f"{dcm.stem}_mask.nii",
            msk_dir / f"{dcm.stem}.nii.gz",
            msk_dir / f"{dcm.stem}.nii",
        ]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            print(f"  [{tag}] SKIP {dcm.name}: no matching mask "
                  f"(looked for {dcm.stem}_mask.nii[.gz] / {dcm.stem}.nii[.gz])")
            continue
        out_path = out_dir / f"{dcm.stem}.npz"
        if out_path.exists() and not force:
            continue
        img, msk, spacing = preprocess_image_mask(dcm, mask_path, preproc)
        save_npz_labeled(out_path, img, msk, spacing)
        print(f"  [{tag}] {dcm.name} -> {out_path.name}  shape={img.shape}  fg={int(msk.sum())} voxels")
        n += 1
    return n


def prepare_labeled(force: bool) -> int:
    cfg = _load_data_cfg("seg")
    return _prepare_labeled_from_dirs(
        img_dir=Path(str(cfg.raw_image_dir)),
        msk_dir=Path(str(cfg.raw_mask_dir)),
        out_dir=Path(str(cfg.cache_dir)),
        preproc=_make_preproc(cfg),
        tag="labeled",
        force=force,
    )


def prepare_labeled_test(force: bool) -> int:
    """External, patient-disjoint held-out test set. Skips silently if no source dir."""
    cfg = _load_data_cfg("seg")
    return _prepare_labeled_from_dirs(
        img_dir=Path(str(cfg.test_raw_image_dir)),
        msk_dir=Path(str(cfg.test_raw_mask_dir)),
        out_dir=Path(str(cfg.test_cache_dir)),
        preproc=_make_preproc(cfg),
        tag="labeled_test",
        force=force,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["unlabeled", "labeled", "labeled_test"], default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("== Preparing data ==")
    if args.only in (None, "unlabeled"):
        n = prepare_unlabeled(args.force)
        print(f"Unlabeled: {n} files written.")
    if args.only in (None, "labeled"):
        n = prepare_labeled(args.force)
        print(f"Labeled (train+val pool): {n} files written.")
    if args.only in (None, "labeled_test"):
        n = prepare_labeled_test(args.force)
        print(f"Labeled-test (held-out, patient-disjoint): {n} files written.")


if __name__ == "__main__":
    main()
