"""DICOM / NIfTI -> normalized tensor cache.

Run once via `scripts/prepare_data.py`. Each input volume becomes a single .npz
containing keys:
    image:   float32, normalized to [0, 1], shape (Z, Y, X)
    mask:    uint8, present only for labeled samples
    spacing: float32 [sz, sy, sx] of the resampled volume

We canonicalize axis order to (Z, Y, X) for both modalities so downstream code
never has to think about it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import pydicom
import scipy.ndimage as ndi

from . import smart_crop as sc


@dataclass
class PreprocessConfig:
    target_spacing: Tuple[float, float, float]   # (sz, sy, sx) in mm
    target_shape: Tuple[int, int, int]           # (Z, Y, X)
    clip_percentiles: Tuple[float, float]        # (low, high) in percent
    # Smart LV-centred cropping (see src/data/smart_crop.py). When False, the
    # legacy symmetric center_pad_or_crop is used (original behaviour).
    smart_crop: bool = False
    crop_size: Tuple[int, int, int] = (40, 40, 40)
    lv_radius: float = 9.0
    use_ring: bool = True


def preproc_from_cfg(p) -> "PreprocessConfig":
    """Build a PreprocessConfig from a config node (OmegaConf DictConfig or dict),
    defaulting the smart-crop fields so configs without them keep legacy behaviour."""
    return PreprocessConfig(
        target_spacing=tuple(p["target_spacing"]),
        target_shape=tuple(p["target_shape"]),
        clip_percentiles=tuple(p["clip_percentiles"]),
        smart_crop=bool(p.get("smart_crop", False)),
        crop_size=tuple(p.get("crop_size", (40, 40, 40))),
        lv_radius=float(p.get("lv_radius", 9.0)),
        use_ring=bool(p.get("use_ring", True)),
    )


# ---------- IO ----------

def load_dicom_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Return (image[Z,Y,X], spacing[sz, sy, sx])."""
    d = pydicom.dcmread(str(path))
    arr = d.pixel_array.astype(np.float32)   # (frames=Z, rows=Y, cols=X) for NM
    if arr.ndim == 2:
        arr = arr[None]  # promote 2D to (1, Y, X)

    px = getattr(d, "PixelSpacing", [1.0, 1.0])
    sy, sx = float(px[0]), float(px[1])
    sz = float(getattr(d, "SliceThickness", getattr(d, "SpacingBetweenSlices", 1.0)))
    return arr, (sz, sy, sx)


def load_nifti_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Return (mask[Z,Y,X], spacing[sz, sy, sx]).

    Reorient the mask to the canonical ('L','P','I') voxel frame that our DICOM
    read (pydicom pixel_array) is aligned to, USING THE NIfTI AFFINE. Masks
    exported with a different axis direction — e.g. center-A's ('L','P','S'),
    whose Z axis is reversed — are corrected here, so image and mask always line
    up regardless of the labeling tool's export convention. This is a no-op for
    masks already in ('L','P','I') (e.g. center-B), so existing aligned cohorts
    are unchanged.
    """
    from nibabel.orientations import (apply_orientation, axcodes2ornt,
                                       io_orientation, ornt_transform)
    nii = nib.load(str(path))
    data = nii.get_fdata().astype(np.float32)
    cur = io_orientation(nii.affine)
    target = axcodes2ornt(("L", "P", "I"))
    data = apply_orientation(data, ornt_transform(cur, target))   # -> (L, P, I) voxel order
    arr = np.transpose(data, (2, 1, 0)).copy()                    # -> (Z, Y, X)
    spacing_xyz = np.abs(np.diag(nii.affine)[:3])    # isotropic 4 mm -> order irrelevant
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    return arr, (sz, sy, sx)


# ---------- Geometry ----------

def resample_to_spacing(
    vol: np.ndarray,
    src_spacing: Tuple[float, float, float],
    dst_spacing: Tuple[float, float, float],
    order: int,
) -> np.ndarray:
    zoom = [s / d for s, d in zip(src_spacing, dst_spacing)]
    return ndi.zoom(vol, zoom=zoom, order=order, mode="nearest").astype(vol.dtype)


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    """Symmetric pad-or-crop to `target_shape`."""
    out = np.full(target_shape, pad_value, dtype=vol.dtype)
    in_shape = vol.shape

    # Compute crop window on input and paste window on output.
    src_starts, src_ends, dst_starts, dst_ends = [], [], [], []
    for i, (s_in, s_out) in enumerate(zip(in_shape, target_shape)):
        if s_in >= s_out:
            start = (s_in - s_out) // 2
            src_starts.append(start)
            src_ends.append(start + s_out)
            dst_starts.append(0)
            dst_ends.append(s_out)
        else:
            start = (s_out - s_in) // 2
            src_starts.append(0)
            src_ends.append(s_in)
            dst_starts.append(start)
            dst_ends.append(start + s_in)

    out[
        dst_starts[0]:dst_ends[0],
        dst_starts[1]:dst_ends[1],
        dst_starts[2]:dst_ends[2],
    ] = vol[
        src_starts[0]:src_ends[0],
        src_starts[1]:src_ends[1],
        src_starts[2]:src_ends[2],
    ]
    return out


# ---------- Intensity ----------

def normalize_intensity(vol: np.ndarray, clip_percentiles: Tuple[float, float]) -> np.ndarray:
    lo_q, hi_q = np.percentile(vol, clip_percentiles)
    lo_q = float(lo_q)
    hi_q = float(max(hi_q, lo_q + 1e-6))
    vol = np.clip(vol, lo_q, hi_q)
    vol = (vol - lo_q) / (hi_q - lo_q)
    return vol.astype(np.float32)


# ---------- High-level entry points ----------

def preprocess_image(
    dicom_path: Path,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img, src_sp = load_dicom_volume(dicom_path)
    img = resample_to_spacing(img, src_sp, cfg.target_spacing, order=1)
    if cfg.smart_crop:
        img = sc.smart_crop_image(img, cfg.crop_size, cfg.target_shape,
                                  lv_radius=cfg.lv_radius, use_ring=cfg.use_ring)
    else:
        img = center_pad_or_crop(img, cfg.target_shape, pad_value=float(img.min()))
    img = normalize_intensity(img, cfg.clip_percentiles)
    return img, cfg.target_spacing


def preprocess_image_mask(
    dicom_path: Path,
    mask_path: Path,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    img, src_sp_img = load_dicom_volume(dicom_path)
    msk, src_sp_msk = load_nifti_mask(mask_path)

    # If shapes already match (no resample needed), avoid drift.
    if img.shape != msk.shape:
        # Bring mask into image's grid by resampling mask using the IMAGE source spacing
        # (the labeler operated on the image grid; mask is delivered in the image frame).
        msk = resample_to_spacing(msk, src_sp_msk, src_sp_img, order=0)
        if msk.shape != img.shape:
            msk = center_pad_or_crop(msk, img.shape, pad_value=0)

    img = resample_to_spacing(img, src_sp_img, cfg.target_spacing, order=1)
    msk = resample_to_spacing(msk, src_sp_img, cfg.target_spacing, order=0)

    if cfg.smart_crop:
        # Same label-free localizer as pretraining; crop image+mask with one box;
        # verify the LV mask is fully contained (warn loudly if any voxel clipped).
        img, msk, clipped, total_lv = sc.smart_crop_image_mask(
            img, msk, cfg.crop_size, cfg.target_shape,
            lv_radius=cfg.lv_radius, use_ring=cfg.use_ring)
        if clipped > 0:
            print(f"[smart_crop] WARNING {dicom_path.stem}: {clipped}/{total_lv} LV voxels "
                  f"clipped by the {tuple(cfg.crop_size)} box — review this case.")
    else:
        img = center_pad_or_crop(img, cfg.target_shape, pad_value=float(img.min()))
        msk = center_pad_or_crop(msk, cfg.target_shape, pad_value=0)

    img = normalize_intensity(img, cfg.clip_percentiles)
    msk = (msk > 0.5).astype(np.uint8)
    return img, msk, cfg.target_spacing


# ---------- Cache writers ----------

def save_npz_unlabeled(out_path: Path, image: np.ndarray, spacing: Tuple[float, float, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image=image.astype(np.float32),
                        spacing=np.array(spacing, dtype=np.float32))


def save_npz_labeled(out_path: Path, image: np.ndarray, mask: np.ndarray,
                     spacing: Tuple[float, float, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image=image.astype(np.float32),
                        mask=mask.astype(np.uint8),
                        spacing=np.array(spacing, dtype=np.float32))


def load_cached_unlabeled(path: Path) -> np.ndarray:
    z = np.load(path)
    return z["image"]


def load_cached_labeled(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["image"], z["mask"]


def load_gt_on_grid(gt_dir, stem: str, target_shape, target_spacing):
    """Return a binary GT mask (Z,Y,X) on the given grid, or None if not found.

    Supports two GT sources, auto-detected:
      * `<stem>.npz` with a `mask` key (already on the model grid, e.g. a cache dir)
      * `<stem>.nii.gz` or `<stem>_mask.nii.gz` raw mask (resampled + cropped here)
    """
    gt_dir = Path(gt_dir)
    npz = gt_dir / f"{stem}.npz"
    if npz.exists():
        z = np.load(npz)
        if "mask" in z.files:
            m = (z["mask"] > 0).astype(np.uint8)
            if tuple(m.shape) != tuple(target_shape):
                m = center_pad_or_crop(m, tuple(target_shape), pad_value=0).astype(np.uint8)
            return m
    # Accept either extension, with or without the `_mask` suffix.
    for cand in (gt_dir / f"{stem}_mask.nii.gz", gt_dir / f"{stem}_mask.nii",
                 gt_dir / f"{stem}.nii.gz", gt_dir / f"{stem}.nii"):
        if cand.exists():
            msk, src_sp = load_nifti_mask(cand)
            msk = resample_to_spacing(msk, src_sp, tuple(target_spacing), order=0)
            msk = center_pad_or_crop(msk, tuple(target_shape), pad_value=0)
            return (msk > 0.5).astype(np.uint8)
    return None
