"""Label-free LV-centred cropping, ported verbatim from the standalone
crop_smart.py / crop_smart_labeled.py so pretraining and finetuning crop the
heart identically.

Positioning is ALWAYS label-free (intensity center + ring guard) — the mask is
never used to place the box, so train/test use the same localizer and no GT
location leaks into the input. For labeled data the mask is cropped with the SAME
box and we VERIFY that no LV voxel was clipped (caller warns on misses).

`pad_to` frames the fixed-size box in the target grid WITHOUT resampling, so the
heart keeps its true voxel size — identical scale across pretraining, finetuning,
and every center/tracer cell.

Defaults match the standalone scripts (crop_size 40^3, LV radius 9 voxels), which
are valid directly because the project's grid is 4 mm isotropic — the same grid
those scripts were calibrated on.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

DEFAULT_CROP_SIZE = (40, 40, 40)
DEFAULT_LV_RADIUS = 9.0


# ---------------- center finding (verbatim logic from crop_smart.py) ----------------

def _gaussian_center_prior(shape):
    D, H, W = shape
    z, y, x = np.ogrid[:D, :H, :W]
    cz, cy, cx = D / 2, H / 2, W / 2
    sigma = min(D, H, W) / 3.0
    dist_sq = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    return np.exp(-dist_sq / (2 * sigma ** 2))


def find_center_intensity(volume):
    D, H, W = volume.shape
    gaussian_mask = _gaussian_center_prior(volume.shape)
    nonzero = volume[volume > 0]
    p99 = np.percentile(nonzero, 99.0) if nonzero.size > 0 else np.percentile(volume, 99.0)
    vol_clipped = np.clip(volume, 0, p99)
    weighted_vol = vol_clipped * gaussian_mask
    threshold = np.percentile(weighted_vol, 99.0)
    binary_mask = weighted_vol > threshold
    try:
        coords = ndimage.center_of_mass(binary_mask)
        if coords is None or not np.all(np.isfinite(coords)):
            raise ValueError
        return np.array(coords).astype(float), True
    except Exception:
        return np.array([D // 2, H // 2, W // 2], dtype=float), False


def _annulus_kernel(R, shell=1.5):
    rad = int(np.ceil(R + shell + 1))
    zz, yy, xx = np.indices((2 * rad + 1,) * 3) - rad
    r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    k = np.zeros_like(r, dtype=np.float32)
    k[np.abs(r - R) <= shell] = 1.0
    k[r < (R - shell)] = -1.0
    pos, neg = (k > 0).sum(), (k < 0).sum()
    if pos == 0 or neg == 0:
        k[r <= R] = 1.0
        return k - k.mean()
    k[k > 0] /= pos
    k[k < 0] /= neg
    return k


def find_center_ring(volume, lv_radius, p99=None):
    if p99 is None:
        nz = volume[volume > 0]
        p99 = np.percentile(nz, 99.0) if nz.size else np.percentile(volume, 99.0)
    v = np.clip(volume, 0, p99).astype(np.float32)
    v = ndimage.gaussian_filter(v, sigma=1.0)
    k = _annulus_kernel(lv_radius)
    resp = ndimage.convolve(v, k, mode="constant", cval=0.0)
    resp *= _gaussian_center_prior(volume.shape)
    resp = np.maximum(resp, 0)
    idx = np.unravel_index(int(np.argmax(resp)), resp.shape)
    center = np.array(idx, dtype=float)
    grid = np.indices(volume.shape).astype(np.float32)
    d2 = sum((grid[a] - center[a]) ** 2 for a in range(3))
    w = v * (d2 <= lv_radius ** 2)
    if w.sum() > 0:
        center = np.array([(w * grid[a]).sum() for a in range(3)]) / w.sum()
    return center, float(resp[idx]), float(p99)


def choose_center(volume, lv_radius=DEFAULT_LV_RADIUS, use_ring=True,
                  disagree_vox=4.0, ring_conf=0.10, verbose=False):
    c_int, ok = find_center_intensity(volume)
    if not use_ring:
        return np.round(c_int).astype(int), "intensity"
    c_ring, peak, scale = find_center_ring(volume, lv_radius)
    disagree = float(np.linalg.norm(c_ring - c_int))
    confident = peak > ring_conf * scale
    if confident and disagree > disagree_vox:
        center, method = c_ring, "ring_guard"
    else:
        center, method = c_int, "intensity"
    if verbose:
        print(f"   intensity={np.round(c_int,1)} ring={np.round(c_ring,1)} "
              f"disagree={disagree:.1f} ringPeak/scale={peak/scale:.3f} -> {method}")
    return np.round(center).astype(int), method


# ---------------- crop + pad (verbatim) ----------------

def crop_at(volume, center, crop_size):
    D, H, W = volume.shape
    cz, cy, cx = crop_size
    z_min = max(0, int(center[0]) - cz // 2)
    y_min = max(0, int(center[1]) - cy // 2)
    x_min = max(0, int(center[2]) - cx // 2)
    z_max = min(D, z_min + cz)
    y_max = min(H, y_min + cy)
    x_max = min(W, x_min + cx)
    cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    pad_z, pad_y, pad_x = cz - cropped.shape[0], cy - cropped.shape[1], cx - cropped.shape[2]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        cropped = np.pad(cropped, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    return cropped, (z_min, y_min, x_min), (z_max, y_max, x_max)


def pad_to(vol, target, pad_value=0):
    target = np.array(target, dtype=int)
    cur = np.array(vol.shape, dtype=int)
    out = np.full(tuple(target), pad_value, dtype=vol.dtype)
    keep = np.minimum(cur, target)
    src0 = np.maximum((cur - target) // 2, 0)
    dst0 = np.maximum((target - cur) // 2, 0)
    src = tuple(slice(int(s), int(s + k)) for s, k in zip(src0, keep))
    dst = tuple(slice(int(d), int(d + k)) for d, k in zip(dst0, keep))
    out[dst] = vol[src]
    return out


# ---------------- high-level helpers used by preprocessing.py ----------------

def smart_crop_image(vol: np.ndarray, crop_size, target_shape,
                     lv_radius=DEFAULT_LV_RADIUS, use_ring=True) -> np.ndarray:
    """Label-free LV-centred crop, padded (not resampled) to target_shape."""
    center, _ = choose_center(vol, lv_radius=lv_radius, use_ring=use_ring)
    cropped, _, _ = crop_at(vol, center, tuple(crop_size))
    return pad_to(cropped, tuple(target_shape), pad_value=float(vol.min()))


def smart_crop_image_mask(img: np.ndarray, msk: np.ndarray, crop_size, target_shape,
                          lv_radius=DEFAULT_LV_RADIUS, use_ring=True,
                          label_value: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Same label-free localizer on the IMAGE; crop image AND mask with the SAME
    box; verify no LV voxel was clipped. Returns (img_out, msk_out, clipped, total_lv)."""
    center, _ = choose_center(img, lv_radius=lv_radius, use_ring=use_ring)
    cropped_img, mins, maxs = crop_at(img, center, tuple(crop_size))
    cropped_msk, _, _ = crop_at(msk, center, tuple(crop_size))

    fg = (msk == label_value) if label_value is not None else (msk > 0)
    total_lv = int(fg.sum())
    inside = int(fg[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].sum())
    clipped = total_lv - inside

    img_out = pad_to(cropped_img, tuple(target_shape), pad_value=float(img.min()))
    msk_out = pad_to(cropped_msk, tuple(target_shape), pad_value=0)
    return img_out, msk_out, clipped, total_lv
