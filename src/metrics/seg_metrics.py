"""Segmentation metrics for binary LV. All metrics are computed on the foreground class.

Available:
  dice_score, iou_score, sensitivity, specificity, precision: voxel-set metrics
  hausdorff_95, average_surface_distance: surface distance metrics (require scipy)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_binary(pred_logits: torch.Tensor, target: torch.Tensor, fg_class: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = pred_logits.argmax(dim=1)              # (B,Z,Y,X)
    return (pred == fg_class), (target == fg_class)


def metrics_from_masks(pred_bin: np.ndarray, gt_bin: np.ndarray,
                       spacing=(1.0, 1.0, 1.0)) -> dict:
    """Compute all segmentation metrics from two binary (Z,Y,X) masks.

    Builds pseudo-logits from the prediction so the exact same code path as
    training/test (argmax-based) is used. Returns a plain dict of floats.
    """
    import torch.nn.functional as F
    t = torch.from_numpy(pred_bin.astype(np.int64)).clamp(0, 1)
    pl = F.one_hot(t, 2).permute(3, 0, 1, 2).unsqueeze(0).float() * 30.0   # (1,2,Z,Y,X)
    gt = torch.from_numpy(gt_bin.astype(np.int64)).unsqueeze(0)            # (1,Z,Y,X)
    out = {
        "dice": float(dice_score(pl, gt)),
        "iou": float(iou_score(pl, gt)),
        "sensitivity": float(sensitivity(pl, gt)),
        "precision": float(precision(pl, gt)),
        "pred_fg_voxels": int(pred_bin.sum()),
        "gt_fg_voxels": int(gt_bin.sum()),
    }
    try:
        out["hd95"] = float(hausdorff_95(pl, gt, spacing=spacing))
        out["asd"] = float(average_surface_distance(pl, gt, spacing=spacing))
    except Exception:
        out["hd95"] = float("nan"); out["asd"] = float("nan")
    return out


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p, t = _as_binary(pred_logits, target)
    p, t = p.float(), t.float()
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean()


def iou_score(pred_logits, target, eps=1e-6):
    p, t = _as_binary(pred_logits, target)
    p, t = p.float(), t.float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean()


def confusion_components(pred_logits, target):
    p, t = _as_binary(pred_logits, target)
    p, t = p.float(), t.float()
    tp = (p * t).sum(dim=(1, 2, 3))
    fp = (p * (1 - t)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * t).sum(dim=(1, 2, 3))
    tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
    return tp, fp, fn, tn


def sensitivity(pred_logits, target, eps=1e-6):
    tp, _, fn, _ = confusion_components(pred_logits, target)
    return ((tp + eps) / (tp + fn + eps)).mean()


def precision(pred_logits, target, eps=1e-6):
    tp, fp, _, _ = confusion_components(pred_logits, target)
    return ((tp + eps) / (tp + fp + eps)).mean()


def hausdorff_95(pred_logits: torch.Tensor, target: torch.Tensor, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """95-percentile Hausdorff distance (mm if `spacing` is mm). Computed on CPU per-sample.

    Returns 0 if both masks are empty; +inf if exactly one is empty.
    """
    from scipy.ndimage import distance_transform_edt

    p, t = _as_binary(pred_logits, target)
    p_np = p.cpu().numpy().astype(bool)
    t_np = t.cpu().numpy().astype(bool)
    vals = []
    for i in range(p_np.shape[0]):
        a, b = p_np[i], t_np[i]
        if not a.any() and not b.any():
            vals.append(0.0); continue
        if not a.any() or not b.any():
            vals.append(float("inf")); continue
        d_a = distance_transform_edt(~a, sampling=spacing)
        d_b = distance_transform_edt(~b, sampling=spacing)
        surf_a = a & ~_erode(a)
        surf_b = b & ~_erode(b)
        dists = np.concatenate([d_b[surf_a], d_a[surf_b]])
        vals.append(float(np.percentile(dists, 95)))
    return torch.tensor(np.mean(vals), dtype=torch.float32)


def _erode(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask)


def average_surface_distance(pred_logits, target, spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    from scipy.ndimage import distance_transform_edt
    p, t = _as_binary(pred_logits, target)
    p_np = p.cpu().numpy().astype(bool)
    t_np = t.cpu().numpy().astype(bool)
    vals = []
    for i in range(p_np.shape[0]):
        a, b = p_np[i], t_np[i]
        if not a.any() and not b.any():
            vals.append(0.0); continue
        if not a.any() or not b.any():
            vals.append(float("inf")); continue
        d_a = distance_transform_edt(~a, sampling=spacing)
        d_b = distance_transform_edt(~b, sampling=spacing)
        surf_a = a & ~_erode(a)
        surf_b = b & ~_erode(b)
        vals.append(float(np.mean(np.concatenate([d_b[surf_a], d_a[surf_b]]))))
    return torch.tensor(np.mean(vals), dtype=torch.float32)
