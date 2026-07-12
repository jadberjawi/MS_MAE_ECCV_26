"""Split helpers.

For supervised finetuning we use a STRICT hold-out test set:

    all labeled samples
        │
        ├── test set       (held-out, never seen during training or model selection)
        └── train+val pool
                │
                └── KFold over this pool → each fold's val comes from INSIDE
                                            the train+val pool, never overlaps test.

After fitting each fold, the best checkpoint is evaluated on the SAME held-out
test set. Test metrics are averaged across folds for the final report.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold


def kfold_indices(n_items: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Plain K-Fold over n_items. Returns [(train_idx, val_idx), ...].

    Gracefully reduces n_folds to n_items if you have fewer samples than folds.
    """
    if n_items < 2:
        raise ValueError(f"Need at least 2 samples for any CV, got {n_items}.")
    effective_folds = min(max(2, n_folds), n_items)
    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(np.arange(n_items))]


def train_val_holdout(n_items: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Single random split (no K-fold), used by SSL pretraining and the linear probe."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = max(1, int(round(n_items * val_fraction)))
    return idx[n_val:], idx[:n_val]


def subsample_indices(
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Deterministically pick `round(len * fraction)` indices from `indices`.

    Used for label-efficiency experiments: shrink ONLY the train pool so we can
    measure how performance scales with %labels.

    Nested property: with the SAME seed, the subset for fraction=p1 is a SUBSET
    of the subset for fraction=p2 when p1 < p2. This makes a sweep like
    {10%, 20%, 50%, 100%} apples-to-apples — each smaller fraction is contained
    in every larger one, so any improvement is purely due to having more data,
    not to having different data.
    """
    arr = np.asarray(indices)
    n = len(arr)
    if fraction >= 1.0:
        return arr
    if fraction <= 0.0 or n == 0:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction} with n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(arr)
    k = max(1, int(round(n * float(fraction))))
    return np.sort(perm[:k])


def holdout_then_kfold(
    n_items: int,
    test_fraction: float,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Hold out a deterministic test set first, then K-fold the remainder.

    Returns:
        test_idx        : np.ndarray of indices into the original n_items
        folds           : list of (train_idx, val_idx) pairs, each indices into
                          the original n_items (already mapped, NOT relative to
                          the train+val pool).
    """
    if n_items < 3:
        raise ValueError(
            f"Need at least 3 samples (>=1 test + >=2 for KFold), got {n_items}."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)

    n_test = max(1, int(round(n_items * float(test_fraction))))
    # Guarantee at least 2 samples remain for K-fold.
    n_test = min(n_test, n_items - 2)

    test_idx = np.sort(perm[:n_test])
    pool = np.sort(perm[n_test:])

    inner_folds = kfold_indices(len(pool), n_folds=n_folds, seed=seed)
    # Map inner indices (into `pool`) back to original indices.
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_rel, va_rel in inner_folds:
        folds.append((pool[tr_rel], pool[va_rel]))
    return test_idx, folds
