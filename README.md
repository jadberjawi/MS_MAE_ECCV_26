# MS-MAE: Multi-Scale Masked Autoencoder Pretraining for SPECT-MPI LV Segmentation

Reference implementation for our ECCV submission. **MS-MAE** is a self-supervised
pretraining strategy for 3D SPECT myocardial-perfusion volumes. It augments a
standard masked autoencoder with a second, coarse reconstruction stream so that a
single shared encoder is pushed to capture both **local detail** (fine, masked
patch reconstruction) and **global structure** (coarse, whole-volume
reconstruction). The pretrained encoder is then transferred to a supervised
left-ventricle (LV) segmentation task.

1. **Data preparation** — `scripts/prepare_data.py`
2. **MS-MAE pretraining** (our contribution) — `scripts/pretrain.py`
3. **Supervised finetuning** — `scripts/finetune.py`

---

## The method in one paragraph

Given a clean volume `x`, we corrupt it into `x_masked` by masking a fraction of
non-overlapping 3D patch-cubes. A single U-Net encoder encodes `x_masked` once.
Two lightweight ("intentionally weak") decoders then read the same bottleneck
features:

- **Fine stream** reconstructs the full-resolution volume; the loss is computed
  **only on masked voxels** (standard MAE objective).
- **Coarse stream** reconstructs a downsampled `16×16×16` version of the clean
  volume, forcing the bottleneck to encode global anatomy.

```
loss = λ_1 · L_fine(masked)  +  λ_2 · L_coarse(global)
```

Because both decoders consume the *same* encoding of the *masked* input, the
coarse stream remains a genuine prediction task rather than a trivial low-res
autoencoder. The core logic lives in
[`src/strategies/ms_mae/strategy.py`](src/strategies/ms_mae/strategy.py).

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch FIRST, matching your CUDA driver (see top of requirements.txt).
# CPU-only example:
pip install torch torchvision

# Then the rest:
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, PyTorch ≥ 2.4, PyTorch-Lightning 2.x.

---

## 2. Prepare the data

### Expected raw layout

Place your raw files here (DICOM images, NIfTI masks):

```
data/
├── unlabeled/DICOM/        # *.dcm — unlabeled volumes for pretraining
├── labeled/
│   ├── DICOM/              # *.dcm — labeled volumes for finetuning (train+val pool)
│   └── masks/              # <stem>_mask.nii.gz — LV masks, one per image
└── labeled_test/           # OPTIONAL patient-disjoint held-out test set
    ├── DICOM/
    └── masks/
```

Each image DICOM `foo.dcm` must have a matching mask named `foo_mask.nii.gz`
(the loader also accepts `foo.nii.gz`, `foo_mask.nii`, `foo.nii`).

### Build the cache

```bash
python scripts/prepare_data.py                  # unlabeled + labeled + labeled_test (if present)
# or one split at a time:
python scripts/prepare_data.py --only unlabeled
python scripts/prepare_data.py --only labeled
python scripts/prepare_data.py --only labeled_test
python scripts/prepare_data.py --force          # overwrite existing cache
```

This resamples every volume to **4 mm isotropic**, applies a **label-free
LV-centred crop**, normalizes intensities to `[0, 1]`, and writes one compressed
`.npz` per volume under `cache/`. The **same** preprocessing is used for
pretraining and finetuning, so the encoder sees consistent geometry and scale
across both stages. All preprocessing parameters live in
[`configs/data/ssl.yaml`](configs/data/ssl.yaml) and
[`configs/data/seg.yaml`](configs/data/seg.yaml).

```
cache/
├── unlabeled/      # *.npz  {image, spacing}
├── labeled/        # *.npz  {image, mask, spacing}
└── labeled_test/   # *.npz  (only if you prepared a held-out set)
```

---

## 3. Pretrain with MS-MAE

```bash
python scripts/pretrain.py
```

Common overrides (Hydra syntax — everything is configurable from the CLI):

```bash
# Longer schedule, higher mask ratio:
python scripts/pretrain.py trainer.max_epochs=300 pretrain.mask_ratio=0.75

# Change the coarse/fine loss balance:
python scripts/pretrain.py pretrain.loss.lambda_coarse=1.0

# Quick local smoke test, no logging:
python scripts/pretrain.py use_wandb=false trainer.max_epochs=2 trainer.accelerator=cpu
```

The best/last checkpoints are written to:

```
outputs/pretrain/ms_mae_unet3d_seed1337/checkpoints/last.ckpt
```

Key knobs — [`configs/pretrain/ms_mae.yaml`](configs/pretrain/ms_mae.yaml):

| Parameter | Default | Meaning |
|---|---|---|
| `mask_ratio` | `0.6` | fraction of patch-cubes masked |
| `patch_size` | `[4,4,4]` | masking granularity (voxels) |
| `coarse_shape` | `[16,16,16]` | resolution of the coarse reconstruction target |
| `loss.lambda_fine` | `1.0` | weight of the fine (masked) stream |
| `loss.lambda_coarse` | `0.5` | weight of the coarse (global) stream |

---

## 4. Finetune for LV segmentation

**From the MS-MAE pretrained encoder** (the main result):

```bash
python scripts/finetune.py \
    finetune=from_pretrained \
    ssl_tag=ms_mae \
    finetune.pretrained_ckpt=outputs/pretrain/ms_mae_unet3d_seed1337/checkpoints/last.ckpt
```

**From-scratch baseline** (random init, for comparison):

```bash
python scripts/finetune.py finetune=from_scratch ssl_tag=scratch
```

### Split policy

- If `cache/labeled_test/` exists, it is used as a **fixed, patient-disjoint test
  set** and K-fold cross-validation runs over the full labeled pool.
- Otherwise a random `test_fraction` (default `0.2`) is held out first, then the
  remainder is K-folded. Validation and test sets never overlap.

Useful overrides:

```bash
# Single fold for debugging:
python scripts/finetune.py finetune=from_pretrained finetune.pretrained_ckpt=<ckpt> split.fold=0

# Label-efficiency sweep (shrinks only the train set; val/test stay full):
python scripts/finetune.py -m split.train_fraction=0.1,0.2,0.5,1.0
```

Results are written to:

```
outputs/finetune/<run_name>/cv_summary.json     # per-fold + aggregate val/test metrics
```

Reported metrics: Dice, IoU, sensitivity, precision, HD95, ASD (per fold and
averaged over folds on the held-out test set).

---

## Repository layout

```
MS_MAE_ECCV/
├── scripts/
│   ├── prepare_data.py          # DICOM/NIfTI -> normalized .npz cache
│   ├── pretrain.py              # MS-MAE self-supervised pretraining
│   └── finetune.py              # supervised LV segmentation (K-fold + held-out test)
├── configs/                     # Hydra configs (data / backbone / strategy / trainer / optim)
│   ├── pretrain.yaml            # pretraining entry config
│   ├── finetune.yaml            # finetuning entry config
│   ├── pretrain/ms_mae.yaml     # MS-MAE hyperparameters
│   └── ...
└── src/
    ├── data/                    # preprocessing, smart-crop, datasets, augmentation, splits
    ├── models/
    │   ├── backbones/unet3d.py  # shared 3D U-Net encoder + segmentation decoder
    │   └── heads/mae_decoder.py # weak reconstruction decoder (fine + coarse heads)
    ├── strategies/
    │   ├── ms_mae/strategy.py   # >>> the MS-MAE method <<<
    │   └── supervised/strategy.py
    ├── engine/                  # Lightning trainer/callbacks + encoder EMA
    ├── losses/  metrics/  utils/
    └── ...
```

The **same** `UNet3DEncoder` class is instantiated for pretraining and
finetuning, so weight transfer between the two stages is exact and key-compatible.

---

## Reproducibility notes

- All randomness is seeded via the top-level `seed` (default `1337`).
- The split seed (`split.seed`) is kept fixed across multi-seed runs so the test
  set never changes — this makes paired statistical tests across methods valid.
- Weights & Biases logging is optional: set `use_wandb=false` to run fully
  offline. When enabled, set `WANDB_API_KEY` in your environment first.

