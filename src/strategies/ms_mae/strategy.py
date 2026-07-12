"""Multi-Scale Masked Autoencoder (MS-MAE) pretraining strategy.

This is the paper's contribution. Two reconstruction objectives share ONE encoder:

  * FINE   : standard MAE — reconstruct the masked patches at full resolution.
             Drives local detail / texture completion.
  * COARSE : reconstruct a downsampled (e.g. 16^3) version of the CLEAN volume
             from the bottleneck. Drives global-structure encoding.

Both heads consume the SAME single encoder pass on the MASKED input, so the
coarse stream is still a genuine prediction task (the input was corrupted), not
a trivial low-res pass-through autoencoder.

    loss = lambda_fine * L_fine(masked) + lambda_coarse * L_coarse(global)

Logged per epoch: loss, loss_fine, loss_coarse, masked_psnr (fine stream).
"""
from __future__ import annotations

from typing import Any, Dict

import hydra
import torch
import torch.nn.functional as F

from ..base import BasePretrainStrategy
from .masking import random_cube_mask, apply_mask


class MSMAEStrategy(BasePretrainStrategy):
    def __init__(
        self,
        encoder,
        optim_cfg: Dict[str, Any],
        mask_ratio: float,
        patch_size,
        mask_value: float,
        coarse_shape,
        fine_decoder: Dict[str, Any],
        coarse_decoder: Dict[str, Any],
        loss: Dict[str, Any],
    ):
        super().__init__(encoder=encoder, optim_cfg=optim_cfg)
        self.mask_ratio = float(mask_ratio)
        self.patch_size = tuple(patch_size)
        self.mask_value = float(mask_value)
        self.coarse_shape = tuple(coarse_shape)
        self.loss_cfg = loss
        self.lambda_fine = float(loss.get("lambda_fine", 1.0))
        self.lambda_coarse = float(loss.get("lambda_coarse", 0.5))

        # Both heads are the same weak-decoder class; the coarse one is simply
        # asked for a small target shape at forward time.
        self.fine_decoder = hydra.utils.instantiate(fine_decoder, encoder=encoder)
        self.coarse_decoder = hydra.utils.instantiate(coarse_decoder, encoder=encoder)

    def _fine_loss(self, recon, target, mask):
        m = mask.unsqueeze(1)
        eps = 1e-6
        kind = self.loss_cfg.get("type", "l1_masked")
        if kind == "l2_masked":
            per_voxel = (recon - target) ** 2
        else:  # l1_masked (default)
            per_voxel = (recon - target).abs()
        return (per_voxel * m).sum() / (m.sum() + eps)

    def pretext_step(self, batch: dict, stage: str) -> Dict[str, torch.Tensor]:
        x = batch["image"]                                   # (B,1,Z,Y,X) CLEAN target
        x_in = batch.get("image_aug", x)                     # augmented input (== x if no aug)
        B, _, Z, Y, X = x.shape
        device = x.device

        masks = torch.stack([
            random_cube_mask((Z, Y, X), self.patch_size, self.mask_ratio, device=device)
            for _ in range(B)
        ], dim=0)
        # Mask the AUGMENTED input; both fine and coarse reconstruct the CLEAN x.
        x_masked = apply_mask(x_in, masks, fill=self.mask_value)

        feats = self.encoder(x_masked)                       # single shared encode

        # Fine: full-resolution masked reconstruction.
        recon_fine = self.fine_decoder(feats, target_shape=(Z, Y, X))
        loss_fine = self._fine_loss(recon_fine, x, masks)

        # Coarse: global low-res reconstruction of the CLEAN volume.
        # trilinear downsample (portable across CUDA/MPS; adaptive_avg_pool3d is
        # not implemented on MPS). For a smooth global target this is equivalent.
        coarse_target = F.interpolate(x, size=self.coarse_shape, mode="trilinear",
                                      align_corners=False)                 # (B,1,*coarse)
        recon_coarse = self.coarse_decoder(feats, target_shape=self.coarse_shape)
        loss_coarse = (recon_coarse - coarse_target).abs().mean()

        loss = self.lambda_fine * loss_fine + self.lambda_coarse * loss_coarse

        with torch.no_grad():
            m = masks.unsqueeze(1)
            mse = ((recon_fine - x) ** 2 * m).sum() / (m.sum() + 1e-6)
            psnr = -10.0 * torch.log10(mse + 1e-10)

        return {
            "loss": loss,
            "loss_fine": loss_fine.detach(),
            "loss_coarse": loss_coarse.detach(),
            "masked_psnr": psnr.detach(),
        }
