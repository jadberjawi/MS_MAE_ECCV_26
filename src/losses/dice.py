"""Soft Dice loss for multi-class 3D segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, include_background: bool = False, smooth: float = 1.0):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, Z, Y, X); target: (B, Z, Y, X) long
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()    # (B,C,Z,Y,X)

        if not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = (probs * target_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 include_background: bool = False, smooth: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(include_background=include_background, smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
        self.dw = float(dice_weight)
        self.cw = float(ce_weight)

    def forward(self, logits, target):
        return self.dw * self.dice(logits, target) + self.cw * self.ce(logits, target)
