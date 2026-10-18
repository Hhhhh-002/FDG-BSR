from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, shave: int = 4) -> float:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    if shave > 0 and pred.shape[-1] > shave * 2 and pred.shape[-2] > shave * 2:
        pred = pred[..., shave:-shave, shave:-shave]
        target = target[..., shave:-shave, shave:-shave]
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)

