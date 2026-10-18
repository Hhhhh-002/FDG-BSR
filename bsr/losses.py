from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.net = None
        if enabled:
            try:
                from torchvision.models import vgg19, VGG19_Weights

                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval()
                for p in vgg.parameters():
                    p.requires_grad_(False)
                self.net = vgg
            except Exception:
                self.enabled = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.net is None:
            return pred.new_tensor(0.0)
        self.net = self.net.to(pred.device)
        return F.l1_loss(self.net(pred.clamp(0, 1)), self.net(target.clamp(0, 1)))


def generator_adv_loss(dt_fake: torch.Tensor, ds_fake: torch.Tensor, lambda_t: float = 1.0, lambda_s: float = 1.0) -> torch.Tensor:
    target = torch.ones_like(dt_fake)
    loss_t = F.binary_cross_entropy_with_logits(dt_fake, target)
    target_s = torch.ones_like(ds_fake)
    loss_s = F.binary_cross_entropy_with_logits(ds_fake, target_s)
    return lambda_t * loss_t + lambda_s * loss_s


def discriminator_loss(
    dt_real: torch.Tensor,
    ds_real: torch.Tensor,
    dt_fake: torch.Tensor,
    ds_fake: torch.Tensor,
    lambda_t: float = 1.0,
    lambda_s: float = 1.0,
) -> torch.Tensor:
    loss_t = F.binary_cross_entropy_with_logits(dt_real, torch.ones_like(dt_real))
    loss_t = loss_t + F.binary_cross_entropy_with_logits(dt_fake, torch.zeros_like(dt_fake))
    loss_s = F.binary_cross_entropy_with_logits(ds_real, torch.ones_like(ds_real))
    loss_s = loss_s + F.binary_cross_entropy_with_logits(ds_fake, torch.zeros_like(ds_fake))
    return lambda_t * loss_t + lambda_s * loss_s
