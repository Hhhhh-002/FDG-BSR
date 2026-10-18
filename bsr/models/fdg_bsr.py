from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    GatedFFN,
    PixelShuffleBlock,
    RDE,
    ResidualDenseBlock,
    ShiftedWindowBlock,
    default_conv,
)


class FactorizedIDR(nn.Module):
    def __init__(self, channels: int = 64, degradation_prior_dim: int = 28) -> None:
        super().__init__()
        self.stem = nn.Sequential(default_conv(3, channels, 3), nn.LeakyReLU(0.2, inplace=True))
        self.prior_mlp = nn.Sequential(
            nn.Linear(degradation_prior_dim, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels, channels),
        )
        self.encoder = nn.Sequential(RDE(channels), RDE(channels))
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels, channels),
        )
        self.spatial_branch = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.channel_conv = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.LeakyReLU(0.2, inplace=True))
        self.channel_fc = nn.Linear(channels, channels)

    def forward(
        self,
        lr: torch.Tensor,
        degradation_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fd = self.encode(lr, degradation_prior)
        zg = self.global_branch(fd)
        zs = self.spatial_branch(fd)
        zc = self.channel_fc(F.adaptive_avg_pool2d(self.channel_conv(fd), 1).flatten(1))
        return zg, zs, zc

    def encode(self, lr: torch.Tensor, degradation_prior: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.stem(lr)
        if degradation_prior is not None:
            prior_feat = self.prior_mlp(degradation_prior.to(device=lr.device, dtype=lr.dtype))
            feat = feat + prior_feat[:, :, None, None]
        return self.encoder(feat)

    def shared_encoder_parameters(self):
        yield from self.stem.parameters()
        yield from self.prior_mlp.parameters()
        yield from self.encoder.parameters()


class DualBranchDMR(nn.Module):
    def __init__(self, channels: int = 64, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.local = nn.Sequential(
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
            ResidualDenseBlock(channels),
        )
        self.global_branch = nn.Sequential(
            ShiftedWindowBlock(channels, window_size, num_heads, shift=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            GatedFFN(channels),
        )
        self.channel_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.out = default_conv(channels, channels, 3)

    def forward(self, x: torch.Tensor, zg: torch.Tensor, zs: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        fl = x + self.local(x)
        fg = self.global_branch(x)
        zg_map = zg[:, :, None, None]
        zc_gate = self.channel_gate(zc)[:, :, None, None]
        zs_gate = F.interpolate(zs, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fused = (fl + fg + zg_map) * zc_gate * zs_gate
        return x + self.out(fused)


class CrossGatedFusion(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.out = default_conv(channels // 2, channels, 3)

    def forward(self, f0: torch.Tensor, prev: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        fm = self.mix(torch.cat([f0, prev, current], dim=1))
        fa, fb = fm.chunk(2, dim=1)
        return self.out(F.gelu(fa) * fb)


class FeatureReconstructionGroup(nn.Module):
    def __init__(self, channels: int = 64, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.ddmr1 = DualBranchDMR(channels, window_size, num_heads)
        self.ddmr2 = DualBranchDMR(channels, window_size, num_heads)
        self.cgf = CrossGatedFusion(channels)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        zg: torch.Tensor,
        zs: torch.Tensor,
        zc: torch.Tensor,
    ) -> torch.Tensor:
        h1 = self.ddmr1(x, zg, zs, zc)
        h2 = self.ddmr2(h1, zg, zs, zc)
        return self.cgf(f0, x, h2)


class UpsamplingReconstructionHead(nn.Module):
    def __init__(self, channels: int = 64, scale: int = 4) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError("URH currently implements the thesis x4 setting.")
        self.fuse = default_conv(channels, channels, 3)
        self.up1 = PixelShuffleBlock(channels, act="lrelu")
        self.up2 = PixelShuffleBlock(channels, act="lrelu")
        self.out = nn.Sequential(default_conv(channels, channels, 3), nn.LeakyReLU(0.2, inplace=True), default_conv(channels, 3, 3))

    def forward(self, fn: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        fu = self.fuse(fn) + f0
        return self.out(self.up2(self.up1(fu)))


class FDGBSR(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 6,
        scale: int = 4,
        window_size: int = 8,
        num_heads: int = 4,
        degradation_prior_dim: int = 28,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.degradation_embedding_dim = channels
        self.sfe = default_conv(3, channels, 3)
        self.fidr = FactorizedIDR(channels, degradation_prior_dim)
        self.groups = nn.ModuleList(
            [FeatureReconstructionGroup(channels, window_size, num_heads) for _ in range(num_groups)]
        )
        self.urh = UpsamplingReconstructionHead(channels, scale)

    def forward(self, lr: torch.Tensor, degradation_prior: torch.Tensor | None = None) -> torch.Tensor:
        f0 = self.sfe(lr)
        zg, zs, zc = self.fidr(lr, degradation_prior)
        feat = f0
        for group in self.groups:
            feat = group(feat, f0, zg, zs, zc)
        return self.urh(feat, f0)

    def encode_degradation(self, lr: torch.Tensor, degradation_prior: torch.Tensor | None = None) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.fidr.encode(lr, degradation_prior), 1).flatten(1)

    def degradation_encoder_parameters(self):
        yield from self.fidr.shared_encoder_parameters()

    def degradation_encoder_module(self) -> nn.Module:
        return self.fidr
