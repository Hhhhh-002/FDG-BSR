from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import default_conv


class LightweightDegradationBlock(nn.Module):
    def __init__(self, channels: int = 48, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.dw5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.fuse = nn.Conv2d(hidden * 2, channels, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fe = self.expand(x)
        ff = self.fuse(torch.cat([self.dw3(fe), self.dw5(fe)], dim=1))
        g = self.gate(ff)[:, :, None, None]
        return x + g * ff


class DegradationContextModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 48,
        out_channels: int = 64,
        degradation_prior_dim: int = 28,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(default_conv(in_channels, feat_channels, 3), nn.GELU())
        self.prior_mlp = nn.Sequential(
            nn.Linear(degradation_prior_dim, feat_channels),
            nn.GELU(),
            nn.Linear(feat_channels, feat_channels),
        )
        self.encoder = nn.Sequential(LightweightDegradationBlock(feat_channels), LightweightDegradationBlock(feat_channels))
        self.global_branch = nn.Sequential(
            default_conv(feat_channels, feat_channels, 3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channels, out_channels),
        )
        self.artifact_dw = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 5, padding=2, groups=feat_channels),
            nn.GELU(),
        )
        self.artifact_fc = nn.Linear(feat_channels, out_channels // 2)

    def forward(
        self,
        lr: torch.Tensor,
        degradation_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fd = self.encode(lr, degradation_prior)
        zd = self.global_branch(fd)
        za = self.artifact_fc(F.adaptive_max_pool2d(self.artifact_dw(fd), 1).flatten(1))
        return zd, za

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


class GroupSharedKernelGenerator(nn.Module):
    def __init__(self, condition_dim: int = 64, channels: int = 128, kernel_size: int = 5, num_bases: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_bases = num_bases
        self.coeff = nn.Linear(condition_dim, num_bases)
        self.bases = nn.Parameter(torch.randn(num_bases, channels, kernel_size, kernel_size) * 0.02)

    def forward(self, zd: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.coeff(zd), dim=1)
        return torch.einsum("bk,kchw->bchw", alpha, self.bases)


def dynamic_depthwise_conv(x: torch.Tensor, kernel: torch.Tensor, padding: int = 2) -> torch.Tensor:
    outs = []
    for xi, wi in zip(x, kernel):
        outs.append(F.conv2d(xi.unsqueeze(0), wi.unsqueeze(1), padding=padding, groups=x.shape[1]))
    return torch.cat(outs, dim=0)


class CoupledFusionBlock(nn.Module):
    def __init__(self, channels: int = 64, expanded_channels: int = 128, artifact_dim: int = 32) -> None:
        super().__init__()
        self.expand = nn.Conv2d(channels, expanded_channels, 1)
        self.project = nn.Conv2d(expanded_channels, channels, 1)
        self.struct = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0)),
        )
        self.freq_dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.freq_gate = nn.Sequential(nn.Linear(artifact_dim, channels), nn.Sigmoid())
        self.mix = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor, kernel: torch.Tensor, za: torch.Tensor) -> torch.Tensor:
        dyn = self.project(dynamic_depthwise_conv(F.gelu(self.expand(x)), kernel, padding=2))
        struct = self.struct(x)
        freq_gate = self.freq_gate(za)[:, :, None, None]
        freq = freq_gate * F.gelu(self.freq_dw(x))
        return x + self.mix(torch.cat([dyn, struct, freq], dim=1))


class LightweightGatedFusion(nn.Module):
    def __init__(self, channels: int = 64, condition_dim: int = 64) -> None:
        super().__init__()
        self.compress = nn.Conv2d(channels * 2, channels, 1)
        self.gate = nn.Sequential(nn.Linear(condition_dim, channels), nn.Sigmoid())
        self.out = default_conv(channels, channels, 3)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
        fc = self.compress(torch.cat([h1, h2], dim=1))
        gd = self.gate(zd)[:, :, None, None]
        return h2 + self.out(gd * fc)


class ProgressiveRestorationGroup(nn.Module):
    def __init__(self, channels: int = 64, condition_dim: int = 64, artifact_dim: int = 32) -> None:
        super().__init__()
        self.gskg = GroupSharedKernelGenerator(condition_dim=condition_dim, channels=128)
        self.cfb1 = CoupledFusionBlock(channels, 128, artifact_dim)
        self.cfb2 = CoupledFusionBlock(channels, 128, artifact_dim)
        self.lgf = LightweightGatedFusion(channels, condition_dim)
        self.cfb3 = CoupledFusionBlock(channels, 128, artifact_dim)
        self.cfb4 = CoupledFusionBlock(channels, 128, artifact_dim)
        self.tail = default_conv(channels, channels, 3)

    def forward(self, x: torch.Tensor, zd: torch.Tensor, za: torch.Tensor) -> torch.Tensor:
        wg = self.gskg(zd)
        h1 = self.cfb1(x, wg, za)
        h2 = self.cfb2(h1, wg, za)
        h2_hat = self.lgf(h1, h2, zd)
        h3 = self.cfb3(h2_hat, wg, za)
        h4 = self.cfb4(h3, wg, za)
        return self.tail(h4) + x


class SharedUpsampleUnit(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.conv = default_conv(channels, channels * 4, 3)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class WeightSharedUpsamplingHead(nn.Module):
    def __init__(self, channels: int = 64, mid_channels: int = 32, scale: int = 4) -> None:
        super().__init__()
        if scale != 4:
            raise ValueError("WUH currently implements the thesis x4 setting.")
        self.scale = scale
        self.fuse = default_conv(channels, channels, 3)
        self.compress = nn.Sequential(nn.Conv2d(channels, mid_channels, 1), nn.GELU())
        self.shared_up = SharedUpsampleUnit(mid_channels)
        self.out = nn.Sequential(default_conv(mid_channels, mid_channels, 3), nn.GELU(), default_conv(mid_channels, 3, 3))

    def forward(self, fn: torch.Tensor, f0: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        fr = self.fuse(fn) + f0
        u = self.compress(fr)
        u = self.shared_up(u)
        u = self.shared_up(u)
        residual = self.out(u)
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return base + residual


class DALBSR(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_groups: int = 11,
        scale: int = 4,
        degradation_prior_dim: int = 28,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.degradation_embedding_dim = 48
        self.sfe = default_conv(3, channels, 3)
        self.dcm = DegradationContextModule(out_channels=channels, degradation_prior_dim=degradation_prior_dim)
        self.groups = nn.ModuleList([ProgressiveRestorationGroup(channels) for _ in range(num_groups)])
        self.wuh = WeightSharedUpsamplingHead(channels, scale=scale)

    def forward(self, lr: torch.Tensor, degradation_prior: torch.Tensor | None = None) -> torch.Tensor:
        f0 = self.sfe(lr)
        zd, za = self.dcm(lr, degradation_prior)
        feat = f0
        for group in self.groups:
            feat = group(feat, zd, za)
        return self.wuh(feat, f0, lr)

    def encode_degradation(self, lr: torch.Tensor, degradation_prior: torch.Tensor | None = None) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.dcm.encode(lr, degradation_prior), 1).flatten(1)

    def degradation_encoder_parameters(self):
        yield from self.dcm.shared_encoder_parameters()

    def degradation_encoder_module(self) -> nn.Module:
        return self.dcm
