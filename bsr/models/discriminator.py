from __future__ import annotations

from torch import nn
from torch.nn import functional as F


def snconv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    padding = dilation * (kernel_size // 2)
    return nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
    )


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(snconv(in_channels, out_channels, 3, stride=stride, dilation=dilation), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.block(x)


class TSCD(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.texture_branch = nn.Sequential(ConvAct(3, c), ConvAct(c, c))
        self.structure_branch = nn.Sequential(ConvAct(3, c), ConvAct(c, c, dilation=2))

        self.enc1 = ConvAct(c * 2, c)
        self.enc2 = ConvAct(c, c * 2, stride=2)
        self.enc3 = ConvAct(c * 2, c * 4, stride=2)
        self.bottleneck = ConvAct(c * 4, c * 4)
        self.dec2 = ConvAct(c * 4 + c * 2, c * 2)
        self.dec1 = ConvAct(c * 2 + c, c)
        self.shared_out = ConvAct(c, c)

        self.texture_head = nn.Sequential(ConvAct(c, c), snconv(c, 1, 3))
        self.structure_head = nn.Sequential(ConvAct(c, c, dilation=2), snconv(c, 1, 3, dilation=2))

    def forward(self, x):
        ft = self.texture_branch(x)
        fs = self.structure_branch(x)
        f = self.enc1(F.concat([ft, fs], dim=1))
        e2 = self.enc2(f)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(F.concat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=f.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(F.concat([d1, f], dim=1))
        fd = self.shared_out(d1)
        return self.texture_head(fd), self.structure_head(fd)

