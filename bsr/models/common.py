from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32, num_layers: int = 5) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = channels
        for i in range(num_layers):
            out_channels = channels if i == num_layers - 1 else growth_channels
            self.layers.append(default_conv(in_channels, out_channels, 3))
            in_channels += growth_channels
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for i, conv in enumerate(self.layers):
            y = conv(torch.cat(feats, dim=1))
            if i != len(self.layers) - 1:
                y = self.act(y)
                feats.append(y)
        return x + 0.2 * y


class RDE(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


class WindowAttention(nn.Module):
    def __init__(self, channels: int = 64, window_size: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj(out)


class ShiftedWindowBlock(nn.Module):
    def __init__(self, channels: int = 64, window_size: int = 8, num_heads: int = 4, shift: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm = LayerNorm2d(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        shortcut = x
        x = self.norm(x)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = x.shape[-2:]
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        windows = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, hp, wp, b)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        x = x[:, :, :h, :w]
        return shortcut + x


class GatedFFN(nn.Module):
    def __init__(self, channels: int = 64, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(self.project_in(self.norm(x)))
        x1, x2 = x.chunk(2, dim=1)
        return shortcut + self.project_out(F.gelu(x1) * x2)


class PixelShuffleBlock(nn.Module):
    def __init__(self, channels: int, act: str = "lrelu") -> None:
        super().__init__()
        self.conv = default_conv(channels, channels * 4, 3)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act == "lrelu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


