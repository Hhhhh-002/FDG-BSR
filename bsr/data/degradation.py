from __future__ import annotations

import io
import random
from dataclasses import dataclass

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


DEGRADATION_PRIOR_DIM = 28
BLUR_TYPES = ["iso", "aniso", "generalized", "plateau"]
RESIZE_MODES = ["up", "down", "keep"]
INTERP_MODES = ["bicubic", "bilinear", "area"]


@dataclass
class HHRDConfig:
    scale: int = 4
    kernel_min: int = 7
    kernel_max: int = 21
    sigma1: tuple[float, float] = (0.2, 3.0)
    sigma2: tuple[float, float] = (0.2, 1.5)
    noise_prob: float = 0.5
    jpeg_prob: float = 0.4
    jpeg_quality: tuple[int, int] = (50, 95)
    second_blur_prob: float = 0.8
    sinc_prob: float = 0.2


def _odd_kernel(min_k: int, max_k: int) -> int:
    values = list(range(min_k | 1, max_k + 1, 2))
    return random.choice(values)


def _gaussian_kernel(kernel_size: int, sigma_x: float, sigma_y: float | None = None) -> torch.Tensor:
    sigma_y = sigma_x if sigma_y is None else sigma_y
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() / (2 * sigma_x**2) + yy.square() / (2 * sigma_y**2)))
    return kernel / kernel.sum()


def _apply_kernel(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    kernel = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, *kernel.shape).repeat(c, 1, 1, 1)
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=c)


def _random_blur_with_params(
    x: torch.Tensor,
    sigma_range: tuple[float, float],
    min_k: int,
    max_k: int,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    k = _odd_kernel(min_k, max_k)
    kind = random.choice(BLUR_TYPES)
    sigma_x = random.uniform(*sigma_range)
    sigma_y = sigma_x if kind == "iso" else random.uniform(*sigma_range)
    kernel = _gaussian_kernel(k, sigma_x, sigma_y)
    if kind == "generalized":
        kernel = kernel.pow(random.uniform(0.7, 1.5))
        kernel = kernel / kernel.sum()
    elif kind == "plateau":
        kernel = torch.sqrt(kernel)
        kernel = kernel / kernel.sum()
    params = {"kernel": float(k), "blur_type": kind, "sigma_x": sigma_x, "sigma_y": sigma_y}
    return _apply_kernel(x, kernel), params


def _resize_random_with_params(
    x: torch.Tensor,
    target_size: tuple[int, int] | None = None,
    probs=(0.2, 0.7, 0.1),
) -> tuple[torch.Tensor, dict[str, float | str]]:
    mode = random.choices(RESIZE_MODES, weights=probs, k=1)[0]
    interp = random.choices(INTERP_MODES, weights=[0.4, 0.3, 0.3], k=1)[0]
    h, w = x.shape[-2:]
    if target_size is None:
        if mode == "up":
            factor = random.uniform(1.0, 1.5)
        elif mode == "down":
            factor = random.uniform(0.5, 1.0)
        else:
            factor = 1.0
        target_size = (max(8, int(h * factor)), max(8, int(w * factor)))
    else:
        factor = target_size[0] / max(1, h)
    kwargs = {} if interp == "area" else {"align_corners": False}
    params = {"resize_mode": mode, "interp": interp, "resize_factor": float(factor)}
    return F.interpolate(x, size=target_size, mode=interp, **kwargs), params


def _add_noise_with_params(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | str]]:
    if random.random() > 0.5:
        sigma = random.uniform(1.0, 25.0) / 255.0
        if random.random() < 0.4:
            noise = torch.randn(x.shape[0], 1, *x.shape[-2:], device=x.device, dtype=x.dtype)
        else:
            noise = torch.randn_like(x)
        params = {"noise_type": "gaussian", "noise_level": sigma * 255.0 / 25.0}
        return (x + noise * sigma).clamp(0, 1), params
    vals = 2 ** random.uniform(3, 8)
    params = {"noise_type": "poisson", "noise_level": vals / 256.0}
    return torch.poisson((x * vals).clamp(0, vals)) / vals, params


def _jpeg_compress_with_params(x: torch.Tensor, quality_range: tuple[int, int]) -> tuple[torch.Tensor, dict[str, float]]:
    quality = random.randint(*quality_range)
    outs = []
    for img in x.detach().cpu():
        pil = to_pil_image(img.clamp(0, 1))
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        outs.append(pil_to_tensor(Image.open(buffer).convert("RGB")).float() / 255.0)
    return torch.stack(outs, dim=0).to(device=x.device, dtype=x.dtype), {"jpeg_quality": quality / 100.0}


def _sinc_artifact(x: torch.Tensor) -> torch.Tensor:
    blurred = _apply_kernel(x, _gaussian_kernel(7, 1.0))
    return (x + 0.15 * (x - blurred)).clamp(0, 1)


def _one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _build_prior(
    cfg: HHRDConfig,
    blur1: dict[str, float | str],
    resize1: dict[str, float | str],
    resize2: dict[str, float | str],
    resize3: dict[str, float | str],
    noise1: dict[str, float | str] | None,
    noise2: dict[str, float | str] | None,
    second_blur: dict[str, float | str] | None,
    jpeg: dict[str, float] | None,
    sinc_flag: bool,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
) -> torch.Tensor:
    noise1_type = "gaussian" if noise1 is None else str(noise1["noise_type"])
    noise2_type = "gaussian" if noise2 is None else str(noise2["noise_type"])
    values = [
        float(cfg.scale) / 4.0,
        float(blur1["kernel"]) / float(cfg.kernel_max),
        *_one_hot(str(blur1["blur_type"]), BLUR_TYPES),
        float(blur1["sigma_x"]) / cfg.sigma1[1],
        float(blur1["sigma_y"]) / cfg.sigma1[1],
        float(resize1["resize_factor"]),
        *_one_hot(str(resize1["resize_mode"]), RESIZE_MODES),
        *_one_hot(str(resize1["interp"]), INTERP_MODES),
        float(resize2["resize_factor"]),
        1.0 if noise1 is not None else 0.0,
        0.0 if noise1_type == "gaussian" else 1.0,
        0.0 if noise1 is None else float(noise1["noise_level"]),
        float(resize3["resize_factor"]),
        1.0 if noise2 is not None else 0.0,
        0.0 if noise2_type == "gaussian" else 1.0,
        0.0 if noise2 is None else float(noise2["noise_level"]),
        1.0 if second_blur is not None else 0.0,
        0.0 if second_blur is None else float(second_blur["sigma_x"]) / cfg.sigma2[1],
        1.0 if jpeg is not None else 0.0,
        0.0 if jpeg is None else float(jpeg["jpeg_quality"]),
        1.0 if sinc_flag else 0.0,
    ]
    prior = torch.tensor(values, dtype=dtype, device=device)
    if prior.numel() != DEGRADATION_PRIOR_DIM:
        raise RuntimeError(f"Expected {DEGRADATION_PRIOR_DIM} degradation prior values, got {prior.numel()}")
    return prior.unsqueeze(0).repeat(batch, 1)


class HHRD:
    def __init__(self, config: HHRDConfig | None = None) -> None:
        self.config = config or HHRDConfig()

    def __call__(self, hr: torch.Tensor, return_params: bool = False):
        single = hr.dim() == 3
        if single:
            hr = hr.unsqueeze(0)
        cfg = self.config
        target = (hr.shape[-2] // cfg.scale, hr.shape[-1] // cfg.scale)

        x, blur1 = _random_blur_with_params(hr, cfg.sigma1, cfg.kernel_min, cfg.kernel_max)
        x, resize1 = _resize_random_with_params(x, probs=(0.2, 0.7, 0.1))

        noise1_params = None
        noise2_params = None
        resize2_params: dict[str, float | str] = {"resize_mode": "keep", "interp": "bicubic", "resize_factor": 1.0}
        resize3_params: dict[str, float | str] = {"resize_mode": "keep", "interp": "bicubic", "resize_factor": 1.0}

        def resize2_op(y: torch.Tensor) -> torch.Tensor:
            nonlocal resize2_params
            y, resize2_params = _resize_random_with_params(y, probs=(0.3, 0.4, 0.3))
            return y

        def noise1_op(y: torch.Tensor) -> torch.Tensor:
            nonlocal noise1_params
            y, noise1_params = _add_noise_with_params(y)
            return y

        ops = [resize2_op]
        if random.random() < cfg.noise_prob:
            ops.append(noise1_op)
        random.shuffle(ops)
        for op in ops:
            x = op(x)

        second_blur = None
        if random.random() < cfg.second_blur_prob:
            x, second_blur = _random_blur_with_params(x, cfg.sigma2, cfg.kernel_min, cfg.kernel_max)

        jpeg_params = None

        def resize3_op(y: torch.Tensor) -> torch.Tensor:
            nonlocal resize3_params
            y, resize3_params = _resize_random_with_params(y, probs=(0.3, 0.4, 0.3))
            return y

        def noise2_op(y: torch.Tensor) -> torch.Tensor:
            nonlocal noise2_params
            y, noise2_params = _add_noise_with_params(y)
            return y

        ops2 = [resize3_op]
        if random.random() < cfg.noise_prob:
            ops2.append(noise2_op)
        if random.random() < cfg.jpeg_prob:
            def jpeg_op(y: torch.Tensor) -> torch.Tensor:
                nonlocal jpeg_params
                y, jpeg_params = _jpeg_compress_with_params(y, cfg.jpeg_quality)
                return y

            ops2.append(jpeg_op)
        random.shuffle(ops2)
        for op in ops2:
            x = op(x)

        x = F.interpolate(x, size=target, mode="bicubic", align_corners=False).clamp(0, 1)
        sinc_flag = random.random() < cfg.sinc_prob
        if sinc_flag:
            x = _sinc_artifact(x)

        params = _build_prior(
            cfg,
            blur1,
            resize1,
            resize2_params,
            resize3_params,
            noise1_params,
            noise2_params,
            second_blur,
            jpeg_params,
            sinc_flag,
            hr.device,
            hr.dtype,
            hr.shape[0],
        )
        if single:
            x = x.squeeze(0)
            params = params.squeeze(0)
        return (x, params) if return_params else x
