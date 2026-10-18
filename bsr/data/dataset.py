from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .degradation import HHRD


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_rgb(path: Path) -> torch.Tensor:
    return pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0


def random_crop(hr: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = hr.shape
    if h < size or w < size:
        target_h = max(size, h)
        target_w = max(size, w)
        hr = torch.nn.functional.interpolate(
            hr.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        _, h, w = hr.shape
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return hr[:, top : top + size, left : left + size]


def augment_pair(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[2])
        hr = torch.flip(hr, dims=[2])
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=[1])
        hr = torch.flip(hr, dims=[1])
    k = random.randint(0, 3)
    if k:
        lr = torch.rot90(lr, k, dims=[1, 2])
        hr = torch.rot90(hr, k, dims=[1, 2])
    return lr, hr


class UnpairedHRDataset(Dataset):
    def __init__(self, hr_dir: str | Path, crop_size: int = 256, scale: int = 4, augment: bool = True) -> None:
        self.paths = list_images(hr_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {hr_dir}")
        self.crop_size = crop_size
        self.scale = scale
        self.augment = augment
        self.degrader = HHRD()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        hr = random_crop(load_rgb(self.paths[idx]), self.crop_size)
        lr, prior = self.degrader(hr, return_params=True)
        if self.augment:
            lr, hr = augment_pair(lr, hr)
        return {"lr": lr, "hr": hr, "prior": prior, "name": self.paths[idx].stem}


class PairedImageDataset(Dataset):
    def __init__(self, lr_dir: str | Path, hr_dir: str | Path | None = None) -> None:
        self.lr_paths = list_images(lr_dir)
        if not self.lr_paths:
            raise FileNotFoundError(f"No LR images found under {lr_dir}")
        self.hr_paths = None
        if hr_dir:
            hr_paths = list_images(hr_dir)
            stem_map = {p.stem: p for p in hr_paths}
            self.hr_paths = [stem_map.get(p.stem) for p in self.lr_paths]
            if any(p is None for p in self.hr_paths):
                self.hr_paths = hr_paths[: len(self.lr_paths)]

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {"lr": load_rgb(self.lr_paths[idx]), "name": self.lr_paths[idx].stem}
        if self.hr_paths is not None and idx < len(self.hr_paths):
            item["hr"] = load_rgb(self.hr_paths[idx])
        return item
