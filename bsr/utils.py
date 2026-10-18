from __future__ import annotations

import random
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from .models import DALBSR, FDGBSR


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_generator(name: str, **kwargs):
    name = name.lower().replace("_", "-")
    if name in {"fdg", "fdg-bsr", "fdgbsr"}:
        return FDGBSR(**kwargs)
    if name in {"dal", "dal-bsr", "dalbsr"}:
        return DALBSR(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img[0]
    to_pil_image(img).save(path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return ckpt


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, step: int = 0, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
