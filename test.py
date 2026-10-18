from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bsr.data import PairedImageDataset
from bsr.utils import build_generator, load_checkpoint, save_image_tensor, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FDG-BSR or DAL-BSR inference.")
    parser.add_argument("--model", default="fdg-bsr", choices=["fdg-bsr", "dal-bsr"])
    parser.add_argument("--checkpoint", default="", help="Optional .pth checkpoint.")
    parser.add_argument("--input", "--lr-dir", dest="lr_dir", required=True, help="Directory containing LR images.")
    parser.add_argument("--output", "--out-dir", dest="out_dir", default="results")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    groups = args.num_groups if args.num_groups is not None else (6 if args.model == "fdg-bsr" else 11)
    model = build_generator(args.model, channels=args.channels, num_groups=groups, scale=args.scale).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dataset = PairedImageDataset(args.lr_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    out_dir = Path(args.out_dir)
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            name = batch["name"][0]
            sr = model(lr).clamp(0, 1)
            save_image_tensor(sr, out_dir / f"{sr_name(name)}.png")


def sr_name(name: str) -> str:
    return f"{name[:-3] if name.endswith('_LR') else name}_SR"


if __name__ == "__main__":
    main()
