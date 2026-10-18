import argparse
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def image_paths(root):
    root = Path(root)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(str(root))
    return paths


def base_name(path, suffix):
    name = path.stem
    return name[: -len(suffix)] if name.endswith(suffix) else name


def load_image(path, device):
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float().unsqueeze(0).to(device) / 255.0


def psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    sr_map = {base_name(path, "_SR"): path for path in image_paths(args.sr_dir)}
    hr_paths = image_paths(args.hr_dir)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    psnr_values = []
    lpips_values = []
    with torch.no_grad():
        for hr_path in hr_paths:
            key = base_name(hr_path, "_HR")
            if key not in sr_map:
                raise FileNotFoundError(f"missing SR image for {hr_path.name}")
            sr = load_image(sr_map[key], device)
            hr = load_image(hr_path, device)
            if sr.shape != hr.shape:
                raise ValueError(f"shape mismatch: {sr_map[key].name} and {hr_path.name}")
            psnr_values.append(psnr(sr, hr))
            lpips_values.append(metric(sr * 2.0 - 1.0, hr * 2.0 - 1.0).item())
    print(f"Average PSNR: {sum(psnr_values) / len(psnr_values):.4f}")
    print(f"Average LPIPS: {sum(lpips_values) / len(lpips_values):.6f}")


if __name__ == "__main__":
    main()
