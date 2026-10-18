from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bsr.data import UnpairedHRDataset
from bsr.contrastive import (
    DegradationContrastiveQueue,
    ProjectionHead,
    clone_momentum_module,
    momentum_update,
    prior_weighted_moco_loss,
)
from bsr.losses import PerceptualLoss, discriminator_loss, generator_adv_loss
from bsr.models import TSCD
from bsr.utils import build_generator, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FDG-BSR or DAL-BSR.")
    parser.add_argument("--model", default="fdg-bsr", choices=["fdg-bsr", "dal-bsr"])
    parser.add_argument("--train-hr", required=True, help="Directory containing HR training images.")
    parser.add_argument("--work-dir", default="runs/experiment")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--resume", default="")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-groups", type=int, default=None)
    parser.add_argument("--lambda-con", type=float, default=0.1)
    parser.add_argument("--lambda-per", type=float, default=1.0)
    parser.add_argument("--lambda-adv", type=float, default=5e-3)
    parser.add_argument("--contrast-dim", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--prior-temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-perceptual", action="store_true")
    return parser.parse_args()


def set_stage_trainable(model: torch.nn.Module, stage: int) -> None:
    for p in model.parameters():
        p.requires_grad_(stage != 1)
    if stage == 1:
        for p in model.degradation_encoder_parameters():
            p.requires_grad_(True)


def make_model(args: argparse.Namespace) -> torch.nn.Module:
    groups = args.num_groups
    if groups is None:
        groups = 6 if args.model == "fdg-bsr" else 11
    kwargs = {"channels": args.channels, "num_groups": groups, "scale": args.scale}
    return build_generator(args.model, **kwargs)


def build_contrastive_modules(model: torch.nn.Module, args: argparse.Namespace, device: torch.device):
    online_encoder = model.degradation_encoder_module()
    momentum_encoder = clone_momentum_module(online_encoder).to(device)
    projector = ProjectionHead(model.degradation_embedding_dim, out_dim=args.contrast_dim).to(device)
    momentum_projector = clone_momentum_module(projector).to(device)
    queue = DegradationContrastiveQueue(
        embedding_dim=args.contrast_dim,
        prior_dim=28,
        queue_size=args.queue_size,
    ).to(device)
    return online_encoder, momentum_encoder, projector, momentum_projector, queue


def encode_for_contrast(encoder: torch.nn.Module, projector: torch.nn.Module, lr: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    feat = encoder.encode(lr, prior)
    pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    return projector(pooled)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnpairedHRDataset(args.train_hr, crop_size=args.crop_size, scale=args.scale, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    iterator = iter(loader)

    model = make_model(args).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    set_stage_trainable(model, args.stage)

    online_encoder = momentum_encoder = projector = momentum_projector = queue = None
    contrastive_params = []
    if args.stage in {1, 2}:
        online_encoder, momentum_encoder, projector, momentum_projector, queue = build_contrastive_modules(model, args, device)
        if args.resume:
            if "projector" in ckpt:
                projector.load_state_dict(ckpt["projector"], strict=False)
            if "momentum_encoder" in ckpt:
                momentum_encoder.load_state_dict(ckpt["momentum_encoder"], strict=False)
            if "momentum_projector" in ckpt:
                momentum_projector.load_state_dict(ckpt["momentum_projector"], strict=False)
            if "contrast_queue" in ckpt:
                queue.load_state_dict(ckpt["contrast_queue"], strict=False)
        contrastive_params = list(projector.parameters())

    lr = args.lr
    if lr is None:
        lr = 2e-4 if args.stage in {1, 2} else 1e-4
    opt_g = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad] + contrastive_params,
        lr=lr,
        betas=(0.9, 0.99),
    )

    discriminator = None
    opt_d = None
    perceptual = None
    if args.stage == 3:
        discriminator = TSCD().to(device)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))
        perceptual = PerceptualLoss(enabled=not args.no_perceptual)

    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        lr_img = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        prior = batch["prior"].to(device)

        if args.stage == 1:
            assert online_encoder is not None and momentum_encoder is not None
            assert projector is not None and momentum_projector is not None and queue is not None
            q = encode_for_contrast(online_encoder, projector, lr_img, prior)
            with torch.no_grad():
                momentum_update(online_encoder, momentum_encoder, args.momentum)
                momentum_update(projector, momentum_projector, args.momentum)
                k = encode_for_contrast(momentum_encoder, momentum_projector, lr_img, prior)
            loss_g = prior_weighted_moco_loss(
                q,
                k,
                prior,
                queue,
                temperature=args.temperature,
                prior_temperature=args.prior_temperature,
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            with torch.no_grad():
                queue.enqueue(k, prior)
            log = {"con": loss_g.item()}

        elif args.stage == 2:
            assert online_encoder is not None and momentum_encoder is not None
            assert projector is not None and momentum_projector is not None and queue is not None
            sr = model(lr_img)
            loss_l1 = F.l1_loss(sr, hr)
            q = encode_for_contrast(online_encoder, projector, lr_img, prior)
            with torch.no_grad():
                momentum_update(online_encoder, momentum_encoder, args.momentum)
                momentum_update(projector, momentum_projector, args.momentum)
                k = encode_for_contrast(momentum_encoder, momentum_projector, lr_img, prior)
            loss_con = prior_weighted_moco_loss(
                q,
                k,
                prior,
                queue,
                temperature=args.temperature,
                prior_temperature=args.prior_temperature,
            )
            loss_g = loss_l1 + args.lambda_con * loss_con
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            with torch.no_grad():
                queue.enqueue(k, prior)
            log = {"l1": loss_l1.item(), "con": loss_con.item(), "total": loss_g.item()}

        else:
            assert discriminator is not None and opt_d is not None and perceptual is not None
            sr = model(lr_img)
            with torch.no_grad():
                sr_detached = sr.detach()
            dt_real, ds_real = discriminator(hr)
            dt_fake, ds_fake = discriminator(sr_detached)
            loss_d = discriminator_loss(dt_real, ds_real, dt_fake, ds_fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            dt_fake_g, ds_fake_g = discriminator(sr)
            loss_l1 = F.l1_loss(sr, hr)
            loss_per = perceptual(sr, hr)
            loss_adv = generator_adv_loss(dt_fake_g, ds_fake_g)
            loss_g = loss_l1 + args.lambda_per * loss_per + args.lambda_adv * loss_adv
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            log = {"l1": loss_l1.item(), "per": loss_per.item(), "adv": loss_adv.item(), "d": loss_d.item()}

        if step == 1 or step % 20 == 0:
            msg = " ".join(f"{k}={v:.4f}" for k, v in log.items())
            print(f"[{args.model} stage{args.stage}] step {step}/{args.steps} {msg}")

        if step % args.save_every == 0 or step == args.steps:
            extra = {}
            if args.stage in {1, 2}:
                assert momentum_encoder is not None and momentum_projector is not None and projector is not None and queue is not None
                extra = {
                    "projector": projector.state_dict(),
                    "momentum_encoder": momentum_encoder.state_dict(),
                    "momentum_projector": momentum_projector.state_dict(),
                    "contrast_queue": queue.state_dict(),
                }
            save_checkpoint(work_dir / f"{args.model}_stage{args.stage}_{step}.pth", model, opt_g, step, extra=extra)
            if discriminator is not None:
                save_checkpoint(work_dir / f"tscd_stage{args.stage}_{step}.pth", discriminator, opt_d, step)


if __name__ == "__main__":
    main()
