#!/usr/bin/env python
"""Train BraillyNet on synthetic pages, optionally mixed with real annotated photos.

Examples
--------
# synthetic only
python scripts/train.py --out runs/synth --epochs 30

# fine-tune on real photos (Angelina-format CSV annotations) + synthetic mix
python scripts/train.py --out runs/ft --init runs/synth/best.pt \
    --real-list data/AngelinaDataset/books/train.txt data/AngelinaDataset/handwritten/train.txt \
    --real-ratio 0.6 --epochs 20 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.data import RealDataset, SynthDataset, read_image_list  # noqa: E402
from brailly.losses import brailly_loss  # noqa: E402
from brailly.model import BraillyNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--synth-per-epoch", type=int, default=4000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--real-list", nargs="*", default=[], help="image list .txt files (Angelina format)")
    ap.add_argument("--real-ratio", type=float, default=0.5, help="fraction of real crops per epoch")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this wall-time (0 = off)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt (optimizer + schedule)")
    ap.add_argument("--epochs-per-run", type=int, default=0, help="exit after N epochs (use with --resume)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model = BraillyNet(pretrained=args.init is None)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)

    synth = SynthDataset(args.synth_per_epoch, args.size, seed=1)
    datasets = [synth]
    real = None
    if args.real_list:
        imgs = [p for lf in args.real_list for p in read_image_list(lf)]
        n_real = int(args.synth_per_epoch * args.real_ratio / max(1e-6, 1 - args.real_ratio))
        real = RealDataset(imgs, args.size, crops_per_image=max(1, n_real // max(1, len(imgs))), seed=2)
        datasets.append(real)
        print(f"real images: {len(imgs)}  real crops/epoch: {len(real)}")
    train_ds = ConcatDataset(datasets)
    val_ds = SynthDataset(args.val_size, args.size, seed=999)

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    vdl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    best = float("inf")
    start_ep = 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best = ck.get("best", best)
        start_ep = ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']}")
    t_start = time.time()
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    step = 0
    done_this_run = 0
    for ep in range(start_ep, args.epochs):
        synth.epoch = ep
        if real is not None:
            real.epoch = ep
        model.train()
        t0 = time.time()
        agg = {}
        for i, (x, heat, dots, dmask, reg, rmask) in enumerate(dl):
            out = model(x)
            loss, parts = brailly_loss(out, heat, dots, dmask, reg, rmask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(dl)} loss {loss.item():.4f} {parts} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (i + 1):.2f}s/it", flush=True)
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                break
        # validation
        model.eval()
        vl, vparts, n = 0.0, {}, 0
        with torch.no_grad():
            for x, heat, dots, dmask, reg, rmask in vdl:
                l, p = brailly_loss(model(x), heat, dots, dmask, reg, rmask)
                vl += l.item()
                n += 1
                for k, v in p.items():
                    vparts[k] = vparts.get(k, 0) + v / len(vdl)
        vl /= max(1, n)
        rec = {"epoch": ep, "val_loss": vl, "val": vparts,
               "train": {k: v / max(1, len(dl)) for k, v in agg.items()}, "minutes": (time.time() - t_start) / 60}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        is_best = vl < best
        best = min(best, vl)
        ck = {"model": model.state_dict(), "epoch": ep, "val_loss": vl, "args": vars(args)}
        if is_best:
            torch.save(ck, os.path.join(args.out, "best.pt"))
        ck.update(opt=opt.state_dict(), sched=sched.state_dict(), best=best)
        torch.save(ck, last_path)
        done_this_run += 1
        if args.epochs_per_run and done_this_run >= args.epochs_per_run:
            break
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            print("time budget reached")
            break


if __name__ == "__main__":
    main()
