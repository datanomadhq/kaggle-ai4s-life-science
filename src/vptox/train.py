"""Train a brightfield -> Cell Painting model under a fixed compute budget (optimizer steps).

Example
-------
python -m vptox.train --manifest data/hepatopac/manifest.csv --npy-dir data/hepatopac/npy_ds2 \
    --out-dir results/hepatopac/unet --model unet --steps 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import CHANNELS
from .data import FieldDataset, load_stats
from .metrics import pearson
from .model import build_model, translation_loss


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def subset_wells(m: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Ablation helper: keep a fraction of the training wells (stratified by treatment group)."""
    if frac >= 1.0:
        return m
    rng = np.random.default_rng(seed)
    keep = []
    for g, wells in m[m.split == "train"].groupby("group")["well"].unique().items():
        wells = list(wells)
        rng.shuffle(wells)
        keep += wells[: max(1, int(round(frac * len(wells))))]
    return m[(m.split != "train") | m.well.isin(keep)]


@torch.no_grad()
def evaluate(model, loader, device, ssim_weight: float, nll: str) -> dict:
    model.eval()
    losses, pcc = [], {c: [] for c in CHANNELS}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        mu, logb = model(x)
        loss, _ = translation_loss(mu, logb, y, ssim_weight=ssim_weight, nll=nll)
        losses.append(loss.item())
        mu_np, y_np = mu.float().cpu().numpy(), y.float().cpu().numpy()
        for b in range(mu_np.shape[0]):
            for i, c in enumerate(CHANNELS):
                pcc[c].append(pearson(mu_np[b, i], y_np[b, i]))
    model.train()
    out = {"val_loss": float(np.mean(losses))}
    for c in CHANNELS:
        out[f"val_pcc_{c}"] = float(np.mean(pcc[c]))
    out["val_pcc_mean"] = float(np.mean([out[f"val_pcc_{c}"] for c in CHANNELS]))
    return out


def infinite(loader):
    while True:
        yield from loader


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="unet", choices=["unet", "unet_nounc", "linear", "smallcnn"])
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000, help="total optimizer steps (compute budget)")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ssim-weight", type=float, default=0.0)
    ap.add_argument("--nll-mode", default="detached", choices=["detached", "joint", "none"],
                    help="uncertainty training: L1 mean + NLL on detached residuals (default), joint Laplace NLL, or plain L1")
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training wells to use (ablation)")
    ap.add_argument("--max-val-fields", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--init-from", default="", help="checkpoint to initialise the weights from (fine-tuning across systems)")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = get_device(a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    m = pd.read_csv(a.manifest)
    m = subset_wells(m, a.train_frac, a.seed)
    stats = load_stats(a.npy_dir)
    train_ds = FieldDataset(m[m.split == "train"], a.npy_dir, stats, crop=a.crop, augment=True, samples_per_field=4)
    val_m = m[m.split == "val"]
    if len(val_m) == 0:  # tiny smoke-test subsets may have no validation well: fall back to a few training fields
        print("WARNING: no validation wells in manifest; validating on training fields (smoke test only)")
        val_m = m[m.split == "train"].iloc[:8]
    if len(val_m) > a.max_val_fields:  # a fixed, evenly spaced subset keeps validation cheap
        val_m = val_m.iloc[np.linspace(0, len(val_m) - 1, a.max_val_fields).astype(int)]
    val_ds = FieldDataset(val_m, a.npy_dir, stats, crop=None, augment=False)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(a.workers, 2))

    model = build_model(a.model, in_ch=1, out_ch=len(CHANNELS), base=a.base, depth=a.depth).to(device)
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {a.init_from} (step {ck.get('step')})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={a.model} params={n_params/1e6:.2f}M device={device} train_fields={len(train_ds.m)} "
          f"train_wells={train_ds.m.well.nunique()} val_fields={len(val_ds)} steps={a.steps}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    warm = min(200, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    nll = a.nll_mode

    best, t0, step = -1.0, time.time(), 0
    tl, tp = [], []
    model.train()
    with open(out / "log.csv", "w", newline="") as f:
        writer = None
        for x, y, _ in infinite(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            mu, logb = model(x)
            loss, parts = translation_loss(mu, logb, y, ssim_weight=a.ssim_weight, nll=nll)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            tl.append(loss.item())
            tp.append(parts["l1"])
            budget_hit = a.time_budget_min and (time.time() - t0) / 60 > a.time_budget_min
            if step % a.val_every == 0 or step == a.steps or budget_hit:
                row = {"step": step, "train_loss": float(np.mean(tl)), "train_l1": float(np.mean(tp)),
                       "minutes": (time.time() - t0) / 60, "lr": opt.param_groups[0]["lr"]}
                row.update(evaluate(model, val_dl, device, a.ssim_weight, nll))
                tl, tp = [], []
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                print(f"step {step:5d} train_loss {row['train_loss']:.4f} l1 {row['train_l1']:.4f} "
                      f"val_loss {row['val_loss']:.4f} val_pcc {row['val_pcc_mean']:.3f} "
                      f"({', '.join(f'{c}={row[f'val_pcc_{c}']:.2f}' for c in CHANNELS)}) {row['minutes']:.1f} min", flush=True)
                ck = {"model": model.state_dict(), "args": vars(a), "step": step, "stats": stats}
                torch.save(ck, out / "last.pt")
                if row["val_pcc_mean"] > best:
                    best = row["val_pcc_mean"]
                    torch.save(ck, out / "best.pt")
            if step >= a.steps or budget_hit:
                break
    (out / "done.json").write_text(json.dumps({"best_val_pcc_mean": best, "minutes": (time.time() - t0) / 60,
                                               "params": n_params, "steps": step}))


if __name__ == "__main__":
    main()
