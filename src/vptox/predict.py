"""Predict virtual Cell Painting channels (and uncertainty) for the fields of a split, and score them.

Writes per-field metrics (CSV), aggregated metrics (JSON), optional float16 prediction stacks
(``pred/<field_id>.npy`` with shape [10, H, W] = 5 means + 5 Laplace scales) and figure panels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import CHANNELS
from .data import FieldDataset
from .metrics import calibration, field_metrics
from .model import build_model
from .train import get_device


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_model(a["model"], in_ch=1, out_ch=len(CHANNELS), base=a["base"], depth=a["depth"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_field(model, x: torch.Tensor, tta: bool = True):
    """x: [1, 1, H, W]. Returns mu [5,H,W], scale [5,H,W] or None (numpy, float32)."""
    flips = [(False, False), (True, False), (False, True), (True, True)] if tta else [(False, False)]
    mus, bs = [], []
    for fh, fv in flips:
        xi = x
        if fh:
            xi = xi.flip(-1)
        if fv:
            xi = xi.flip(-2)
        mu, logb = model(xi)
        if fh:
            mu = mu.flip(-1)
            logb = logb.flip(-1) if logb is not None else None
        if fv:
            mu = mu.flip(-2)
            logb = logb.flip(-2) if logb is not None else None
        mus.append(mu)
        if logb is not None:
            bs.append(torch.exp(logb))
    mu = torch.stack(mus).mean(0)[0].float().cpu().numpy()
    b = torch.stack(bs).mean(0)[0].float().cpu().numpy() if bs else None
    return mu, b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--max-fields", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = get_device(a.device)
    out = Path(a.out_dir)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    model, ck = load_model(a.checkpoint, device)
    stats = ck["stats"]
    m = pd.read_csv(a.manifest)
    m = m[m.split == a.split].reset_index(drop=True) if a.split != "all" else m
    if a.max_fields:
        m = m.iloc[np.linspace(0, len(m) - 1, min(a.max_fields, len(m))).astype(int)].reset_index(drop=True)
    ds = FieldDataset(m, a.npy_dir, stats, crop=None, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    rows = []
    for x, y, i in tqdm(dl, desc=f"predict {a.split}"):
        r = m.iloc[int(i)]
        mu, b = predict_field(model, x.to(device), tta=not a.no_tta)
        H, W = ds.load(int(i))[0].shape[1:]  # original field size: remove reflect padding
        mu, y_np = mu[:, :H, :W], y[0].numpy()[:, :H, :W]
        row = {"field_id": r.field_id, "well": r.well, "site": r.site, "group": r.group, "compound": r.compound,
               "concentration": r.concentration}
        row.update(field_metrics(mu, y_np, CHANNELS))
        if b is not None:
            b = b[:, :H, :W]
            cal = calibration(mu, b, y_np, CHANNELS)
            row.update({k: v for k, v in cal.items() if not k.startswith("rel_")})
            row.update({f"mean_scale_{c}": float(b[j].mean()) for j, c in enumerate(CHANNELS)})
        rows.append(row)
        if a.save_pred:
            arr = mu if b is None else np.concatenate([mu, b])
            np.save(out / "pred" / f"{r.field_id}.npy", arr.astype(np.float16))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"metrics_{a.split}.csv", index=False)
    num = df.select_dtypes("number").drop(columns=["site"])
    agg = {"n_fields": len(df), "n_wells": int(df.well.nunique())}
    agg.update({k: float(v) for k, v in num.mean().items()})
    agg.update({f"{k}_sd_well": float(v) for k, v in df.groupby("well")[num.columns].mean().std().items()})
    for c in CHANNELS:
        agg[f"pcc_{c}_ci95"] = float(1.96 * df.groupby("well")[f"pcc_{c}"].mean().std() / np.sqrt(df.well.nunique()))
    (out / f"summary_{a.split}.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()
                      if k.startswith(("pcc_", "ssim_", "n_")) and not k.endswith(("_sd_well", "_ci95"))}, indent=1))


if __name__ == "__main__":
    main()
