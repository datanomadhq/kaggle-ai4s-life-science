"""Stack the six TIFFs of each field into one ``(6, H, W)`` uint16 .npy and compute normalisation stats.

Channel order in the stack: Brightfield, DNA, ER, RNA, AGP, Mito (see ``vptox.CHANNELS``).
Normalisation statistics (per-channel low/high percentiles) are computed on *training* fields only
and stored as JSON next to the stacks; they map raw intensities to roughly [0, 1].
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from . import CHANNELS, INPUT

ORDER = [INPUT] + CHANNELS


def stack_field(row: pd.Series, image_dir: Path, out_dir: Path) -> Path:
    out = out_dir / f"{row['field_id']}.npy"
    if out.exists():
        return out
    imgs = [tifffile.imread(image_dir / row[f"file_{c}"]) for c in ORDER]
    arr = np.stack(imgs).astype(np.uint16)
    np.save(out, arr)
    return out


def compute_stats(manifest: pd.DataFrame, npy_dir: Path, n_sample: int = 200, seed: int = 0,
                  low: float = 0.5, high: float = 99.8) -> dict:
    rng = np.random.default_rng(seed)
    train = manifest[manifest["split"] == "train"]
    ids = rng.choice(train["field_id"].values, size=min(n_sample, len(train)), replace=False)
    pix = {c: [] for c in ORDER}
    for fid in ids:
        a = np.load(npy_dir / f"{fid}.npy", mmap_mode="r")
        sub = a[:, ::4, ::4]  # subsample pixels
        for i, c in enumerate(ORDER):
            pix[c].append(np.asarray(sub[i]).ravel())
    stats = {}
    for c in ORDER:
        v = np.concatenate(pix[c]).astype(np.float32)
        lo, hi = np.percentile(v, [low, high])
        stats[c] = {"low": float(lo), "high": float(hi), "mean": float(v.mean()), "std": float(v.std())}
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stats-sample", type=int, default=200)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest)
    image_dir, out_dir = Path(a.image_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(tqdm(ex.map(lambda r: stack_field(r[1], image_dir, out_dir), m.iterrows()), total=len(m), desc="stack"))
    stats = compute_stats(m, out_dir, n_sample=a.stats_sample)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
