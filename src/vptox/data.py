"""Datasets: random crops of stacked fields for training, full fields for evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import CHANNELS, INPUT

ORDER = [INPUT] + CHANNELS


def normalise_input(bf: np.ndarray) -> np.ndarray:
    """Per-image robust standardisation of the brightfield image (median / IQR-based)."""
    bf = bf.astype(np.float32)
    med = float(np.median(bf))
    q1, q3 = np.percentile(bf, [25, 75])
    scale = max(float(q3 - q1) / 1.349, 1e-3)
    return ((bf - med) / scale).astype(np.float32)


def normalise_targets(fluo: np.ndarray, stats: dict) -> np.ndarray:
    """Global per-channel percentile scaling to roughly [0, 1] (values above 1 are kept, not clipped)."""
    out = np.empty(fluo.shape, dtype=np.float32)
    for i, c in enumerate(CHANNELS):
        lo, hi = stats[c]["low"], stats[c]["high"]
        out[i] = (fluo[i].astype(np.float32) - lo) / max(hi - lo, 1e-3)
    return np.clip(out, -0.5, 3.0)


class FieldDataset(Dataset):
    """Yields (input[1,h,w], target[5,h,w]) float32 tensors.

    crop=None returns the full field padded to a multiple of ``pad_to``; otherwise a random crop
    with random flips / 90-degree rotations (``augment=True``).
    """

    def __init__(self, manifest: pd.DataFrame, npy_dir: str | Path, stats: dict, crop: int | None = 256,
                 augment: bool = True, pad_to: int = 16, samples_per_field: int = 1,
                 input_channels: tuple[str, ...] = (INPUT,)):
        self.m = manifest.reset_index(drop=True)
        self.npy_dir = Path(npy_dir)
        self.stats = stats
        self.crop = crop
        self.augment = augment
        self.pad_to = pad_to
        self.spf = samples_per_field
        self.in_idx = [ORDER.index(c) for c in input_channels]

    def __len__(self):
        return len(self.m) * self.spf

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.m.iloc[i]
        a = np.load(self.npy_dir / f"{r['field_id']}.npy", mmap_mode="r")
        return a, r

    def __getitem__(self, idx):
        i = idx % len(self.m)
        a, r = self.load(i)
        if self.crop is not None:
            H, W = a.shape[1:]
            y = np.random.randint(0, H - self.crop + 1)
            x = np.random.randint(0, W - self.crop + 1)
            a = np.asarray(a[:, y:y + self.crop, x:x + self.crop])
        else:
            a = np.asarray(a)
        x_in = np.stack([normalise_input(a[j]) for j in self.in_idx])
        y = normalise_targets(a[1:6], self.stats)
        if self.augment:
            if np.random.rand() < 0.5:
                x_in, y = x_in[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x_in, y = x_in[:, ::-1, :], y[:, ::-1, :]
            k = np.random.randint(4)
            x_in, y = np.rot90(x_in, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        if self.crop is None and self.pad_to:
            H, W = y.shape[1:]
            ph, pw = (-H) % self.pad_to, (-W) % self.pad_to
            if ph or pw:
                x_in = np.pad(x_in, ((0, 0), (0, ph), (0, pw)), mode="reflect")
                y = np.pad(y, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        return torch.from_numpy(np.ascontiguousarray(x_in)), torch.from_numpy(np.ascontiguousarray(y)), i


def load_stats(npy_dir: str | Path) -> dict:
    return json.loads((Path(npy_dir) / "stats.json").read_text())
