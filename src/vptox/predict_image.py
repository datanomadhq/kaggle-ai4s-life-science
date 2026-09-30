"""Virtual Cell Painting for a single brightfield image (TIFF/PNG), with uncertainty maps and an RGB overlay.

python -m vptox.predict_image --checkpoint results/hepatopac/unet/best.pt --input field.tif --out out/ [--downsample 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch

from . import CHANNELS
from .data import normalise_input
from .predict import load_model, predict_field
from .preprocess import downsample
from .train import get_device


def to_rgb(virt: np.ndarray) -> np.ndarray:
    """Simple 3-colour composite: DNA -> blue, Mito -> red, AGP+ER -> green."""
    def n(x):
        return np.clip(x, 0, 1)
    rgb = np.stack([n(virt[CHANNELS.index("Mito")]), n(0.5 * virt[CHANNELS.index("AGP")] + 0.5 * virt[CHANNELS.index("ER")]),
                    n(virt[CHANNELS.index("DNA")])], -1)
    return (rgb * 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--downsample", type=int, default=2, help="block-average factor applied before inference (model trained at 2x)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(a.device)
    model, _ = load_model(a.checkpoint, device)
    img = tifffile.imread(a.input) if a.input.lower().endswith((".tif", ".tiff")) else np.array(__import__("PIL.Image").Image.open(a.input))
    if img.ndim == 3:
        img = img.mean(-1)
    img = downsample(img.astype(np.uint16), a.downsample)
    H, W = img.shape
    x = normalise_input(img)
    ph, pw = (-H) % 16, (-W) % 16
    x = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
    mu, b = predict_field(model, torch.from_numpy(x)[None, None].to(device), tta=True)
    mu = mu[:, :H, :W]
    for i, c in enumerate(CHANNELS):
        tifffile.imwrite(out / f"virtual_{c}.tif", mu[i].astype(np.float32))
        if b is not None:
            tifffile.imwrite(out / f"uncertainty_{c}.tif", b[i, :H, :W].astype(np.float32))
    from PIL import Image
    Image.fromarray(to_rgb(mu)).save(out / "virtual_rgb.png")
    print(f"wrote {len(CHANNELS)} virtual channels" + (" + uncertainty maps" if b is not None else "") + f" and virtual_rgb.png -> {out}")


if __name__ == "__main__":
    main()
