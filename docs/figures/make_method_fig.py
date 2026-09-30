"""Draw the method schematic (brightfield -> U-Net -> 5 virtual channels + uncertainty -> downstream readouts)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vptox import CHANNELS  # noqa: E402

CMAPS = {"DNA": "Blues", "ER": "Greens", "RNA": "Reds", "AGP": "Purples", "Mito": "Oranges"}


def box(ax, x, y, w, h, text, fc="#e2e8f0", ec="#334155", fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#334155"))


def main(field_npy: str | None = None, out: str = "docs/figures/fig_method.png"):
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.7, 4.6)
    ax.axis("off")
    # input thumbnail
    if field_npy and Path(field_npy).exists():
        a = np.load(field_npy)
        bf = a[0, 100:400, 100:400].astype(float)
        ax.imshow(np.clip((bf - np.percentile(bf, 1)) / (np.percentile(bf, 99) - np.percentile(bf, 1)), 0, 1),
                  cmap="gray", extent=(0.3, 2.1, 1.6, 3.4))
    else:
        box(ax, 0.3, 1.6, 1.8, 1.8, "brightfield\n(1 plane, 20x)", fc="#f8fafc")
    ax.text(1.2, 3.55, "brightfield image (label-free)", ha="center", fontsize=10, fontweight="bold")
    arrow(ax, 2.2, 2.5, 2.9, 2.5)
    # U-Net
    box(ax, 3.0, 1.2, 2.6, 2.6, "U-Net\n4 levels, 32-512 ch\n7.8 M parameters\n\nrandom 256$^2$ crops,\nflips / rotations", fc="#dbeafe", fs=10)
    ax.text(4.3, 3.95, "encoder-decoder with skip connections", ha="center", fontsize=9, color="#475569")
    arrow(ax, 5.7, 2.9, 6.4, 3.3)
    arrow(ax, 5.7, 2.1, 6.4, 1.6)
    # outputs: 5 channel thumbnails
    x0 = 6.5
    for i, c in enumerate(CHANNELS):
        if field_npy and Path(field_npy).exists():
            im = a[1 + i, 100:400, 100:400].astype(float)
            ax.imshow(np.clip(im / np.percentile(im, 99.5), 0, 1), cmap=CMAPS[c], extent=(x0 + i * 0.95, x0 + i * 0.95 + 0.85, 3.0, 3.85))
        else:
            box(ax, x0 + i * 0.95, 3.0, 0.85, 0.85, c, fc="#f1f5f9", fs=8)
        ax.text(x0 + i * 0.95 + 0.42, 3.9, c, ha="center", fontsize=8)
    ax.text(x0 + 2.3, 4.25, r"means $\mu_c$: five virtual Cell Painting channels", ha="center", fontsize=10, fontweight="bold")
    # uncertainty
    if field_npy and Path(field_npy).exists():
        rng = np.random.default_rng(0)
        u = np.abs(np.gradient(bf)[0]) + 0.3 * rng.random(bf.shape)
        ax.imshow(u, cmap="magma", extent=(x0, x0 + 0.85, 0.9, 1.75))
    else:
        box(ax, x0, 0.9, 0.85, 0.85, "b", fc="#fde68a", fs=8)
    ax.text(x0 + 2.6, 1.3, r"scales $b_c$: per-pixel expected error  (Laplace NLL:  $|y-\mu|/b + \log b$)", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(x0 + 2.6, 0.95, "trust map for QC; calibration checked by sparsification", ha="left", va="center", fontsize=9, color="#475569")
    # downstream
    arrow(ax, x0 + 4.8, 3.4, x0 + 5.3, 3.4)
    box(ax, x0 + 5.35, 2.35, 1.1, 1.9, "downstream\nnuclei count\nfeatures\ntreatment effect\ncytotoxicity", fc="#dcfce7", fs=8.5)
    ax.text(x0 + 5.9, 4.4, "same pipeline on real\nand virtual stains", ha="center", fontsize=8, color="#475569")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else "docs/figures/fig_method.png")
