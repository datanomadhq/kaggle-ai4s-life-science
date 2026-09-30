"""Pixel-level metrics per channel and uncertainty calibration."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def field_metrics(pred: np.ndarray, target: np.ndarray, channels: list[str]) -> dict:
    """pred/target: [C, H, W] in normalised units (targets ~[0,1])."""
    out = {}
    for i, c in enumerate(channels):
        p, t = pred[i], target[i]
        out[f"pcc_{c}"] = pearson(p, t)
        out[f"mae_{c}"] = float(np.abs(p - t).mean())
        tc, pc = np.clip(t, 0, 1), np.clip(p, 0, 1)
        out[f"ssim_{c}"] = float(structural_similarity(pc, tc, data_range=1.0))
        mse = float(((pc - tc) ** 2).mean())
        out[f"psnr_{c}"] = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    return out


def calibration(pred: np.ndarray, scale: np.ndarray, target: np.ndarray, channels: list[str], n_bins: int = 10) -> dict:
    """Uncertainty quality: Spearman(|err|, b) and the area under the sparsification error curve (AUSE).

    AUSE compares the MAE obtained when removing pixels in order of predicted uncertainty with the
    oracle ordering by true error; 0 is perfect, larger is worse. Also returns per-bin mean |err|
    vs mean predicted scale for reliability plots.
    """
    from scipy.stats import spearmanr
    out = {}
    rng = np.random.default_rng(0)
    for i, c in enumerate(channels):
        e = np.abs(pred[i] - target[i]).ravel()
        b = scale[i].ravel()
        idx = rng.choice(e.size, size=min(200_000, e.size), replace=False)
        e, b = e[idx], b[idx]
        out[f"spearman_{c}"] = float(spearmanr(e, b).correlation)
        # sparsification
        order_u = np.argsort(-b)
        order_o = np.argsort(-e)
        fracs = np.linspace(0, 0.99, 50)
        curve_u, curve_o = [], []
        for f in fracs:
            k = int(f * e.size)
            curve_u.append(e[order_u[k:]].mean())
            curve_o.append(e[order_o[k:]].mean())
        curve_u, curve_o = np.array(curve_u) / e.mean(), np.array(curve_o) / e.mean()
        out[f"ause_{c}"] = float(np.trapezoid(curve_u - curve_o, fracs))
        # reliability bins
        qs = np.quantile(b, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(qs, b, side="right") - 1, 0, n_bins - 1)
        out[f"rel_{c}"] = [(float(b[bins == j].mean()), float(e[bins == j].mean())) for j in range(n_bins) if (bins == j).any()]
    return out
