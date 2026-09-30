"""Figures and tables for the report, generated from results/ artefacts.

- results_table.md / .tex : per-channel Pearson / SSIM for every experiment (mean over test fields, 95% CI over wells)
- fig_examples.png        : brightfield | real vs virtual channels | uncertainty, for a few test fields
- fig_channels.png        : per-channel Pearson bars for all experiments
- fig_calibration.png     : reliability (predicted scale vs |error|) and sparsification summary
- fig_trainsize.png       : Pearson vs fraction of training wells
- fig_downstream.png      : nuclei counts real vs virtual; feature fidelity; well effect concordance
- fig_training.png        : training curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import CHANNELS  # noqa: E402
from .data import normalise_input, normalise_targets  # noqa: E402

LABELS = {"unet": "U-Net + uncertainty (ours)", "unet_nounc": "U-Net, L1 only", "unet_jointnll": "U-Net, joint NLL",
          "unet_ssim": "U-Net + SSIM", "smallcnn": "Shallow CNN", "linear": "Per-pixel linear",
          "unet_frac25": "U-Net, 25% wells", "unet_frac50": "U-Net, 50% wells", "unet_fullres": "U-Net, full resolution",
          "transfer_from_axiom": "Zero-shot from U2OS model", "transfer_from_hepatopac": "Zero-shot from HepatoPAC model"}
COLORS = {"DNA": "#4C72B0", "ER": "#55A868", "RNA": "#C44E52", "AGP": "#8172B2", "Mito": "#CCB974"}
CMAPS = {"DNA": "Blues", "ER": "Greens", "RNA": "Reds", "AGP": "Purples", "Mito": "Oranges"}


def load_summaries(res: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(res.iterdir()):
        s = d / "summary_test.json"
        if s.exists():
            j = json.loads(s.read_text())
            j["experiment"] = d.name
            done = d / "done.json"
            if done.exists():
                j.update({f"train_{k}": v for k, v in json.loads(done.read_text()).items()})
            rows.append(j)
    return pd.DataFrame(rows).set_index("experiment")


def results_table(df: pd.DataFrame, out: Path):
    cols = [f"pcc_{c}" for c in CHANNELS]
    lines_md = ["| Model | " + " | ".join(f"PCC {c}" for c in CHANNELS) + " | PCC mean | SSIM mean | params (M) |",
                "|---|" + "---|" * (len(CHANNELS) + 3)]
    lines_tex = []
    for e, r in df.iterrows():
        pcc = [f"{r[f'pcc_{c}']:.3f} ± {r.get(f'pcc_{c}_ci95', 0):.3f}" for c in CHANNELS]
        mean_pcc = np.mean([r[f"pcc_{c}"] for c in CHANNELS])
        mean_ssim = np.mean([r[f"ssim_{c}"] for c in CHANNELS])
        params = r.get("train_params", np.nan) / 1e6
        lines_md.append(f"| {LABELS.get(e, e)} | " + " | ".join(pcc) + f" | {mean_pcc:.3f} | {mean_ssim:.3f} | {params:.2f} |")
        lines_tex.append(f"{LABELS.get(e, e)} & " + " & ".join(p.replace('±', r'$\pm$') for p in pcc) +
                         f" & {mean_pcc:.3f} & {mean_ssim:.3f} & {params:.2f} \\\\")
    (out / "results_table.md").write_text("\n".join(lines_md) + "\n")
    (out / "results_table.tex").write_text("\n".join(lines_tex) + "\n")


def fig_channels(df: pd.DataFrame, out: Path):
    exps = [e for e in LABELS if e in df.index]
    x = np.arange(len(CHANNELS))
    w = 0.8 / len(exps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, e in enumerate(exps):
        vals = [df.loc[e, f"pcc_{c}"] for c in CHANNELS]
        errs = [df.loc[e].get(f"pcc_{c}_ci95", 0) for c in CHANNELS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, label=LABELS[e], capsize=2)
    ax.set_xticks(x, CHANNELS)
    ax.set_ylabel("Pearson r (test fields)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_channels.png", dpi=200)
    plt.close(fig)


def fig_examples(res: Path, manifest: pd.DataFrame, npy_dir: Path, out: Path, n: int = 3, crop: int = 400):
    pred_dir = res / "unet" / "pred"
    met = pd.read_csv(res / "unet" / "metrics_test.csv")
    stats = json.loads((npy_dir / "stats.json").read_text())
    # pick fields spanning the quality range: best, median, worst by mean PCC
    met["pcc_mean"] = met[[f"pcc_{c}" for c in CHANNELS]].mean(1)
    met = met.sort_values("pcc_mean")
    picks = [met.iloc[-1], met.iloc[len(met) // 2], met.iloc[0]][:n]
    rows = len(picks) * 2
    fig, axes = plt.subplots(rows, 6, figsize=(15, 2.6 * rows))
    for k, r in enumerate(picks):
        raw = np.load(npy_dir / f"{r.field_id}.npy")
        real = normalise_targets(raw[1:6], stats)
        bf = normalise_input(raw[0])
        pv = np.load(pred_dir / f"{r.field_id}.npy").astype(np.float32)
        virt, unc = pv[:5], pv[5:] if pv.shape[0] > 5 else None
        y0, x0 = (raw.shape[1] - crop) // 2, (raw.shape[2] - crop) // 2
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        a_top, a_bot = axes[2 * k], axes[2 * k + 1]
        a_top[0].imshow(bf[sl], cmap="gray", vmin=-3, vmax=3)
        a_top[0].set_title(f"brightfield  {r.well} s{int(r.site)}\n{r.compound} {r.concentration}", fontsize=8)
        if unc is not None:
            a_bot[0].imshow(unc.mean(0)[sl], cmap="magma", vmin=0, vmax=np.percentile(unc.mean(0), 99))
            a_bot[0].set_title("predicted uncertainty (mean scale)", fontsize=8)
        for j, c in enumerate(CHANNELS):
            vmax = max(np.percentile(real[j][sl], 99.5), 0.05)
            a_top[j + 1].imshow(real[j][sl], cmap=CMAPS[c], vmin=0, vmax=vmax)
            a_top[j + 1].set_title(f"real {c}", fontsize=8)
            a_bot[j + 1].imshow(virt[j][sl], cmap=CMAPS[c], vmin=0, vmax=vmax)
            a_bot[j + 1].set_title(f"virtual {c}  r={r[f'pcc_{c}']:.2f}", fontsize=8)
        for ax in list(a_top) + list(a_bot):
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_examples.png", dpi=150)
    plt.close(fig)


def fig_calibration(res: Path, out: Path):
    met = pd.read_csv(res / "unet" / "metrics_test.csv")
    if "spearman_DNA" not in met.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    sp = [met[f"spearman_{c}"].mean() for c in CHANNELS]
    au = [met[f"ause_{c}"].mean() for c in CHANNELS]
    axes[0].bar(CHANNELS, sp, color=[COLORS[c] for c in CHANNELS])
    axes[0].set_ylabel("Spearman(|error|, predicted scale)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Uncertainty ranks errors")
    axes[1].bar(CHANNELS, au, color=[COLORS[c] for c in CHANNELS])
    axes[1].set_ylabel("AUSE (lower is better)")
    axes[1].set_title("Sparsification error")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", dpi=200)
    plt.close(fig)
    # field-level: mean predicted scale vs field MAE
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for j, c in enumerate(CHANNELS):
        axes[j].scatter(met[f"mean_scale_{c}"], met[f"mae_{c}"], s=6, alpha=0.5, color=COLORS[c])
        r = np.corrcoef(met[f"mean_scale_{c}"], met[f"mae_{c}"])[0, 1]
        axes[j].set_title(f"{c}: field-level r={r:.2f}", fontsize=9)
        axes[j].set_xlabel("mean predicted scale")
        axes[j].set_ylabel("field MAE")
    fig.tight_layout()
    fig.savefig(out / "fig_calibration_fields.png", dpi=200)
    plt.close(fig)


def fig_trainsize(df: pd.DataFrame, out: Path):
    pts = [(0.25, "unet_frac25"), (0.5, "unet_frac50"), (1.0, "unet")]
    pts = [(f, e) for f, e in pts if e in df.index]
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for c in CHANNELS:
        ax.plot([f for f, _ in pts], [df.loc[e, f"pcc_{c}"] for _, e in pts], "o-", color=COLORS[c], label=c)
    ax.set_xlabel("fraction of training wells")
    ax.set_ylabel("Pearson r (test)")
    ax.set_xticks([f for f, _ in pts])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_trainsize.png", dpi=200)
    plt.close(fig)


def fig_training(res: Path, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for e in LABELS:
        p = res / e / "log.csv"
        if not p.exists():
            continue
        log = pd.read_csv(p)
        axes[0].plot(log.step, log.train_l1, label=LABELS[e])
        axes[1].plot(log.step, log.val_pcc_mean, label=LABELS[e])
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("train L1")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("val mean Pearson")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_training.png", dpi=200)
    plt.close(fig)


def fig_downstream(res: Path, out: Path):
    d = res / "downstream"
    if not (d / "summary.json").exists():
        return
    nuc = pd.read_csv(d / "nuclei.csv")
    fid = pd.read_csv(d / "feature_fidelity.csv")
    eff = pd.read_csv(d / "well_effect.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axes[0]
    ax.scatter(nuc.n_real, nuc.n_virtual, s=8, alpha=0.5)
    lim = max(nuc.n_real.max(), nuc.n_virtual.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    r = np.corrcoef(nuc.n_real, nuc.n_virtual)[0, 1]
    ax.set_xlabel("nuclei per field (real DNA)")
    ax.set_ylabel("nuclei per field (virtual DNA)")
    ax.set_title(f"Nuclear counts, r = {r:.2f}")
    ax = axes[1]
    fid = fid.sort_values("pearson")
    colors = ["#888888" if not any(f.startswith(c + "_") for c in CHANNELS) else COLORS[f.split("_")[0]] for f in fid.feature]
    ax.barh(range(len(fid)), fid.pearson, color=colors)
    ax.set_yticks(range(len(fid)), fid.feature, fontsize=4)
    ax.set_xlabel("Pearson r (real vs virtual feature, across fields)")
    ax.set_title("Feature fidelity")
    ax.axvline(0.7, color="k", ls="--", lw=0.8)
    ax = axes[2]
    tr = eff[eff.compound != "control"]
    ax.scatter(tr.effect_real, tr.effect_virtual, c=["#C44E52" if c == "high" else "#4C72B0" for c in tr.concentration])
    for _, r_ in tr.iterrows():
        ax.annotate(f"{r_.compound}-{r_.concentration[0]}", (r_.effect_real, r_.effect_virtual), fontsize=7)
    ax.set_xlabel("effect magnitude vs controls (real stain)")
    ax.set_ylabel("effect magnitude (virtual stain)")
    from scipy.stats import spearmanr
    ax.set_title(f"Per-well compound effect, Spearman = {spearmanr(tr.effect_real, tr.effect_virtual).correlation:.2f}")
    fig.tight_layout()
    fig.savefig(out / "fig_downstream.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res, out, npy_dir = Path(a.results), Path(a.out_dir), Path(a.npy_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(a.manifest)
    df = load_summaries(res)
    if len(df):
        results_table(df, out)
        fig_channels(df, out)
        fig_trainsize(df, out)
    if (res / "unet" / "metrics_test.csv").exists():
        fig_examples(res, manifest, npy_dir, out)
        fig_calibration(res, out)
    fig_training(res, out)
    fig_downstream(res, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
