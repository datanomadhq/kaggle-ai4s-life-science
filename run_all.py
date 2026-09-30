#!/usr/bin/env python
"""VirtualPaint-Tox entry script: download -> manifest -> preprocess -> train -> evaluate -> downstream -> figures.

    python run_all.py --quick     # ~20 min smoke test on a small subset (6 wells x 8 fields), 2 epochs
    python run_all.py             # full reproduction of the report (HepatoPAC Islands, ~33 GB download)
    python run_all.py --stage eval   # re-run a single stage on existing artefacts

Stages: download, manifest, preprocess, train, eval, downstream, figures. Each stage is skipped when its
outputs already exist unless --force is given. All results are written under results/ (quick: results_quick/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# main model + baselines/ablations evaluated in the report (name -> extra train args)
EXPERIMENTS = {
    "unet": ["--model", "unet"],                                    # ours: U-Net + Laplace uncertainty head
    "unet_nounc": ["--model", "unet_nounc"],                        # ablation: no uncertainty head (plain L1)
    "unet_ssim": ["--model", "unet", "--ssim-weight", "0.5"],       # ablation: + SSIM term
    "smallcnn": ["--model", "smallcnn"],                            # baseline: shallow CNN (25 px receptive field)
    "linear": ["--model", "linear"],                                # baseline: per-pixel affine map
    "unet_frac25": ["--model", "unet", "--train-frac", "0.25"],     # ablation: 25% of training wells
    "unet_frac50": ["--model", "unet", "--train-frac", "0.5"],      # ablation: 50% of training wells
}
QUICK_EXPERIMENTS = ["unet", "linear"]


def run(cmd: list[str]):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps for the main model (ablations get --ablation-steps)")
    ap.add_argument("--ablation-steps", type=int, default=None)
    ap.add_argument("--experiments", default="", help="comma-separated subset of experiments to train")
    a = ap.parse_args()

    data = ROOT / "data" / "hepatopac"
    img = data / "islands"
    res = ROOT / ("results_quick" if a.quick else "results") / "hepatopac"
    manifest = (res / "manifest.csv") if a.quick else (data / "manifest.csv")
    npy = data / ("npy_quick" if a.quick else "npy_ds2")
    steps = a.steps or (40 if a.quick else 5000)
    ablation_steps = a.ablation_steps or (40 if a.quick else 1500)
    stages = ["download", "manifest", "preprocess", "train", "eval", "downstream", "figures"] if a.stage == "all" else a.stage.split(",")
    exps = a.experiments.split(",") if a.experiments else (QUICK_EXPERIMENTS if a.quick else list(EXPERIMENTS))
    quick_wells = "A01,B02,C03,E01,G03,H06"
    quick_sites = ",".join(str(s) for s in range(1, 9))

    if "download" in stages:
        cmd = [PY, "-m", "vptox.download", "--dataset", "hepatopac_islands", "--out", img]
        if a.quick:
            cmd += ["--wells", quick_wells, "--sites", quick_sites]
        run(cmd)
    if "manifest" in stages and (a.force or not manifest.exists()):
        run([PY, "-m", "vptox.manifest", "--load-data", data / "meta" / "hepatopac_islands_load_data.csv",
             "--platemap", data / "meta" / "platemap.txt", "--image-dir", img, "--out", manifest])
    if "preprocess" in stages and (a.force or not (npy / "stats.json").exists()):
        run([PY, "-m", "vptox.preprocess", "--manifest", manifest, "--image-dir", img, "--out-dir", npy,
             "--downsample", "2", "--stats-sample", "40" if a.quick else "200"])
    if "train" in stages:
        for name in exps:
            out = res / name
            if (out / "done.json").exists() and not a.force:
                print(f"skip train {name} (done)")
                continue
            n_steps = steps if name == "unet" else ablation_steps
            if name == "linear":
                n_steps = min(n_steps, 500)
            cmd = [PY, "-m", "vptox.train", "--manifest", manifest, "--npy-dir", npy, "--out-dir", out,
                   "--steps", n_steps] + EXPERIMENTS[name]
            if a.quick:
                cmd += ["--val-every", "20", "--max-val-fields", "8", "--batch", "8"]
            run(cmd)
    if "eval" in stages:
        for name in exps:
            out = res / name
            if not (out / "best.pt").exists():
                continue
            if (out / "summary_test.json").exists() and not a.force:
                continue
            cmd = [PY, "-m", "vptox.predict", "--checkpoint", out / "best.pt", "--manifest", manifest,
                   "--npy-dir", npy, "--out-dir", out, "--split", "test"]
            if name == "unet":
                cmd += ["--save-pred"]
            run(cmd)
    if "downstream" in stages:
        run([PY, "-m", "vptox.downstream", "--manifest", manifest, "--npy-dir", npy, "--pred-dir", res / "unet" / "pred",
             "--out-dir", res / "downstream"] + (["--quick"] if a.quick else []))
    if "figures" in stages:
        run([PY, "-m", "vptox.figures", "--results", res, "--manifest", manifest, "--npy-dir", npy,
             "--out-dir", res / "figures"])
    print("all done ->", res)


if __name__ == "__main__":
    main()
