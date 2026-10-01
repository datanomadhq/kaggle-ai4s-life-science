#!/usr/bin/env python
"""VirtualPaint-Tox entry script: download -> manifest -> preprocess -> train -> evaluate -> downstream -> figures.

    python run_all.py --quick                 # ~20 min smoke test on a small HepatoPAC subset (6 wells x 8 fields)
    python run_all.py                         # full HepatoPAC reproduction (33 GB download, 7 models)
    python run_all.py --dataset axiom         # Axiom U2OS subset with MTT/LDH cytotoxicity (34 GB download)
    python run_all.py --stage eval,figures    # re-run late stages on existing artefacts

Stages: download, manifest, preprocess, train, eval, downstream, figures (dataset-specific stages are skipped
where they do not apply). A stage is skipped when its outputs exist unless --force is given. Results go to
results/<dataset>/ (quick: results_quick/).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# main model + baselines/ablations evaluated in the report (name -> extra train args)
EXPERIMENTS = {  # order = training order
    "unet": ["--model", "unet"],                                    # ours: U-Net, L1 mean + Laplace scale head on detached residuals
    "linear": ["--model", "linear"],                                # baseline: per-pixel affine map
    "smallcnn": ["--model", "smallcnn"],                            # baseline: shallow CNN (25 px receptive field)
    "unet_nounc": ["--model", "unet_nounc"],                        # ablation: no uncertainty head (plain L1)
    "unet_jointnll": ["--model", "unet", "--nll-mode", "joint"],    # ablation: classic joint heteroscedastic NLL
    "unet_frac25": ["--model", "unet", "--train-frac", "0.25"],     # ablation: 25% of training wells
    "unet_frac50": ["--model", "unet", "--train-frac", "0.5"],      # ablation: 50% of training wells
    "unet_fullres": ["--model", "unet"],                            # ablation: full resolution (0.59 um/px), npy override below
    "unet_ssim": ["--model", "unet", "--ssim-weight", "0.5"],       # ablation: + SSIM term
    "unet_short": ["--model", "unet"],                              # ours at the ablation budget (1,500 steps) for like-for-like comparison
    "unet_short_seed1": ["--model", "unet", "--seed", "1"],         # second seed of the above: run-to-run variance
    "unet_nounc_long": ["--model", "unet_nounc"],                   # L1-only at the full 5,000-step budget: is the uncertainty head free?
    # cross-system fine-tuning (deployment recipe): start from the other system's model, 500 steps on the target system
    "unet_ft_from_hepatopac": ["--model", "unet", "--init-from", "results/hepatopac/unet/best.pt", "--lr", "1e-4"],
    "unet_ft_from_axiom": ["--model", "unet", "--init-from", "results/axiom/unet/best.pt", "--lr", "1e-4"],
}
NPY_OVERRIDE = {"unet_fullres": "npy"}  # experiment -> data/<dataset>/<dir> with differently preprocessed stacks
STEPS_OVERRIDE = {"unet_nounc_long": 5000, "unet_ft_from_hepatopac": 500, "unet_ft_from_axiom": 500}
DATASET_EXPERIMENTS = {"hepatopac": list(EXPERIMENTS), "axiom": ["unet", "linear"]}
QUICK_EXPERIMENTS = ["unet", "linear"]


def run(cmd: list[str]):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hepatopac", choices=["hepatopac", "axiom"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps for the main model (ablations get --ablation-steps)")
    ap.add_argument("--ablation-steps", type=int, default=None)
    ap.add_argument("--experiments", default="", help="comma-separated subset of experiments to train")
    a = ap.parse_args()

    data = ROOT / "data" / a.dataset
    res = ROOT / ("results_quick" if a.quick else "results") / a.dataset
    stages = ["download", "manifest", "preprocess", "train", "eval", "downstream", "figures"] if a.stage == "all" else a.stage.split(",")
    exps = a.experiments.split(",") if a.experiments else (QUICK_EXPERIMENTS if a.quick else DATASET_EXPERIMENTS[a.dataset])
    steps = a.steps or (40 if a.quick else (5000 if a.dataset == "hepatopac" else 3000))
    ablation_steps = a.ablation_steps or (40 if a.quick else 1500)

    if a.dataset == "hepatopac":
        img = data / "islands"
        manifest = (res / "manifest.csv") if a.quick else (data / "manifest.csv")
        npy = data / ("npy_quick" if a.quick else "npy_ds2")
        # 3 wells each of compound 1 low and compound 4 high + 4 controls, 6 fields each: gives train/val/test wells per group
        quick_wells, quick_sites = "A01,B01,D06,A06,B06,G02,C03,C04,C05,F03", ",".join(str(s) for s in range(1, 7))
        if "download" in stages:
            cmd = [PY, "-m", "vptox.download", "--dataset", "hepatopac_islands", "--out", img]
            if a.quick:
                cmd += ["--wells", quick_wells, "--sites", quick_sites]
            run(cmd)
        if "manifest" in stages and (a.force or not manifest.exists()):
            cmd = [PY, "-m", "vptox.manifest", "--load-data", data / "meta" / "hepatopac_islands_load_data.csv",
                   "--platemap", data / "meta" / "platemap.txt", "--image-dir", img, "--out", manifest]
            if a.quick:
                cmd += ["--wells", quick_wells, "--sites", quick_sites]
            run(cmd)
        downsample = "2"
    else:  # axiom
        img = data / "images"
        manifest = data / "manifest.csv"
        npy = data / "npy_ds4"
        if "download" in stages:
            if not (data / "selection.csv").exists():
                # metadata (load_data.csv + biochem/metadata parquet) for batch prod_25, then the well selection
                run(["aws", "s3", "cp", "--no-sign-request", "--quiet", "--recursive", "--exclude", "*", "--include", "*/load_data.csv",
                     "s3://cellpainting-gallery/cpg0037-oasis/axiom/workspace/load_data_csv/prod_25/", data / "meta" / "load_data_prod_25/"])
                run(["aws", "s3", "cp", "--no-sign-request", "--quiet", "--recursive", "--exclude", "*", "--include", "*.parquet",
                     "s3://cellpainting-gallery/cpg0037-oasis/axiom/workspace/metadata/prod_25/", data / "meta" / "metadata_prod_25/"])
                run([PY, "-m", "vptox.axiom_select", "--meta-dir", data / "meta", "--batch", "prod_25", "--out", data])
            run([PY, "-m", "vptox.download", "--keys-file", data / "keys.txt", "--out", img, "--workers", "8"])
        if "manifest" in stages and (a.force or not manifest.exists()):
            run([PY, "-m", "vptox.axiom_manifest", "--selection", data / "selection.csv", "--image-dir", img, "--out", manifest])
        downsample = "4"

    if "preprocess" in stages and (a.force or not (npy / "stats.json").exists()):
        run([PY, "-m", "vptox.preprocess", "--manifest", manifest, "--image-dir", img, "--out-dir", npy,
             "--downsample", downsample, "--stats-sample", "40" if a.quick else "200"])
    if "train" in stages:
        for name in exps:
            out = res / name
            if (out / "done.json").exists() and not a.force:
                print(f"skip train {name} (done)")
                continue
            n_steps = steps if name == "unet" else STEPS_OVERRIDE.get(name, ablation_steps)
            if name == "linear":
                n_steps = min(n_steps, 500)
            npy_e = data / NPY_OVERRIDE[name] if name in NPY_OVERRIDE else npy
            if name in NPY_OVERRIDE and not (npy_e / "stats.json").exists():
                run([PY, "-m", "vptox.preprocess", "--manifest", manifest, "--image-dir", img, "--out-dir", npy_e, "--downsample", "1"])
            cmd = [PY, "-m", "vptox.train", "--manifest", manifest, "--npy-dir", npy_e, "--out-dir", out,
                   "--steps", n_steps] + EXPERIMENTS[name]
            if a.quick:
                cmd += ["--val-every", "20", "--max-val-fields", "8", "--batch", "8"]
            run(cmd)
    if "eval" in stages:
        for name in exps:
            out = res / name
            if not (out / "best.pt").exists() and not a.quick:
                # evaluation without training: fetch the released checkpoint (GitHub release v1.0-checkpoints)
                run([PY, "-m", "vptox.checkpoints", "--dataset", a.dataset, "--experiments", name])
            if not (out / "best.pt").exists():
                continue
            if (out / "summary_test.json").exists() and not a.force:
                continue
            npy_e = data / NPY_OVERRIDE[name] if name in NPY_OVERRIDE else npy
            cmd = [PY, "-m", "vptox.predict", "--checkpoint", out / "best.pt", "--manifest", manifest,
                   "--npy-dir", npy_e, "--out-dir", out, "--split", "test"]
            if name == "unet":
                cmd += ["--save-pred"]
            run(cmd)
        # cross-system zero-shot transfer: the other dataset's model applied to this test set
        other = "axiom" if a.dataset == "hepatopac" else "hepatopac"
        ck = ROOT / ("results_quick" if a.quick else "results") / other / "unet" / "best.pt"
        if ck.exists() and (a.force or not (res / f"transfer_from_{other}" / "summary_test.json").exists()):
            run([PY, "-m", "vptox.predict", "--checkpoint", ck, "--manifest", manifest, "--npy-dir", npy,
                 "--out-dir", res / f"transfer_from_{other}", "--split", "test"])
    if "downstream" in stages:
        if a.dataset == "hepatopac":
            run([PY, "-m", "vptox.downstream", "--manifest", manifest, "--npy-dir", npy, "--pred-dir", res / "unet" / "pred",
                 "--out-dir", res / "downstream"] + (["--quick"] if a.quick else []))
        else:
            run([PY, "-m", "vptox.toxicity", "--manifest", manifest, "--npy-dir", npy, "--pred-dir", res / "unet" / "pred",
                 "--out-dir", res / "downstream"])
    if "figures" in stages:
        run([PY, "-m", "vptox.figures", "--results", res, "--manifest", manifest, "--npy-dir", npy,
             "--out-dir", res / "figures"])
    print("all done ->", res)


if __name__ == "__main__":
    main()
