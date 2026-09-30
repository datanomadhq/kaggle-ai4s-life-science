# Kaggle Writeup (draft; paste into the Kaggle Writeup editor)

**Submission category: Model & Algorithm**

## Demo video

`<YouTube unlisted link, viewable without login>` (max 5 min; built with `docs/video/build_video.py`)

## Code repository

https://github.com/datanomadhq/kaggle-ai4s-life-science (public; MIT; `python run_all.py --quick` reproduces the pipeline in ~20 min, `python run_all.py` reproduces every number in the report)

## Project summary (200-300 words)

Drug-induced liver injury is the leading cause of drug withdrawal, and organ-on-chip / microphysiological systems (MPS) are being adopted to test compounds on human tissue instead of animals. The richest image-based toxicology readout, Cell Painting, stains five cellular compartments with fluorescent dyes, but the stain is terminal: an expensive, long-lived liver chip can be read only once, and the fluorescence channels are consumed.

VirtualPaint-Tox predicts all five Cell Painting channels (DNA, ER, RNA, actin/Golgi/membrane, mitochondria) from a single plain **brightfield** image, which is free, non-toxic and repeatable, and it reports a **per-pixel uncertainty map** that tells the user where the virtual stain can be trusted. The model is a U-Net with a heteroscedastic Laplace head trained on the public OASIS Consortium data (Cell Painting Gallery, CC0): a **HepatoPAC primary human hepatocyte micropatterned co-culture, a liver MPS used industrially for DILI testing**, with 2,352 fields from 48 wells treated with five compounds at two doses plus vehicle controls. Splits are by well, so every test image comes from a well never seen in training.

Beyond pixel metrics (Pearson `<x>` for DNA to `<y>` for AGP on held-out wells; calibrated uncertainty, Spearman `<z>` between predicted scale and true error), we validate *biological* fidelity: nuclear counts from the virtual DNA channel match the real ones (`r = <>`), `<>`% of interpretable morphological features are preserved, and treated wells separate from controls and rank by effect magnitude equally well with virtual and real features, far above brightfield texture alone. On a second public system, U2OS cells with per-well MTT/LDH cytotoxicity for 150+ compounds, label-free virtual-stain features predict cytotoxicity with AUC `<>` versus `<>` for the real stain.

Everything runs on a laptop; code, checkpoints and the exact data subset are public.

## Technical report

`<public PDF link (GitHub docs/report/report.pdf)>` — also summarised below.

## Optional demo

`python -m vptox.predict_image --checkpoint ... --input your_brightfield.tif --out out/` produces virtual channels, uncertainty maps and an RGB composite for any 20x brightfield image.

## Team

Dmitriy Kompaneets (team leader; AI/CS). No biology/bioengineering member (no cross-disciplinary bonus claimed).
