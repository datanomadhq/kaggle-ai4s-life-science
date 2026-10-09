# Kaggle Writeup (draft; paste into the Kaggle Writeup editor)

**Submission category: Model & Algorithm**

## Demo video

https://youtu.be/Y-NtpgwLSFM (3:29, unlisted, viewable without login; built with `docs/video/build_video.py`)

## Code repository

https://github.com/datanomadhq/kaggle-ai4s-life-science (public; MIT; `python run_all.py --quick` reproduces the pipeline in ~20 min, `python run_all.py` reproduces every number in the report)

## Project summary (200-300 words)

Drug-induced liver injury is the leading cause of drug withdrawal, and organ-on-chip / microphysiological systems (MPS) are replacing animals for testing compounds on human tissue. The richest image-based toxicology readout, Cell Painting, stains five cellular compartments, but the stain is terminal: an expensive, long-lived liver chip can be read only once.

VirtualPaint-Tox predicts all five Cell Painting channels (DNA, ER, RNA, actin/Golgi/membrane, mitochondria) from one plain **brightfield** image, which is free, non-toxic and repeatable, together with a **per-pixel uncertainty map** showing where the virtual stain can be trusted. It is a U-Net with a Laplace uncertainty head, trained on public CC0 data from the OASIS Consortium: **HepatoPAC, a primary human hepatocyte liver MPS used industrially for DILI testing** (2,352 fields, 48 wells, five compounds at two doses). Splits are by well.

On 12 held-out wells the virtual channels reach Pearson r = 0.73 (DNA), 0.77 (ER), 0.85 (RNA), 0.73 (AGP) and 0.85 (mitochondria), mean 0.79 versus 0.65 for a shallow CNN. The uncertainty ranks pixel errors (Spearman 0.49-0.63) and whole-field errors (r = 0.95-0.98) and flags imaging artefacts. Biological fidelity holds too: nuclear counts agree at r = 0.71, 67% of 49 morphological features are preserved with r > 0.7, and the ranking of compound effects across treated wells is recovered (Spearman 0.55). On a second public system, U2OS cells with measured MTT/LDH cytotoxicity, virtual-stain features predict the cytotoxicity of 56 unseen compounds with AUC 0.92, equal to the real stain (0.92) and above brightfield texture alone (0.83). A model moves to a new system with 500 fine-tuning steps (r 0.54 to 0.73).

The value: a repeatable, label-free toxicity readout for long-lived chips, with a built-in trust map. Everything runs on a laptop; code, checkpoints and the exact data subsets are public.

## Technical report

https://github.com/datanomadhq/kaggle-ai4s-life-science/blob/main/docs/report/report.pdf (direct download: https://github.com/datanomadhq/kaggle-ai4s-life-science/releases/download/v1.0-checkpoints/report.pdf) — also summarised below.

## Optional demo

`python -m vptox.predict_image --checkpoint ... --input your_brightfield.tif --out out/` produces virtual channels, uncertainty maps and an RGB composite for any 20x brightfield image.

## Team

Dmitriy Kompaneets (team leader; AI/CS). No biology/bioengineering member (no cross-disciplinary bonus claimed).
