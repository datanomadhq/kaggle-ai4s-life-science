## Category Declaration

**Model & Algorithm**

## Demo Video

https://youtu.be/Y-NtpgwLSFM (3:29, viewable without login; also in the media gallery above)

## Code Repository Link

https://github.com/datanomadhq/kaggle-ai4s-life-science (public, MIT licence). `python run_all.py --quick` runs every stage from a fresh clone in a few minutes; `python run_all.py` reproduces every number in the report. Trained checkpoints are assets of the release `v1.0-checkpoints`.

## Project Summary

Drug-induced liver injury is the leading cause of drug withdrawal, and organ-on-chip / microphysiological systems (MPS) are replacing animals for testing compounds on human tissue. The richest image-based toxicology readout, Cell Painting, stains five cellular compartments, but the stain is terminal: an expensive, long-lived liver chip can be read only once.

VirtualPaint-Tox predicts all five Cell Painting channels (DNA, ER, RNA, actin/Golgi/membrane, mitochondria) from one plain **brightfield** image, which is free, non-toxic and repeatable, together with a **per-pixel uncertainty map** showing where the virtual stain can be trusted. It is a U-Net with a Laplace uncertainty head, trained on public CC0 data from the OASIS Consortium: **HepatoPAC, a primary human hepatocyte liver MPS used industrially for DILI testing** (2,352 fields, 48 wells, five compounds at two doses). Splits are by well.

On 12 held-out wells the virtual channels reach Pearson r = 0.73 (DNA), 0.77 (ER), 0.85 (RNA), 0.73 (AGP) and 0.85 (mitochondria), mean 0.79 versus 0.65 for a shallow CNN. The uncertainty ranks pixel errors (Spearman 0.49-0.63) and whole-field errors (r = 0.95-0.98) and flags imaging artefacts. Biological fidelity holds too: nuclear counts agree at r = 0.71, 67% of 49 morphological features are preserved with r > 0.7, and the ranking of compound effects across treated wells is recovered (Spearman 0.55). On a second public system, U2OS cells with measured MTT/LDH cytotoxicity, virtual-stain features predict the cytotoxicity of 56 unseen compounds with AUC 0.92, equal to the real stain (0.92) and above brightfield texture alone (0.83). A model moves to a new system with 500 fine-tuning steps (r 0.54 to 0.73).

The value: a repeatable, label-free toxicity readout for long-lived chips, with a built-in trust map. Everything runs on a laptop; code, checkpoints and the exact data subsets are public.

## Technical Report

Full report (15 pages, PDF): attached to this Writeup and at https://github.com/datanomadhq/kaggle-ai4s-life-science/blob/main/docs/report/report.pdf

- **Data.** Public, CC0, from the Cell Painting Gallery (OASIS Consortium, cpg0037). HepatoPAC liver MPS: 48 wells x 49 fields = 2,352 fields with brightfield and five Cell Painting dyes, five compounds at two doses plus controls; split by well, 12 held-out test wells (588 fields). Axiom U2OS subset: 689 fields, 186 compounds with per-well MTT and LDH viability; test set of 227 wells from 56 compounds never seen in training.
- **Method.** A 2D U-Net (7.76M parameters) maps one brightfield image to the five stain channels. The mean is trained with L1; a Laplace scale head is trained on detached residuals, so the uncertainty cannot be inflated to hide errors. 256 x 256 crops at 1.19 um per pixel, 5,000 steps on a laptop GPU.
- **Experiments.** Baselines: per-pixel linear map and a shallow CNN. Ablations: L1 only, joint heteroscedastic loss, added SSIM term, 25% and 50% of the training wells, full resolution, a second seed, and the scale head on detached features. Cross-system transfer between the liver MPS and U2OS, zero-shot and after 500 fine-tuning steps.
- **Results.** Mean Pearson r 0.787 on held-out wells (DNA 0.73, ER 0.77, RNA 0.85, AGP 0.73, mitochondria 0.85) against 0.665 for the shallow CNN and 0.08 for the linear map. Predicted uncertainty ranks pixel errors (Spearman 0.49-0.63) and field errors (r 0.95-0.98). Nuclear counts from the virtual DNA channel agree with the real stain at r 0.71; 67% of 49 morphological features are reproduced with r > 0.7. On U2OS, virtual-stain features predict cytotoxicity of unseen compounds with AUC 0.92 (real stain 0.92, brightfield texture 0.83). Transfer: 0.54-0.57 zero-shot, 0.67-0.73 after 500 fine-tuning steps. A later variant with the scale head on detached features reaches 0.817; the downstream analyses use the earlier 0.787 model and are therefore conservative.
- **Reproduction.** One entry script, pinned environment, public data fetched by URL, released checkpoints; no paid services or special hardware. A fresh clone was run end to end on 2026-10-08.
- **Limitations.** One HepatoPAC plate with five anonymised compounds, so treated-versus-control separation on the liver MPS is weak for real and virtual stains alike; DNA is the hardest channel at this resolution; the uncertainty is aleatoric and becomes less reliable under domain shift until the model is fine-tuned; nuclei are segmented with a classical method; images are 2D.

## Optional Demo Link

`python -m vptox.predict_image --checkpoint <checkpoint> --input your_brightfield.tif --out out/` produces the five virtual channels, uncertainty maps and an RGB composite for any 20x brightfield image (instructions in the repository README).

## Team

Dmitriy Kompaneets (team leader; AI/CS). No biology or bioengineering member; no cross-disciplinary bonus claimed.
