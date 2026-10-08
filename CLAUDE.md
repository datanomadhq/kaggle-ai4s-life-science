# Kaggle: AI4S Open Innovation: AI for Life Science

Hackathon repo for https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien (a track of the 5th Pazhou Algorithm Competition). **Final deadline 2026-10-10 15:59 UTC (11:59 EDT).** Prize: $22,200 per the Kaggle listing (the pages state no amount). Team size up to 5. Competition pages in `reference/competition_pages.txt`.

## Deadline first

At the start of every session, check the clock against the deadline and state the time remaining. Nothing can be submitted after it.

## What is judged

A Kaggle Writeup containing a demo video (max 5 minutes, viewable without login), a public code repo (reproducible without paid services, proprietary hardware or non-public data; README, environment files, entry script) and a technical report (about 15-20 pages). One category: Model & Algorithm, Tool & Platform, or End-to-End System. Criteria: Problem Importance & Impact 30%, Technical Approach & Innovation 30%, Results & Validation 20%, Reproducibility 10%, Presentation (video) 10%. No leaderboard. The category must be declared at the top of the Writeup. "Solutions validated on real OoC experimental data are strongly encouraged."

## Steps only Dmitriy can do

- Fill in the organisers' external Google registration form (link in `reference/competition_pages.txt`, "PAGE abstract"), or the entry is not eligible for awards.
- Approve making this repo public (the rules require a public repo); it stays private until then.
- The Writeup is submitted on the Kaggle website; prepare everything, then hand over that final step.
- Upload the demo video (YouTube unlisted or Kaggle attachment) and put the link in the Writeup.

## Project decision (made 2026-09-30)

**Title:** *VirtualPaint-Tox: label-free virtual Cell Painting with pixel-wise uncertainty for non-destructive toxicity readouts in liver microphysiological systems.*
**Category:** Model & Algorithm.

**Problem.** Cell Painting (5 fluorescent dyes) is the workhorse morphological assay for toxicity screening, but staining is terminal, costly and impossible to repeat on the same tissue. Organ-on-chip / microphysiological systems (MPS) are expensive and long-lived, so labs want *non-destructive, longitudinal* readouts. We learn to predict the five Cell Painting channels (DNA, ER, RNA/nucleoli, AGP, Mito) from a plain **brightfield** image, together with a **per-pixel uncertainty map** that tells the user where the virtual stain can be trusted, and we test whether the *virtual* stain carries the same toxicological information as the real one (compound-effect detection, nuclear counts, cytotoxicity vs biochemical MTT/LDH).

**Data (all public, CC0 1.0, Cell Painting Gallery on AWS Open Data, bucket `cellpainting-gallery`, no credentials):**
1. `cpg0037-oasis/hepatopac` (OASIS Consortium / HESI + Broad, 2026): **HepatoPAC** micropatterned primary-human-hepatocyte / stromal co-culture, a liver MPS used industrially for DILI testing. Opera Phenix 20x, 1080x1080, 6 channels = brightfield + 5 Cell Painting dyes. Islands scan: 48 wells x 49 fields (2,352 fields, ~33 GB); 5 anonymised compounds at low/high dose + 8 negative-control wells (platemap in `hepatopac/workspace/metadata`). **Primary dataset: real MPS data.**
2. `cpg0037-oasis/axiom` (Axiom Bio, U2OS, Operetta): brightfield + 5 dyes, 1,085 compounds x 8 doses x 2 replicates, with per-well **MTT and LDH cytotoxicity** in `biochem.parquet`. Images are ~8.9 MB each (34,555 files / 308 GB per plate) so we download a targeted well subset (cytotoxic wells + matched non-toxic + DMSO, 2 fields/well). **Toxicity ground truth.**
3. Stretch: `cpg0038-tegtmeyer-neuropainting` (iPSC-derived neurons, brightfield + 5 dyes, 2160x2160): zero-shot / fine-tune transfer to neural cultures (sponsor CellShells works on neural organ-on-chip).
Not used: `cpg0037-oasis/xellar` (organ-chip Cell Painting, no brightfield) and `insphero` (3D microtissue z-stacks, 440 GB, no brightfield); cited as future work.

**Method.** 2D U-Net (brightfield -> 5 channels) trained on 256x256 crops with L1 + multi-scale SSIM; a heteroscedastic (Laplace) head outputs per-pixel scale so the model reports its own uncertainty; test-time flips for a cheap ensemble. Splits are by **well** (and held-out compounds) so that no field of a test well is seen in training. Baselines: per-pixel linear regression, small CNN, U-Net without uncertainty. Ablations: loss, training-set size, input normalisation, per-channel difficulty, uncertainty calibration. Downstream validation: (a) pixel metrics per channel (Pearson, SSIM, PSNR); (b) nuclear segmentation agreement (Cellpose on real vs virtual DNA); (c) simple morphological profiles from real vs virtual channels -> can the virtual stain separate treated from control wells and low from high dose? (d) on Axiom: predict MTT viability from virtual-stain features vs real-stain features vs brightfield-only.

**Why this wins on the rubric.** Impact (30%): toxicity + 3Rs + MPS monitoring, exactly the "Bright-field to Fluorescence" and "Drug Toxicity" impact areas, validated on real MPS data. Innovation (30%): uncertainty-aware virtual staining evaluated for *downstream toxicological fidelity*, not just pixels; first on a liver MPS. Results (20%): baselines, ablations, calibration, biochemical ground truth. Reproducibility (10%): one entry script, pinned env, public data by URL, tiny subset mode, saved checkpoints. Presentation (10%): image-to-image results are visually compelling.

**Machine budget.** Apple Silicon (shared): cap at 4 CPU threads (`OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`), ~16 GB RAM, PyTorch MPS for training. Data lives in `data/` (gitignored). Virtualenv `.venv` via `uv venv --python 3.12`; pinned in `requirements.txt`.

## Day-by-day plan (UTC)

- **Sep 30 (eve)**: decision, CLAUDE.md, download HepatoPAC Islands, scaffold code (manifest, preprocessing, U-Net, train, eval).
- **Oct 1**: preprocess to npy; train baseline U-Net on HepatoPAC (well split); pixel metrics; first figures. Start Axiom subset download.
- **Oct 2**: uncertainty head + calibration; ablations (loss, train size); Cellpose downstream eval; treatment-effect eval on HepatoPAC.
- **Oct 3**: Axiom: train/evaluate; MTT viability prediction from virtual vs real stain. Report skeleton (LaTeX via tectonic).
- **Oct 4**: NeuroPainting transfer (stretch) or more ablations; finalise result tables; README + entry script `run_all.py` end-to-end test in `--quick` mode.
- **Oct 5-6**: technical report writing (15-20 pages, figures), Writeup text (200-300 word summary).
- **Oct 7**: demo video (slides -> PNG -> ffmpeg + `say` narration), storyboard, repo polish, licence/attribution section.
- **Oct 8**: full clean reproduction from scratch in a fresh venv; fix; freeze checkpoints (GitHub release or repo `models/`).
- **Oct 9**: buffer, handover, Dmitriy: registration form, make repo public, upload video, submit Writeup.
- **Oct 10 15:59 UTC**: deadline. Do not start anything new after Oct 9.

## Status (updated 2026-09-30 22:40 UTC)

- Data downloaded to `data/` (gitignored): HepatoPAC Islands 14,112 TIFFs (33 GB) + stacks `npy` (full res, 31 GB) and `npy_ds2` (2x, 7.7 GB); Axiom subset 4,134 TIFFs (33 GB) + `npy_ds4` (689 fields). Manifests with splits committed in `meta/`.
- Code complete for all stages; smoke-tested end to end (`vptox.train/predict/downstream/toxicity/figures`).
- Overnight GPU queue relaunched 2026-09-30 23:14 UTC with the detached-residual loss (`scripts/run_queue.sh`, master log `data/logs/queue_master.log`): HepatoPAC `unet` 5000 steps then 8 ablations/baselines at 1500 steps (`queue_hepatopac.log`, ETA ~11:00 UTC Oct 1) -> Axiom `unet` 3000 steps + linear + eval (`queue_axiom.log`, ETA ~13:30 UTC) -> HepatoPAC eval again for the zero-shot transfer. Then run `python run_all.py --stage downstream,figures` and `python run_all.py --dataset axiom --stage downstream,figures`.
- Main HepatoPAC model (`results/hepatopac/unet`, detached loss, 5000 steps, 168 min): best val mean Pearson 0.776 at step 4500. **Test (12 wells, 588 fields): Pearson DNA 0.729, ER 0.774, RNA 0.847, AGP 0.732, Mito 0.852 (mean 0.787); SSIM 0.67-0.76; calibration Spearman 0.49-0.63, AUSE 0.15-0.23, field-level r(scale, MAE) 0.95-0.98.** Downstream (test / test+val): nuclei r 0.71 / 0.71, F1 0.56 / 0.55; feature median r 0.81 / 0.78 (67% > 0.7); Mantel 0.46; effect-magnitude Spearman 0.55 / 0.60; dose pairs 3/5 / 4/5; treated-vs-control AUC real/virtual/BF 0.61/0.53/0.51 / 0.71/0.62/0.57. Figures in `results/hepatopac/figures/` (committed), tables in the report.
- All HepatoPAC baselines/ablations trained and evaluated (test mean Pearson): ours 5k 0.787; L1-only 0.718; joint NLL 0.698; +SSIM 0.680; 25%/50% wells 0.654/0.668; full-res 0.638; shallow CNN 0.665; per-pixel 0.077. Table/figures in `results/hepatopac/figures/`, narrative in the report. `unet_short` (ours at 1500 steps) queued last (`data/logs/queue_short.log`).
- **Axiom U2OS (`results/axiom/unet`, 3000 steps, 44 min): test 227 wells / 56 unseen compounds: Pearson DNA 0.81, ER 0.77, RNA 0.81, AGP 0.74, Mito 0.69 (mean 0.76). Label-free cytotoxicity (compound-grouped CV): AUC real 0.924 / virtual 0.917 / brightfield 0.825; MTT Spearman 0.61 / 0.62 / 0.41.** `results/axiom/downstream/` (summary.json, fig_toxicity.png, fig_dose_response.png) committed; report section written.
- Pending from the queue: axiom linear eval, `transfer_from_hepatopac` (axiom test), `transfer_from_axiom` (hepatopac test), `unet_short`. Then: fill transfer + unet_short numbers in the report, regenerate figures (`python run_all.py --stage figures`, both datasets), build the video (`python docs/video/build_video.py`), upload checkpoints (`scripts/make_release.sh`).
- Lesson from the first (joint-NLL) attempt: val Pearson peaked at step 1000 (mean 0.674; DNA 0.43, ER 0.70, RNA 0.79, AGP 0.68, Mito 0.78) then decayed (0.607 at 1500) while train L1 kept falling: joint heteroscedastic NLL lets the mean give up on hard pixels. The default is now L1 for the mean + NLL on detached residuals for the scale; joint NLL kept as ablation `unet_jointnll`. DNA is the hard channel at 1.19 um/px; `unet_fullres` ablation tests resolution.
- Report skeleton `docs/report/report.tex` compiles with tectonic (`tectonic -X compile report.tex`); Writeup draft `docs/writeup.md`; video script `docs/video/script.md`, builder `docs/video/build_video.py` (needs `slides.json`).

## Handoff (written 2026-10-01 ~08:10 UTC; deadline 2026-10-10 15:59 UTC)

### Deliverable status
| Deliverable | Status | Where |
|---|---|---|
| Code repo | complete, reproducible (`python run_all.py --quick` 13 min; full runs reproduce every number); checkpoints on GitHub release `v1.0-checkpoints` | this repo (still **private**) |
| Technical report | complete draft, 15 pages, compiles with `cd docs/report && tectonic -X compile report.tex`; one `\todo{unet\_short}` number pending | `docs/report/report.tex` -> `report.pdf` (PDF not committed yet; commit the final one once) |
| Demo video | built, 3:34, slides + `say` narration, all figures real | `docs/video/demo.mp4` (gitignored; attach to the release and upload to YouTube unlisted) |
| Writeup text | drafted with final numbers | `docs/writeup.md` |
| Results | all experiments done except `unet_short` (ours at 1,500 steps; queued, `data/logs/queue_short.log`) | `results/` (summaries, metrics, figures committed; `pred/` and `*.pt` ignored) |

### Remaining work (by day)
- **Oct 1**: `unet_short` done (test 0.684). Queued extra controls (`data/logs/queue_extra2.log`, then `queue_ft.log`): `unet_short_seed1` (seed variance), `unet_nounc_long` (L1-only, 5k steps), `axiom/unet_ft_from_hepatopac` and `hepatopac/unet_ft_from_axiom` (500-step cross-system fine-tuning). When their `summary_test.json` exist: fill the two `\todo{...}` in the ablation paragraph of `docs/report/report.tex` (`grep -n todo`), add the fine-tuning rows to the transfer table (Table `tab:transfer`) and a sentence, `python run_all.py --stage figures` (and `--dataset axiom --stage figures`), recompile, commit the result JSON/CSV files (not `pred/`, not `*.pt`). Then `zsh scripts/make_release.sh` (adds new checkpoints) and `gh release upload v1.0-checkpoints docs/video/demo.mp4 docs/report/report.pdf --clobber`. Commit `docs/report/report.pdf` once (11 MB).
- **Oct 2-3**: proofread the report (check figures 1-10 placement, the `\todo` macro must not remain anywhere: `grep -n todo docs/report/report.tex`), optionally add a one-page "Related work" paragraph; re-run `python run_all.py --quick` in a fresh clone to confirm the README instructions; polish README.
- **Oct 4-8**: buffer. Possible extras if time: Cellpose instead of classical segmentation; fine-tuning experiment (HepatoPAC model -> U2OS with 100 wells); NeuroPainting (cpg0038, neurons with brightfield) zero-shot demo for the sponsor's neural focus.
- **Oct 9 (latest)**: Dmitriy's steps below. Do not start new experiments after Oct 8.

### Done on 2026-10-08 (no further engineering work is pending)
- Fresh-clone test of the README instructions: `gh repo clone` -> `uv venv` -> install -> `python run_all.py --quick` ran every stage (own 1 GB download, 360 TIFFs) and exited 0 in 3 minutes.
- GitHub release `v1.0-checkpoints` now holds 19 assets: all 17 checkpoints (incl. `unet_detfeat`, `unet_nounc_long`, both fine-tuned models, seed 1), `demo.mp4` and `report.pdf`.
- Report has no `\\todo` left, 15 pages, pushed. Headline model stays the frozen 0.787 model in report, video and writeup; the 0.817 detached-features variant is reported as a later control.

### Steps only Dmitriy can do
1. Fill in the organisers' registration form (link in `reference/competition_pages.txt`, "PAGE abstract"); without it the entry is not evaluated.
2. Make the repo public: `gh repo edit datanomadhq/kaggle-ai4s-life-science --visibility public --accept-visibility-change-consequences` (then `python -m vptox.checkpoints` works without auth; verify once: `curl -sI https://github.com/datanomadhq/kaggle-ai4s-life-science/releases/download/v1.0-checkpoints/hepatopac__unet__best.pt | head -1` should be a 302).
3. Upload `docs/video/demo.mp4` to YouTube (unlisted is fine; must be viewable without login) or attach it to the Kaggle Writeup.
4. Create the Kaggle Writeup from `docs/writeup.md`: category declaration first line, video link, repo link, 200-300-word summary, report link (GitHub `docs/report/report.pdf` raw URL or the release asset), optional demo note. Submit before 2026-10-10 15:59 UTC.
5. Confirm the author name/affiliation on the report title page (`docs/report/report.tex`, `\author`) and in `docs/writeup.md`.

### Exact commands
```
cd /Users/dmitriy/Developer/Kaggle/kaggle-ai4s-life-science && source .venv/bin/activate
python run_all.py --quick                                   # smoke test, ~13 min
python run_all.py --stage figures                           # HepatoPAC table + figures from results/
python run_all.py --dataset axiom --stage downstream,figures
cd docs/report && tectonic -X compile report.tex && cd ../..
python docs/video/build_video.py --out docs/video/demo.mp4  # ~2 min, macOS say + ffmpeg
zsh scripts/make_release.sh                                 # upload checkpoints (gh auth required)
```
Logs of the overnight queue: `data/logs/queue_*.log`. GPU is shared: expect 1-4 s/step.

## Data rules

Use only public datasets with licences that allow this use; never commit raw data. Attribute: dataset publication(s), Cell Painting Gallery (Weisbart et al. 2024, Nature Methods), AWS Open Data Sponsorship Program.

## Conventions

- Entry script: `python run_all.py` (full) and `python run_all.py --quick` (small subset smoke test). Every result table/figure in the report is produced by scripts under `src/` and written to `results/`.
- Commit and push progressively; commit messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
