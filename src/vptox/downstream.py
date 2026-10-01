"""Downstream biological fidelity of the virtual stain (label-free pipeline vs real stain).

For every evaluated field we run the *same* segmentation + feature pipeline on (a) the real Cell
Painting channels and (b) the virtual channels predicted from brightfield only, then ask:

1. nuclei:      do nuclear counts / areas from the virtual DNA channel agree with the real ones?
2. features:    per-feature Pearson correlation real-vs-virtual across fields (which biology survives?)
3. structure:   are well-to-well similarity matrices the same (Mantel correlation)?
4. treatment:   treated-vs-control classification with well-grouped cross-validation, real vs virtual
                vs brightfield-only features; and concordance of per-well effect magnitudes.

Outputs: nuclei.csv, features_{real,virtual,bf}.csv, feature_fidelity.csv, summary.json, figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import pearsonr, spearmanr
from skimage import filters, measure, morphology, segmentation, feature
from tqdm import tqdm

from . import CHANNELS
from .data import normalise_targets

IDX = {c: i for i, c in enumerate(CHANNELS)}


# --------------------------------------------------------------------------- segmentation

def segment_nuclei(dna: np.ndarray, min_area: int = 40) -> np.ndarray:
    """Classical Hoechst segmentation: smoothing, Otsu threshold, distance-transform watershed."""
    img = filters.gaussian(dna.astype(np.float32), sigma=1.5, preserve_range=True)
    thr = filters.threshold_otsu(img)
    mask = img > thr
    mask = morphology.remove_small_objects(mask, min_area)
    mask = ndi.binary_fill_holes(mask)
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=6, labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(dist.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = segmentation.watershed(-dist, markers, mask=mask)
    return lab


def match_objects(lab_a: np.ndarray, lab_b: np.ndarray, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Object-level matching by IoU; returns (matched, n_a, n_b)."""
    n_a, n_b = lab_a.max(), lab_b.max()
    if n_a == 0 or n_b == 0:
        return 0, int(n_a), int(n_b)
    pairs = np.stack([lab_a.ravel(), lab_b.ravel()], 1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    inter = {}
    for a, b in map(tuple, np.unique(pairs, axis=0)):
        inter[(a, b)] = int(((lab_a == a) & (lab_b == b)).sum())
    area_a = np.bincount(lab_a.ravel(), minlength=n_a + 1)
    area_b = np.bincount(lab_b.ravel(), minlength=n_b + 1)
    matched, used_b = 0, set()
    for (a, b), i in sorted(inter.items(), key=lambda kv: -kv[1]):
        if b in used_b:
            continue
        iou = i / (area_a[a] + area_b[b] - i)
        if iou >= iou_thr:
            matched += 1
            used_b.add(b)
    return matched, int(n_a), int(n_b)


# --------------------------------------------------------------------------- features

def field_features(stack: np.ndarray, lab: np.ndarray) -> dict:
    """Simple, interpretable morphological profile of one field.

    stack: [5, H, W] normalised channels; lab: nuclear label image (from the same stain type).
    """
    f = {}
    nuc = lab > 0
    cyto = ndi.binary_dilation(nuc, iterations=8) & ~nuc
    props = measure.regionprops(lab)
    f["nuclei_count"] = len(props)
    f["nuclei_area_mean"] = float(np.mean([p.area for p in props])) if props else 0.0
    f["nuclei_eccentricity_mean"] = float(np.mean([p.eccentricity for p in props])) if props else 0.0
    f["nuclei_solidity_mean"] = float(np.mean([p.solidity for p in props])) if props else 0.0
    for i, c in enumerate(CHANNELS):
        x = stack[i]
        f[f"{c}_mean"] = float(x.mean())
        f[f"{c}_std"] = float(x.std())
        f[f"{c}_p90"] = float(np.percentile(x, 90))
        f[f"{c}_frac_high"] = float((x > 0.5).mean())
        f[f"{c}_nuc_mean"] = float(x[nuc].mean()) if nuc.any() else 0.0
        f[f"{c}_cyto_mean"] = float(x[cyto].mean()) if cyto.any() else 0.0
        f[f"{c}_nuc_cyto_ratio"] = f[f"{c}_nuc_mean"] / (f[f"{c}_cyto_mean"] + 1e-3)
        gy, gx = np.gradient(filters.gaussian(x, 1.0))
        f[f"{c}_gradient_mean"] = float(np.hypot(gx, gy).mean())
        f[f"{c}_texture_contrast"] = float(np.var(x - filters.gaussian(x, 4.0)))
    return f


def brightfield_features(bf: np.ndarray) -> dict:
    """Features computed directly from the (normalised) brightfield image: the label-free control."""
    f = {"bf_std": float(bf.std()), "bf_p90": float(np.percentile(bf, 90)), "bf_p10": float(np.percentile(bf, 10))}
    gy, gx = np.gradient(filters.gaussian(bf, 1.0))
    f["bf_gradient_mean"] = float(np.hypot(gx, gy).mean())
    for s in (2.0, 4.0, 8.0):
        f[f"bf_texture_{int(s)}"] = float(np.var(bf - filters.gaussian(bf, s)))
    edges = filters.sobel(bf)
    f["bf_edge_frac"] = float((edges > np.percentile(edges, 90)).mean())
    return f


# --------------------------------------------------------------------------- analyses

def mantel(d1: np.ndarray, d2: np.ndarray) -> float:
    iu = np.triu_indices_from(d1, 1)
    return float(spearmanr(d1[iu], d2[iu]).correlation)


def well_profiles(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    return df.groupby("well")[feat_cols].median()


def standardise_to_controls(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    ctrl = df[df.compound == "control"][feat_cols]
    mu, sd = ctrl.mean(), ctrl.std().replace(0, 1.0)
    out = df.copy()
    out[feat_cols] = (df[feat_cols] - mu) / sd
    return out


def treated_vs_control_cv(df: pd.DataFrame, feat_cols: list[str], seed: int = 0) -> dict:
    """Leave-one-well-out logistic regression: is a field from a treated well? (balanced accuracy, AUC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = (df.compound != "control").astype(int).values
    X = df[feat_cols].values
    wells = df.well.values
    pred = np.zeros(len(df))
    for w in np.unique(wells):
        tr, te = wells != w, wells == w
        if len(np.unique(y[tr])) < 2:
            pred[te] = 0.5
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return {"auc": float(roc_auc_score(y, pred)), "balanced_acc": float(balanced_accuracy_score(y, pred > 0.5))}


def compound_cv(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Leave-one-well-out compound identification (5 compounds + control) at the well level (mean of field probabilities)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = df.compound.astype(str).values
    X = df[feat_cols].values
    wells = df.well.values
    correct, n = 0, 0
    for w in np.unique(wells):
        tr, te = wells != w, wells == w
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te]).mean(0)
        correct += int(clf.classes_[p.argmax()] == y[te][0])
        n += 1
    return {"well_accuracy": correct / n, "n_wells": n, "chance": 1 / len(np.unique(y))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--pred-dir", required=True, help="directory with <field_id>.npy virtual stains (from predict --save-pred)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--splits", default="test", help="comma-separated manifest splits to analyse (predictions must exist)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(a.manifest)
    stats = json.loads((Path(a.npy_dir) / "stats.json").read_text())
    pred_dir = Path(a.pred_dir)
    m = m[m.split.isin(a.splits.split(",")) & m.field_id.map(lambda f: (pred_dir / f"{f}.npy").exists())].reset_index(drop=True)
    if a.quick:
        m = m.iloc[:: max(1, len(m) // 40)]
    print(f"{len(m)} fields with predictions, {m.well.nunique()} wells")

    rows_n, rows_r, rows_v, rows_b = [], [], [], []
    for _, r in tqdm(m.iterrows(), total=len(m), desc="downstream"):
        raw = np.load(Path(a.npy_dir) / f"{r.field_id}.npy")
        real = normalise_targets(raw[1:6], stats)
        virt = np.load(pred_dir / f"{r.field_id}.npy").astype(np.float32)[:5]
        from .data import normalise_input
        bf = normalise_input(raw[0])
        lab_r = segment_nuclei(real[IDX["DNA"]])
        lab_v = segment_nuclei(virt[IDX["DNA"]])
        matched, n_r, n_v = match_objects(lab_r, lab_v)
        meta = {"field_id": r.field_id, "well": r.well, "compound": str(r.compound), "concentration": r.concentration,
                "group": r.group, "split": r.split}
        rows_n.append({**meta, "n_real": n_r, "n_virtual": n_v, "matched": matched,
                       "precision": matched / max(n_v, 1), "recall": matched / max(n_r, 1)})
        rows_r.append({**meta, **field_features(real, lab_r)})
        rows_v.append({**meta, **field_features(virt, lab_v)})
        rows_b.append({**meta, **brightfield_features(bf)})
    nuc = pd.DataFrame(rows_n)
    fr, fv, fb = pd.DataFrame(rows_r), pd.DataFrame(rows_v), pd.DataFrame(rows_b)
    nuc.to_csv(out / "nuclei.csv", index=False)
    fr.to_csv(out / "features_real.csv", index=False)
    fv.to_csv(out / "features_virtual.csv", index=False)
    fb.to_csv(out / "features_bf.csv", index=False)

    feat_cols = [c for c in fr.columns if c not in ("field_id", "well", "compound", "concentration", "group", "split")]
    bf_cols = [c for c in fb.columns if c.startswith("bf_")]
    summary = {"n_fields": len(m), "n_wells": int(m.well.nunique())}
    f1 = 2 * nuc.matched.sum() / max(nuc.n_real.sum() + nuc.n_virtual.sum(), 1)
    summary["nuclei"] = {"count_pearson": float(pearsonr(nuc.n_real, nuc.n_virtual)[0]),
                         "count_mean_real": float(nuc.n_real.mean()), "count_mean_virtual": float(nuc.n_virtual.mean()),
                         "object_f1_iou50": float(f1), "precision": float(nuc.precision.mean()), "recall": float(nuc.recall.mean())}
    fid = pd.DataFrame({"feature": feat_cols,
                        "pearson": [pearsonr(fr[c], fv[c])[0] if fr[c].std() > 0 and fv[c].std() > 0 else np.nan for c in feat_cols],
                        "spearman": [spearmanr(fr[c], fv[c]).correlation for c in feat_cols]})
    fid.to_csv(out / "feature_fidelity.csv", index=False)
    summary["feature_fidelity"] = {"median_pearson": float(fid.pearson.median()), "frac_pearson_gt_0.7": float((fid.pearson > 0.7).mean()),
                                   "n_features": len(feat_cols)}
    # well-level structure
    wr, wv = well_profiles(standardise_to_controls(fr, feat_cols), feat_cols), well_profiles(standardise_to_controls(fv, feat_cols), feat_cols)
    from scipy.spatial.distance import pdist, squareform
    if len(wr) >= 4:
        d_r, d_v = squareform(pdist(wr.values, "correlation")), squareform(pdist(wv.loc[wr.index].values, "correlation"))
        summary["well_similarity_mantel_spearman"] = mantel(d_r, d_v)
    # effect magnitude per well (mean |z| vs controls) concordance
    zr, zv = standardise_to_controls(fr, feat_cols), standardise_to_controls(fv, feat_cols)
    eff_r = zr.groupby("well")[feat_cols].mean().abs().mean(1)
    eff_v = zv.groupby("well")[feat_cols].mean().abs().mean(1).loc[eff_r.index]
    eff = pd.DataFrame({"effect_real": eff_r, "effect_virtual": eff_v}).join(m.groupby("well")[["compound", "concentration", "group"]].first())
    eff.to_csv(out / "well_effect.csv")
    treated = eff[eff.compound != "control"]
    if len(treated) >= 3:
        summary["effect_magnitude_spearman"] = float(spearmanr(treated.effect_real, treated.effect_virtual).correlation)
        summary["effect_magnitude_pearson"] = float(pearsonr(treated.effect_real, treated.effect_virtual)[0])
        # dose pairs: does the virtual stain order low < high dose the same way as the real stain?
        pairs = []
        for c, g in treated.groupby("compound"):
            lo, hi = g[g.concentration == "low"], g[g.concentration == "high"]
            if len(lo) and len(hi):
                r_dir = np.sign(hi.effect_real.mean() - lo.effect_real.mean())
                v_dir = np.sign(hi.effect_virtual.mean() - lo.effect_virtual.mean())
                pairs.append({"compound": str(c), "real_high_minus_low": float(hi.effect_real.mean() - lo.effect_real.mean()),
                              "virtual_high_minus_low": float(hi.effect_virtual.mean() - lo.effect_virtual.mean()), "agree": bool(r_dir == v_dir)})
        if pairs:
            summary["dose_pairs"] = {"n": len(pairs), "agree": int(sum(p["agree"] for p in pairs)), "pairs": pairs}
    # treated vs control classification, real vs virtual vs brightfield-only
    if (fr.compound == "control").sum() > 0 and (fr.compound != "control").sum() > 0 and fr.well.nunique() >= 4:
        summary["treated_vs_control"] = {"real": treated_vs_control_cv(fr, feat_cols), "virtual": treated_vs_control_cv(fv, feat_cols),
                                         "brightfield": treated_vs_control_cv(fb, bf_cols)}
        if fr.compound.nunique() >= 3 and fr.well.nunique() >= 6:
            summary["compound_id"] = {"real": compound_cv(fr, feat_cols), "virtual": compound_cv(fv, feat_cols), "brightfield": compound_cv(fb, bf_cols)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
