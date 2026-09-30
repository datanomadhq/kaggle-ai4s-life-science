"""Label-free cytotoxicity readouts on the Axiom U2OS subset (per-well MTT / LDH ground truth).

For every test well (unseen compounds) we compute the same interpretable field features from
(a) the real Cell Painting channels, (b) the virtual channels predicted from brightfield, and
(c) brightfield texture only, then ask with compound-grouped cross-validation:

* classification: is the well cytotoxic (MTT < 0.5)?           -> ROC-AUC, average precision
* regression:     predicted MTT viability                       -> Spearman / Pearson with measured MTT
* dose-response:  per-compound curves of nuclear count and predicted viability vs concentration

Outputs: features_{real,virtual,bf}.csv, summary.json, fig_toxicity.png, fig_dose_response.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import normalise_input, normalise_targets  # noqa: E402
from .downstream import IDX, brightfield_features, field_features, segment_nuclei  # noqa: E402

META = ["field_id", "well", "plate", "compound", "dose_um", "mtt", "ldh", "split"]


def grouped_cv(df: pd.DataFrame, feat_cols: list[str], task: str, n_splits: int = 5, seed: int = 0) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = df[feat_cols].values
    groups = df.compound.values
    gkf = GroupKFold(n_splits=n_splits)
    out = {}
    if task == "cls":
        y = (df.mtt < 0.5).astype(int).values
        for name, mk in [("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))),
                         ("gbt", lambda: GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed))]:
            p = np.zeros(len(y))
            for tr, te in gkf.split(X, y, groups):
                clf = mk()
                clf.fit(X[tr], y[tr])
                p[te] = clf.predict_proba(X[te])[:, 1]
            out[name] = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)), "n_pos": int(y.sum()), "n": int(len(y))}
            out[name + "_pred"] = p.tolist()
    else:
        y = df.mtt.values
        for name, mk in [("ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
                         ("gbt", lambda: GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, random_state=seed))]:
            p = np.zeros(len(y))
            for tr, te in gkf.split(X, y, groups):
                reg = mk()
                reg.fit(X[tr], y[tr])
                p[te] = reg.predict(X[te])
            out[name] = {"spearman": float(spearmanr(y, p).correlation), "pearson": float(pearsonr(y, p)[0]),
                         "mae": float(np.abs(y - p).mean())}
            out[name + "_pred"] = p.tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(a.manifest)
    m = m[(m.split == a.split) & m.field_id.map(lambda f: (Path(a.pred_dir) / f"{f}.npy").exists())].reset_index(drop=True)
    stats = json.loads((Path(a.npy_dir) / "stats.json").read_text())
    print(f"{len(m)} fields / {m.well.nunique()} wells / {m.compound.nunique()} compounds; MTT<0.5 wells: {int((m.mtt < 0.5).sum())}")
    rows = {"real": [], "virtual": [], "bf": []}
    for _, r in tqdm(m.iterrows(), total=len(m), desc="toxicity features"):
        raw = np.load(Path(a.npy_dir) / f"{r.field_id}.npy")
        real = normalise_targets(raw[1:6], stats)
        virt = np.load(Path(a.pred_dir) / f"{r.field_id}.npy").astype(np.float32)[:5]
        bf = normalise_input(raw[0])
        meta = {k: r[k] for k in META}
        rows["real"].append({**meta, **field_features(real, segment_nuclei(real[IDX["DNA"]]))})
        rows["virtual"].append({**meta, **field_features(virt, segment_nuclei(virt[IDX["DNA"]]))})
        rows["bf"].append({**meta, **brightfield_features(bf)})
    dfs = {k: pd.DataFrame(v) for k, v in rows.items()}
    for k, df in dfs.items():
        df.to_csv(out / f"features_{k}.csv", index=False)
    # aggregate to wells (median over fields; usually one field per well)
    summary = {"n_fields": len(m), "n_wells": int(m.well.nunique()), "n_compounds": int(m.compound.nunique()),
               "n_toxic_wells": int((m.drop_duplicates('well').mtt < 0.5).sum())}
    preds = {}
    for k, df in dfs.items():
        feat_cols = [c for c in df.columns if c not in META]
        w = df.groupby("well").agg({**{c: "median" for c in feat_cols}, "compound": "first", "mtt": "first", "ldh": "first", "dose_um": "first"}).reset_index()
        cls = grouped_cv(w, feat_cols, "cls")
        reg = grouped_cv(w, feat_cols, "reg")
        summary[k] = {"cytotoxic_cls": {n: v for n, v in cls.items() if not n.endswith("_pred")},
                      "mtt_regression": {n: v for n, v in reg.items() if not n.endswith("_pred")}}
        preds[k] = (w, np.array(cls["gbt_pred"]), np.array(reg["gbt_pred"]))
        # direct, model-free readout: nuclear count vs MTT
        if "nuclei_count" in feat_cols:
            summary[k]["nuclei_count_vs_mtt_spearman"] = float(spearmanr(w.nuclei_count, w.mtt).correlation)
    # real-vs-virtual feature fidelity on this system
    feat_cols = [c for c in dfs["real"].columns if c not in META]
    fid = pd.DataFrame({"feature": feat_cols, "pearson": [pearsonr(dfs["real"][c], dfs["virtual"][c])[0] for c in feat_cols]})
    fid.to_csv(out / "feature_fidelity.csv", index=False)
    summary["feature_fidelity_median_pearson"] = float(fid.pearson.median())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k in ("real", "virtual", "bf", "feature_fidelity_median_pearson")}, indent=1))

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, k, title in zip(axes, ["real", "virtual", "bf"], ["real Cell Painting", "virtual Cell Painting (label-free)", "brightfield texture only"]):
        w, pc, pr = preds[k]
        ax.scatter(w.mtt, pr, s=10, alpha=0.6, c=np.where(w.mtt < 0.5, "#C44E52", "#4C72B0"))
        ax.plot([0, 1.3], [0, 1.3], "k--", lw=0.8)
        ax.set_xlabel("measured MTT viability")
        ax.set_ylabel("predicted MTT (compound-grouped CV)")
        s = summary[k]["mtt_regression"]["gbt"]["spearman"]
        auc = summary[k]["cytotoxic_cls"]["gbt"]["auc"]
        ax.set_title(f"{title}\nSpearman {s:.2f}, cytotoxic AUC {auc:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_toxicity.png", dpi=200)
    plt.close(fig)
    # dose-response for the compounds with the widest MTT range
    wr, _, _ = preds["real"]
    wv, _, pv = preds["virtual"]
    rng_ = wr[wr.compound != "control"].groupby("compound").mtt.agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    top = list(rng_.index[:6])
    fig, axes = plt.subplots(1, len(top), figsize=(3.2 * len(top), 3.4), sharey=False)
    for ax, c in zip(np.atleast_1d(axes), top):
        r_ = wr[wr.compound == c].sort_values("dose_um")
        v_ = wv[wv.compound == c].sort_values("dose_um")
        ax.plot(r_.dose_um, r_.mtt, "ko-", label="MTT (measured)")
        ax.plot(r_.dose_um, r_.nuclei_count / max(wr[wr.compound == 'control'].nuclei_count.median(), 1), "s--", color="#4C72B0", label="nuclei, real DNA")
        ax.plot(v_.dose_um, v_.nuclei_count / max(wv[wv.compound == 'control'].nuclei_count.median(), 1), "^--", color="#C44E52", label="nuclei, virtual DNA")
        ax.set_xscale("log")
        ax.set_title(c[:22], fontsize=9)
        ax.set_xlabel("concentration (uM)")
    np.atleast_1d(axes)[0].set_ylabel("relative to control")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_dose_response.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
