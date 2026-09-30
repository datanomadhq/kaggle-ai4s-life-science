"""Build a field-level manifest from a Cell Painting Gallery ``load_data.csv`` and a platemap.

One row per imaging field (plate, well, site) with the local file name of each channel, the
well's treatment metadata and a train/val/test split assigned *by well* (all fields of a well
share a split, so a test field never shares a well with a training field).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import CHANNELS, INPUT

URL_COLS = {"Brightfield": "URL_OrigBrightfield", "DNA": "URL_OrigDNA", "ER": "URL_OrigER",
            "RNA": "URL_OrigRNA", "AGP": "URL_OrigAGP", "Mito": "URL_OrigMito"}


def load_hepatopac_platemap(path: str | os.PathLike) -> pd.DataFrame:
    pm = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    pm = pm.rename(columns={"well_position": "well"})
    pm["compound"] = pm["compound"].replace("", "control")
    pm["concentration"] = pm["concentration"].replace("", "none")
    pm["group"] = pm["compound"] + "_" + pm["concentration"]
    return pm[["well", "compound", "concentration", "control_type", "group"]]


def build_manifest(load_data_csv: str, platemap: str, image_dir: str, out_csv: str,
                   seed: int = 0, n_test_per_group: int = 1, n_val_per_group: float = 0.5,
                   wells: list[str] | None = None, sites: list[str] | None = None) -> pd.DataFrame:
    ld = pd.read_csv(load_data_csv, dtype=str)
    if wells:
        ld = ld[ld.Metadata_Well.isin(wells)]
    if sites:
        ld = ld[ld.Metadata_Site.isin(sites)]
    rows = []
    for _, r in ld.iterrows():
        rec = {"plate": r["Metadata_Plate"], "well": r["Metadata_Well"], "site": int(r["Metadata_Site"])}
        for ch, col in URL_COLS.items():
            rec[f"file_{ch}"] = os.path.basename(r[col])
        rows.append(rec)
    m = pd.DataFrame(rows)
    m["field_id"] = m["well"] + "_s" + m["site"].astype(str).str.zfill(2)

    pm = load_hepatopac_platemap(platemap)
    m = m.merge(pm, on="well", how="left")
    if m["compound"].isna().any():
        missing = sorted(m.loc[m["compound"].isna(), "well"].unique())
        raise ValueError(f"wells without platemap entry: {missing}")

    # keep only fields whose six files exist locally
    present = np.array([all(Path(image_dir, f).exists() for f in fs)
                        for fs in m[[f"file_{c}" for c in [INPUT] + CHANNELS]].values])
    m = m[present].reset_index(drop=True)

    # split by well, stratified by treatment group
    rng = np.random.default_rng(seed)
    split = {}
    for g, wells in m.groupby("group")["well"].unique().items():
        wells = list(wells)
        rng.shuffle(wells)
        # controls are the reference for every downstream comparison: hold out two of them
        n_test = min(n_test_per_group + (1 if g.startswith("control") else 0), len(wells) - 1)
        n_val = int(round(n_val_per_group)) if n_val_per_group >= 1 else (1 if rng.random() < n_val_per_group else 0)
        n_val = min(n_val, max(0, len(wells) - n_test - 1))
        for i, w in enumerate(wells):
            split[w] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    m["split"] = m["well"].map(split)
    m = m.sort_values(["well", "site"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_csv, index=False)
    return m


def summarize(m: pd.DataFrame) -> str:
    lines = [f"fields: {len(m)}  wells: {m['well'].nunique()}"]
    for s in ["train", "val", "test"]:
        sub = m[m["split"] == s]
        lines.append(f"  {s:5s}: {len(sub):5d} fields, {sub['well'].nunique():3d} wells, groups: "
                     + ", ".join(f"{g}({n})" for g, n in sub.groupby('group')['well'].nunique().items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load-data", required=True)
    ap.add_argument("--platemap", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wells", default="", help="comma-separated subset of wells (smoke tests)")
    ap.add_argument("--sites", default="", help="comma-separated subset of sites (smoke tests)")
    a = ap.parse_args()
    m = build_manifest(a.load_data, a.platemap, a.image_dir, a.out, seed=a.seed,
                       wells=a.wells.split(",") if a.wells else None, sites=a.sites.split(",") if a.sites else None)
    print(summarize(m))


if __name__ == "__main__":
    main()
