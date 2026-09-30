"""Manifest for the Axiom U2OS subset: same columns as the HepatoPAC manifest plus viability labels.

Split is by *compound* (all doses/wells of a compound share a split) so that test wells are unseen
compounds; DMSO wells are distributed over splits by plate. A 2-fold ``fold`` column (also by compound)
supports cross-fitted virtual stains for every well.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import CHANNELS, INPUT

URL_COLS = {"Brightfield": "URL_OrigBrightfield", "DNA": "URL_OrigDNA", "ER": "URL_OrigER",
            "RNA": "URL_OrigRNA", "AGP": "URL_OrigAGP", "Mito": "URL_OrigMito"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = pd.read_csv(a.selection)
    m = pd.DataFrame({
        "field_id": s.field_id, "plate": s.Metadata_Plate, "well": s.Metadata_Plate.str.replace("plate_", "p") + "_" + s.Metadata_Well,
        "site": s.Metadata_Site.astype(int), "compound": s.compound_name, "oasis_id": s.OASIS_ID,
        "dose_um": s.compound_concentration_um, "mtt": s.mtt_normalized, "ldh": s.ldh_normalized,
        "target": s.compound_target, "smiles": s.compound_smiles,
    })
    for ch, col in URL_COLS.items():
        m[f"file_{ch}"] = s[col].map(lambda u: os.path.join(os.path.basename(os.path.dirname(u)), os.path.basename(u)))
    m["compound"] = m["compound"].replace("DMSO", "control")
    m["concentration"] = np.where(m.compound == "control", "none", m.dose_um.round(3).astype(str))
    m["group"] = m.compound
    present = np.array([all(Path(a.image_dir, f).exists() for f in fs)
                        for fs in m[[f"file_{c}" for c in [INPUT] + CHANNELS]].values])
    m = m[present].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    cmpds = sorted(c for c in m.compound.unique() if c != "control")
    rng.shuffle(cmpds)
    n_test, n_val = int(round(a.test_frac * len(cmpds))), int(round(a.val_frac * len(cmpds)))
    split = {c: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, c in enumerate(cmpds)}
    fold = {c: i % 2 for i, c in enumerate(cmpds)}
    m["split"] = m.compound.map(split)
    m["fold"] = m.compound.map(fold)
    ctrl = m.compound == "control"
    wells = m.loc[ctrl, "well"].unique()
    rng.shuffle(wells)
    ctrl_split = {w: ("test" if i < int(round(a.test_frac * len(wells))) else "val" if i < int(round((a.test_frac + a.val_frac) * len(wells))) else "train")
                  for i, w in enumerate(wells)}
    m.loc[ctrl, "split"] = m.loc[ctrl, "well"].map(ctrl_split)
    m.loc[ctrl, "fold"] = [i % 2 for i in range(ctrl.sum())]
    m["fold"] = m["fold"].astype(int)
    m = m.sort_values("field_id").reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.out, index=False)
    for sp in ["train", "val", "test"]:
        sub = m[m.split == sp]
        print(f"{sp:5s}: {len(sub):4d} fields, {sub.well.nunique():4d} wells, {sub[sub.compound != 'control'].compound.nunique():3d} compounds, "
              f"{int((sub.mtt < 0.5).sum()):3d} wells MTT<0.5, {int((sub.compound == 'control').sum()):3d} DMSO")


if __name__ == "__main__":
    main()
