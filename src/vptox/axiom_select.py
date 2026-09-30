"""Select a compact, label-rich subset of the OASIS/Axiom U2OS Cell Painting data (cpg0037-oasis/axiom).

Each Axiom plate has 384 wells x 15 fields x 6 channels (~8.5 MB per TIFF), far too much for a laptop, so
we keep one central field per well and choose compounds with complete 8-point dose series in one
production batch, preferring compounds that actually show cytotoxicity (wide MTT range), plus DMSO
vehicle wells on the same plates. The result is a well-level table with biochemical viability
(MTT, LDH) that we use as ground truth for label-free toxicity prediction.

Writes <out>/selection.csv (one row per selected field with metadata) and <out>/keys.txt (S3 keys).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

URL_COLS = ["URL_OrigBrightfield", "URL_OrigDNA", "URL_OrigER", "URL_OrigRNA", "URL_OrigAGP", "URL_OrigMito"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", required=True, help="directory with load_data_<batch>/ and metadata_<batch>/")
    ap.add_argument("--batch", default="prod_25")
    ap.add_argument("--n-compounds", type=int, default=80)
    ap.add_argument("--n-dmso", type=int, default=96)
    ap.add_argument("--fields-per-well", type=int, default=1)
    ap.add_argument("--extra-toxic-mtt", type=float, default=0.7,
                    help="also include every well of the batch with MTT below this (cytotoxic wells are rare, ~3%%)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    meta = Path(a.meta_dir)
    bio = pd.concat([pd.read_parquet(f) for f in glob.glob(str(meta / f"metadata_{a.batch}" / "*" / "biochem.parquet"))])
    bio = bio.rename(columns={"well": "Metadata_Well", "plate": "Metadata_Plate"})
    ld = pd.concat([pd.read_csv(f, dtype=str) for f in glob.glob(str(meta / f"load_data_{a.batch}" / "*" / "load_data.csv"))])
    for c in ["Metadata_PositionX", "Metadata_PositionY", "Metadata_Site"]:
        ld[c] = ld[c].astype(float)
    # central field(s) per well: closest to the mean field position
    ld["r2"] = (ld.Metadata_PositionX - ld.groupby(["Metadata_Plate", "Metadata_Well"]).Metadata_PositionX.transform("mean")) ** 2 + \
               (ld.Metadata_PositionY - ld.groupby(["Metadata_Plate", "Metadata_Well"]).Metadata_PositionY.transform("mean")) ** 2
    ld = ld.sort_values("r2").groupby(["Metadata_Plate", "Metadata_Well"]).head(a.fields_per_well)

    wells = bio.merge(ld[["Metadata_Plate", "Metadata_Well"]].drop_duplicates(), on=["Metadata_Plate", "Metadata_Well"])
    cmp = wells[wells.compound_name != "DMSO"]
    per = cmp.groupby("compound_name").agg(n=("Metadata_Well", "size"), n_doses=("compound_concentration_um", "nunique"),
                                            mtt_min=("mtt_normalized", "min"), mtt_max=("mtt_normalized", "max"),
                                            ldh_max=("ldh_normalized", "max"))
    per["mtt_range"] = per.mtt_max - per.mtt_min
    complete = per[per.n_doses >= 8].sort_values("mtt_range", ascending=False)
    rng = np.random.default_rng(a.seed)
    n_tox = a.n_compounds // 2
    toxic = list(complete.index[:n_tox])                                   # widest cytotoxic response
    rest = list(complete.index[n_tox:])
    rng.shuffle(rest)
    chosen = toxic + rest[: a.n_compounds - n_tox]                         # plus random (mostly inactive) ones
    sel = cmp[cmp.compound_name.isin(chosen) | (cmp.mtt_normalized < a.extra_toxic_mtt)]
    dmso = wells[wells.compound_name == "DMSO"]
    # spread DMSO wells over the plates that contribute compound wells
    plates = sel.Metadata_Plate.value_counts()
    dmso_take = []
    for p, n in plates.items():
        d = dmso[dmso.Metadata_Plate == p]
        k = max(2, int(round(a.n_dmso * n / len(sel))))
        dmso_take.append(d.sample(min(k, len(d)), random_state=a.seed))
    sel = pd.concat([sel] + dmso_take)
    fields = ld.merge(sel, on=["Metadata_Plate", "Metadata_Well"])
    fields["field_id"] = fields.Metadata_Plate.str.replace("plate_", "p") + "_" + fields.Metadata_Well + "_s" + fields.Metadata_Site.astype(int).astype(str).str.zfill(2)
    keep = ["field_id", "Metadata_Plate", "Metadata_Well", "Metadata_Site", "compound_name", "OASIS_ID", "compound_concentration_um",
            "mtt_normalized", "ldh_normalized", "compound_target", "compound_smiles"] + URL_COLS
    fields = fields[keep].drop_duplicates("field_id").sort_values("field_id").reset_index(drop=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fields.to_csv(out / "selection.csv", index=False)
    keys = sorted({u.replace("s3://cellpainting-gallery/", "") for c in URL_COLS for u in fields[c]})
    (out / "keys.txt").write_text("\n".join(keys) + "\n")
    n_wells = fields[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
    print(f"compounds: {len(chosen)} (complete dose series in {a.batch}: {len(complete)})  wells: {n_wells}  fields: {len(fields)}  "
          f"files: {len(keys)} (~{len(keys) * 8.5 / 1024:.1f} GB)  plates: {fields.Metadata_Plate.nunique()}")
    print("MTT of selected compound wells:", np.round(np.percentile(sel[sel.compound_name != 'DMSO'].mtt_normalized, [0, 10, 25, 50, 75, 100]), 2))
    print("wells with MTT<0.5:", int((sel.mtt_normalized < 0.5).sum()), " MTT<0.8:", int((sel.mtt_normalized < 0.8).sum()))


if __name__ == "__main__":
    main()
