"""Download the public OASIS HepatoPAC images (Cell Painting Gallery, AWS Open Data, no credentials).

Uses plain HTTPS with a thread pool, so no AWS CLI or account is needed. ``--wells`` / ``--sites``
restrict the subset (used by the ``--quick`` smoke test).

Example: python -m vptox.download --dataset hepatopac_islands --out data/hepatopac/islands
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HTTPS = "https://cellpainting-gallery.s3.amazonaws.com/"
DATASETS = {
    "hepatopac_islands": {
        "load_data": "cpg0037-oasis/hepatopac/workspace/load_data_csv/2026_04_13_OASIS_HepatoPAC_Batch1/HepatoPAC_Islands_20260413/load_data.csv",
        "platemap": "cpg0037-oasis/hepatopac/workspace/metadata/platemaps/2026_04_13_OASIS_HepatoPAC_Batch1/platemap/HepatoPAC_Preciscan.txt",
    },
    "hepatopac_preciscan": {
        "load_data": "cpg0037-oasis/hepatopac/workspace/load_data_csv/2026_04_13_OASIS_HepatoPAC_Batch1/HepatoPAC_Preciscan/load_data.csv",
        "platemap": "cpg0037-oasis/hepatopac/workspace/metadata/platemaps/2026_04_13_OASIS_HepatoPAC_Batch1/platemap/HepatoPAC_Preciscan.txt",
    },
}
URL_COLS = ["URL_OrigBrightfield", "URL_OrigDNA", "URL_OrigER", "URL_OrigRNA", "URL_OrigAGP", "URL_OrigMito"]


def fetch(key: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(HTTPS + key, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"retry {attempt+1} for {key}: {e}", file=sys.stderr)
    return dest


def read_csv_from_s3(key: str) -> pd.DataFrame:
    with urllib.request.urlopen(HTTPS + key, timeout=120) as r:
        return pd.read_csv(io.BytesIO(r.read()), dtype=str)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="hepatopac_islands", choices=sorted(DATASETS))
    ap.add_argument("--out", required=True, help="image output directory")
    ap.add_argument("--meta-out", default=None, help="where to store load_data.csv and platemap (default: <out>/../meta)")
    ap.add_argument("--wells", default="", help="comma-separated wells to restrict to (default: all)")
    ap.add_argument("--sites", default="", help="comma-separated site numbers to restrict to (default: all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--keys-file", default="", help="download this explicit list of S3 keys instead of a dataset")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.keys_file:
        keys = [k.strip() for k in Path(a.keys_file).read_text().splitlines() if k.strip()]
    else:
        d = DATASETS[a.dataset]
        meta = Path(a.meta_out) if a.meta_out else out.parent / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        ld_path = fetch(d["load_data"], meta / f"{a.dataset}_load_data.csv")
        fetch(d["platemap"], meta / "platemap.txt")
        ld = pd.read_csv(ld_path, dtype=str)
        if a.wells:
            ld = ld[ld.Metadata_Well.isin(a.wells.split(","))]
        if a.sites:
            ld = ld[ld.Metadata_Site.isin(a.sites.split(","))]
        keys = sorted({u.replace("s3://cellpainting-gallery/", "") for c in URL_COLS for u in ld[c]})
    # with an explicit key list (e.g. several Opera Phenix plates whose file names repeat), keep the parent folder
    def dest(k: str) -> Path:
        return out / os.path.basename(os.path.dirname(k)) / os.path.basename(k) if a.keys_file else out / os.path.basename(k)
    todo = [k for k in keys if not dest(k).exists()]
    print(f"{len(keys)} files, {len(todo)} to download -> {out}")
    n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, k, dest(k)) for k in todo]
        for f in as_completed(futs):
            f.result()
            n += 1
            if n % 500 == 0:
                print(f"  {n}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
