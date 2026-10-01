"""Fetch released checkpoints so that evaluation stages run without training.

Checkpoints are attached to the GitHub release ``v1.0-checkpoints`` of the public repository as
``<dataset>__<experiment>__best.pt``. ``python -m vptox.checkpoints --dataset hepatopac`` downloads the
ones that are missing under results/<dataset>/<experiment>/best.pt.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/datanomadhq/kaggle-ai4s-life-science/releases/download/v1.0-checkpoints/"
ROOT = Path(__file__).resolve().parents[2]


def fetch_checkpoint(dataset: str, experiment: str, results_root: Path = ROOT / "results") -> Path | None:
    dest = results_root / dataset / experiment / "best.pt"
    if dest.exists():
        return dest
    url = f"{RELEASE}{dataset}__{experiment}__best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"downloaded {url} -> {dest}")
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"no released checkpoint for {dataset}/{experiment} ({e})", file=sys.stderr)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="hepatopac")
    ap.add_argument("--experiments", default="unet")
    a = ap.parse_args(argv)
    for e in a.experiments.split(","):
        fetch_checkpoint(a.dataset, e)


if __name__ == "__main__":
    main()
