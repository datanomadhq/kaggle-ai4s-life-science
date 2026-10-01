#!/bin/zsh
# Upload trained checkpoints as GitHub release assets (requires `gh auth login`).
# Assets are named <dataset>__<experiment>__best.pt and fetched by `python -m vptox.checkpoints`.
set -e
cd "$(dirname "$0")/.."
TAG=v1.0-checkpoints
mkdir -p build/release
for d in hepatopac axiom; do
  for p in results/$d/*/best.pt; do
    [ -f "$p" ] || continue
    e=$(basename "$(dirname "$p")")
    cp "$p" "build/release/${d}__${e}__best.pt"
  done
done
ls -la build/release
gh release view $TAG >/dev/null 2>&1 || gh release create $TAG --title "Trained checkpoints" --notes "U-Net checkpoints for VirtualPaint-Tox (see README). Fetched automatically by python -m vptox.checkpoints." --latest=false
gh release upload $TAG build/release/*.pt --clobber
