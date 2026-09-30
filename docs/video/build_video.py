#!/usr/bin/env python
"""Build the demo video from slides + figures + synthesised narration (macOS `say` + ffmpeg).

    python docs/video/build_video.py --out docs/video/demo.mp4 [--voice Samantha] [--rate 175]

Slides are defined in SLIDES below (title, bullets, image, narration). Each slide is rendered to a 1920x1080 PNG
with matplotlib, narrated with `say` (AIFF -> AAC), held for the narration length + 0.6 s, and the parts are
concatenated with ffmpeg. Requires macOS (`say`) and ffmpeg on PATH. Total length is printed at the end.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "hepatopac" / "figures"
FIG_AX = ROOT / "results" / "axiom" / "downstream"
BG, FG, ACC = "#0f172a", "#f1f5f9", "#38bdf8"


def load_slides() -> list[dict]:
    p = Path(__file__).with_name("slides.json")
    return json.loads(p.read_text())


def render_slide(s: dict, out: Path):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.text(80, 990, s["title"], color=ACC, fontsize=38, fontweight="bold", va="top")
    img = s.get("image")
    img_path = (ROOT / img) if img else None
    has_img = img_path is not None and img_path.exists()
    text_w = 0.42 if has_img else 0.9
    y = 880
    for b in s.get("bullets", []):
        for line in textwrap.wrap(b, width=int(60 * text_w / 0.42) if has_img else 95):
            ax.text(90, y, ("•  " if line == textwrap.wrap(b, width=int(60 * text_w / 0.42) if has_img else 95)[0] else "    ") + line,
                    color=FG, fontsize=24, va="top")
            y -= 46
        y -= 14
    if has_img:
        im = mpimg.imread(img_path)
        h, w = im.shape[:2]
        box_w, box_h = 1000, 800
        scale = min(box_w / w, box_h / h)
        ab = AnnotationBbox(OffsetImage(im, zoom=scale * 100 / 100), (1920 - 60 - box_w / 2, 520), frameon=False)
        ax.add_artist(ab)
    if s.get("footer"):
        ax.text(80, 40, s["footer"], color="#94a3b8", fontsize=16, va="bottom")
    ax.text(1840, 40, s.get("page", ""), color="#94a3b8", fontsize=16, va="bottom", ha="right")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("demo.mp4")))
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--rate", type=int, default=178)
    ap.add_argument("--work", default=str(Path(__file__).with_name("build")))
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    slides = load_slides()
    parts, total = [], 0.0
    for i, s in enumerate(slides):
        s["page"] = f"{i + 1}/{len(slides)}"
        png, aiff, m4a, mp4 = work / f"s{i:02d}.png", work / f"s{i:02d}.aiff", work / f"s{i:02d}.m4a", work / f"s{i:02d}.mp4"
        render_slide(s, png)
        subprocess.run(["say", "-v", a.voice, "-r", str(a.rate), "-o", str(aiff), s["narration"]], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-c:a", "aac", "-b:a", "128k", str(m4a)], check=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(m4a)],
                                   capture_output=True, text=True).stdout.strip())
        hold = dur + 0.6
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-i", str(m4a), "-t", f"{hold:.2f}",
                        "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", str(mp4)], check=True)
        parts.append(mp4)
        total += hold
        print(f"slide {i + 1}: {hold:.1f} s  (cumulative {total:.1f} s)")
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", a.out], check=True, cwd=work)
    print(f"wrote {a.out}  total {total / 60:.2f} min" + ("  WARNING: over 5 minutes" if total > 300 else ""))


if __name__ == "__main__":
    main()
