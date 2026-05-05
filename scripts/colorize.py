"""
Standalone colorization CLI — colorize a B&W / sepia image without running SR.

Usage:
    python scripts/colorize.py --input photo_bw.jpg --output photo_color.png
    python scripts/colorize.py --input-dir photos/ --output-dir photos_color/
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from colorizer import Colorizer  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="single image path")
    g.add_argument("--input-dir", help="directory of images to colorize")
    ap.add_argument("--output")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    c = Colorizer()

    if args.input:
        if not args.output:
            ap.error("--output required with --input")
        img = cv2.imread(args.input)
        if img is None:
            raise SystemExit(f"could not read {args.input}")
        out = c.colorize(img)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, out)
        print(f"[done] {args.output}")
        return

    if not args.output_dir:
        ap.error("--output-dir required with --input-dir")
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(in_dir.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] {p.name}")
            continue
        out = c.colorize(img)
        dst = out_dir / p.name
        cv2.imwrite(str(dst), out)
        print(f"[done] {dst}")


if __name__ == "__main__":
    main()
