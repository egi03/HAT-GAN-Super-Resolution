"""
Render the 9-stage degradation storyboard for a single HR image.

Becomes "Figure 2: our degradation model" in the paper.

Usage:
    python scripts/render_storyboard.py --input helpers/original_hr.jpg --out figures/storyboard
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "BasicSR"))

from basicsr.degradation_machine import UltimateDegraderWithStoryboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO_ROOT / "helpers" / "original_hr.jpg"))
    ap.add_argument("--out", default=str(REPO_ROOT / "figures" / "storyboard"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"could not read {args.input}")

    storyboard = UltimateDegraderWithStoryboard().run_with_storyboard(img)
    for name, stage_img in storyboard.items():
        out_path = out_dir / f"{name}.png"
        cv2.imwrite(str(out_path), stage_img)
        print(f"  wrote {out_path}  ({stage_img.shape[1]}x{stage_img.shape[0]})")

    # Also write a single composite (3x3 grid).
    keys = list(storyboard.keys())[:9]
    cells = []
    target_h = 256
    for k in keys:
        im = storyboard[k]
        h, w = im.shape[:2]
        scale = target_h / h
        im = cv2.resize(im, (int(w * scale), target_h))
        cv2.putText(im, k, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(im, k, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cells.append(im)

    # Pad to a uniform width
    target_w = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, target_w - c.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for c in cells]

    import numpy as np
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, 9, 3)]
    composite = np.vstack(rows)
    composite_path = out_dir / "composite.png"
    cv2.imwrite(str(composite_path), composite)
    print(f"  wrote {composite_path}")


if __name__ == "__main__":
    main()
