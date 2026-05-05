"""
Smoke test for HistoricalDegradationDataset.

Pulls 8 samples and writes an 8x2 grid (top: HR/GT, bottom: synthetic LQ
upscaled to GT size with NEAREST so degradations stay visible).

Usage:
    python scripts/smoke_test_dataset.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "BasicSR"))

from basicsr.data.historical_degradation_dataset import HistoricalDegradationDataset  # noqa: E402


def tensor_to_bgr_u8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255.0).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW -> HWC, RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    opt = {
        "dataroot_gt": str(REPO_ROOT / "datasets" / "HR_train"),
        "meta_info": str(REPO_ROOT / "datasets" / "meta_info_train.txt"),
        "io_backend": {"type": "disk"},
        "gt_size": 192,
        "use_hflip": True,
        "use_rot": True,
    }

    ds = HistoricalDegradationDataset(opt)
    n = min(8, len(ds))
    cells_top, cells_bot = [], []
    for i in range(n):
        sample = ds[i]
        gt = tensor_to_bgr_u8(sample["gt"])
        lq = tensor_to_bgr_u8(sample["lq"])
        lq_up = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        cells_top.append(gt)
        cells_bot.append(lq_up)

    grid_top = np.hstack(cells_top)
    grid_bot = np.hstack(cells_bot)
    grid = np.vstack([grid_top, grid_bot])

    out_dir = REPO_ROOT / "figures" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "historical_dataset_smoke.png"
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})")
    print("Top row: GT (192x192). Bottom row: synthetic LQ (48x48 upscaled NEAREST to 192).")


if __name__ == "__main__":
    main()
