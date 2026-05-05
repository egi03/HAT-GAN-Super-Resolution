"""
CPU-only bicubic baseline: PSNR/SSIM/LPIPS over tests/synthetic.
Reads LQ, upscales 4x via bicubic, compares to GT.
Output: results/bicubic_baseline.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_pkg

REPO = Path(__file__).resolve().parents[1]


def main():
    test_dir = REPO / "tests" / "synthetic"
    gt_dir = test_dir / "GT"
    lq_dir = test_dir / "LQ"
    out_path = REPO / "results" / "bicubic_baseline.json"

    print("[load] LPIPS (alex) on CUDA", flush=True)
    lpips_fn = lpips_pkg.LPIPS(net="alex").cuda()

    paths = sorted([p for p in lq_dir.iterdir() if p.is_file()])
    print(f"[run] {len(paths)} test images", flush=True)
    rows = []
    t0 = time.time()
    for i, lq_path in enumerate(paths, 1):
        gt_path = gt_dir / lq_path.name
        if not gt_path.exists():
            continue
        lq = cv2.imread(str(lq_path))
        gt = cv2.imread(str(gt_path))
        if lq is None or gt is None:
            continue
        bic = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnr = float(psnr_fn(gt, bic, data_range=255))
        ssim = float(ssim_fn(
            cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(bic, cv2.COLOR_BGR2GRAY),
            data_range=255,
        ))
        rgb_sr = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_sr = torch.from_numpy(rgb_sr.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2 - 1
        t_gt = torch.from_numpy(rgb_gt.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            lp = float(lpips_fn(t_sr, t_gt).item())
        rows.append({"filename": lq_path.name, "psnr": psnr, "ssim": ssim, "lpips": lp})
        elapsed = time.time() - t0
        print(f"[{i}/{len(paths)}] {lq_path.name}  psnr={psnr:.2f} ssim={ssim:.4f} lpips={lp:.4f}  ({elapsed:.0f}s)", flush=True)

    psnrs = [r["psnr"] for r in rows]
    ssims = [r["ssim"] for r in rows]
    lpipses = [r["lpips"] for r in rows]
    summary = {
        "n": len(rows),
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "lpips_mean": float(np.mean(lpipses)), "lpips_std": float(np.std(lpipses)),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {out_path}", flush=True)
    print(f"Bicubic: psnr={summary['psnr_mean']:.3f}  ssim={summary['ssim_mean']:.4f}  lpips={summary['lpips_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
