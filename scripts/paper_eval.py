"""
Paper-grade evaluation: HAT-GAN 180k vs Bicubic baseline on tests/synthetic/.
Output: results/paper_eval.json with PSNR/SSIM/LPIPS aggregates.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "BasicSR"))
sys.path.insert(0, str(REPO))

from basicsr.archs.hat_arch import HAT
from basicsr.utils import img2tensor, tensor2img
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips as lpips_pkg


def build_hat():
    return HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=2, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, depths=[6]*12, embed_dim=180,
        num_heads=[6]*12, upsampler="pixelshuffle",
    )


@torch.no_grad()
def hat_upscale(model, img_bgr, tile=400, pad=32):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = img2tensor([rgb / 255.0], bgr2rgb=False, float32=True)[0].unsqueeze(0).cuda()
    b, c, h, w = t.size()
    s = 4
    out = torch.zeros((b, c, h*s, w*s), device="cuda")
    wt  = torch.zeros_like(out)
    stride = max(1, tile - pad*2)
    h_idx = list(range(0, max(1, h - tile), stride)) + [max(0, h - tile)]
    w_idx = list(range(0, max(1, w - tile), stride)) + [max(0, w - tile)]
    for y in sorted(set(h_idx)):
        for x in sorted(set(w_idx)):
            tt = t[:, :, y:y+tile, x:x+tile]
            ph = (16 - tt.shape[2] % 16) % 16
            pw = (16 - tt.shape[3] % 16) % 16
            tp = torch.nn.functional.pad(tt, (0, pw, 0, ph), mode="reflect")
            sr = model(tp)
            sr = sr[:, :, : tt.shape[2]*s, : tt.shape[3]*s]
            t_h, t_w = sr.shape[2], sr.shape[3]
            mask = torch.ones((1, 1, t_h, t_w), device="cuda")
            ps = pad*s
            for i in range(ps):
                mask[:, :, i, :]       *= i/ps
                mask[:, :, t_h-i-1, :] *= i/ps
                mask[:, :, :, i]       *= i/ps
                mask[:, :, :, t_w-i-1] *= i/ps
            out[:, :, y*s:y*s+t_h, x*s:x*s+t_w] += sr * mask
            wt [:, :, y*s:y*s+t_h, x*s:x*s+t_w] += mask
    out = out / wt.clamp(min=1e-8)
    return tensor2img([out], rgb2bgr=True)


def compute_metrics(sr_bgr, gt_bgr, lpips_fn):
    if sr_bgr.shape != gt_bgr.shape:
        sr_bgr = cv2.resize(sr_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))
    psnr = float(psnr_fn(gt_bgr, sr_bgr, data_range=255))
    ssim = float(ssim_fn(
        cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY),
        data_range=255,
    ))
    rgb_sr = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t_sr = torch.from_numpy(rgb_sr.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2 - 1
    t_gt = torch.from_numpy(rgb_gt.transpose(2, 0, 1)).unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad():
        lp = float(lpips_fn(t_sr, t_gt).item())
    return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def aggregate(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ckpt = REPO / "BasicSR" / "experiments" / "HAT_GAN_Historical_v1" / "models" / "net_g_180000.pth"
    test_dir = REPO / "tests" / "synthetic"
    out_path = REPO / "results" / "paper_eval.json"

    print(f"[load] checkpoint = {ckpt}", flush=True)
    model = build_hat()
    state = torch.load(str(ckpt), map_location="cuda", weights_only=True)
    key = "params_ema" if "params_ema" in state else "params"
    model.load_state_dict(state[key], strict=True)
    model.eval().cuda()

    print("[load] LPIPS (alex)", flush=True)
    lpips_fn = lpips_pkg.LPIPS(net="alex").cuda()

    gt_dir = test_dir / "GT"
    lq_dir = test_dir / "LQ"
    paths = sorted([p for p in lq_dir.iterdir() if p.is_file()])
    print(f"[run] {len(paths)} test images", flush=True)

    hat_rows = []
    bic_rows = []
    t0 = time.time()
    for i, lq_path in enumerate(paths, 1):
        gt_path = gt_dir / lq_path.name
        if not gt_path.exists():
            continue
        lq = cv2.imread(str(lq_path))
        gt = cv2.imread(str(gt_path))
        if lq is None or gt is None:
            continue

        # HAT-GAN
        sr = hat_upscale(model, lq)
        m_hat = compute_metrics(sr, gt, lpips_fn)
        hat_rows.append(m_hat)

        # Bicubic baseline
        bic = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        m_bic = compute_metrics(bic, gt, lpips_fn)
        bic_rows.append(m_bic)

        elapsed = time.time() - t0
        print(f"[{i}/{len(paths)}] {lq_path.name}  HAT psnr={m_hat['psnr']:.2f} ssim={m_hat['ssim']:.4f} lpips={m_hat['lpips']:.4f}  |  Bicubic psnr={m_bic['psnr']:.2f} ssim={m_bic['ssim']:.4f} lpips={m_bic['lpips']:.4f}  ({elapsed:.0f}s)", flush=True)

    summary = {
        "n": len(hat_rows),
        "checkpoint": str(ckpt),
        "test_dir": str(test_dir),
        "elapsed_sec": round(time.time() - t0, 1),
        "HAT_GAN_180k": {
            "psnr_mean": aggregate(hat_rows, "psnr")[0], "psnr_std": aggregate(hat_rows, "psnr")[1],
            "ssim_mean": aggregate(hat_rows, "ssim")[0], "ssim_std": aggregate(hat_rows, "ssim")[1],
            "lpips_mean": aggregate(hat_rows, "lpips")[0], "lpips_std": aggregate(hat_rows, "lpips")[1],
        },
        "Bicubic": {
            "psnr_mean": aggregate(bic_rows, "psnr")[0], "psnr_std": aggregate(bic_rows, "psnr")[1],
            "ssim_mean": aggregate(bic_rows, "ssim")[0], "ssim_std": aggregate(bic_rows, "ssim")[1],
            "lpips_mean": aggregate(bic_rows, "lpips")[0], "lpips_std": aggregate(bic_rows, "lpips")[1],
        },
        "rows_HAT_GAN": hat_rows,
        "rows_Bicubic": bic_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)
    print(f"HAT-GAN: psnr={summary['HAT_GAN_180k']['psnr_mean']:.3f}  ssim={summary['HAT_GAN_180k']['ssim_mean']:.4f}  lpips={summary['HAT_GAN_180k']['lpips_mean']:.4f}", flush=True)
    print(f"Bicubic: psnr={summary['Bicubic']['psnr_mean']:.3f}  ssim={summary['Bicubic']['ssim_mean']:.4f}  lpips={summary['Bicubic']['lpips_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
