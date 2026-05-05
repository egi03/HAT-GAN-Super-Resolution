"""
Run a HAT-GAN checkpoint on a paired test set and compute reference + no-reference metrics.

Reference (synthetic, has GT): PSNR, SSIM, LPIPS, optional DISTS.
No-reference (historical, no GT): MANIQA, MUSIQ, CLIP-IQA via pyiqa (optional).

Output: results/<tag>.json with per-image and aggregate numbers.

Usage:
    python scripts/run_baseline_metrics.py --checkpoint BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_180000.pth \
        --tag baseline_180k --paired tests/synthetic --no-ref tests/historical
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "BasicSR"))
sys.path.insert(0, str(REPO_ROOT))

from basicsr.archs.hat_arch import HAT  # noqa: E402
from basicsr.utils import img2tensor, tensor2img  # noqa: E402

from skimage.metrics import peak_signal_noise_ratio as psnr_fn  # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

try:
    import lpips as lpips_pkg
    LPIPS_OK = True
except ImportError:
    LPIPS_OK = False

try:
    import pyiqa
    PYIQA_OK = True
except ImportError:
    PYIQA_OK = False


def build_hat_l() -> HAT:
    return HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=2, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5,
        depths=[6] * 12,
        embed_dim=180,
        num_heads=[6] * 12,
        upsampler="pixelshuffle",
    )


def load_checkpoint(ckpt_path: str, params_key: str | None = None) -> HAT:
    model = build_hat_l()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    key = params_key or ("params_ema" if "params_ema" in state else "params")
    model.load_state_dict(state[key], strict=True)
    return model.eval().cuda()


@torch.no_grad()
def hat_upscale(model: HAT, img_bgr: np.ndarray, tile_size: int = 400, tile_pad: int = 32) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = img2tensor([img_rgb / 255.0], bgr2rgb=False, float32=True)[0].unsqueeze(0).cuda()

    b, c, h, w = img_tensor.size()
    scale = 4
    out = torch.zeros((b, c, h * scale, w * scale), device="cuda")
    weight = torch.zeros_like(out)

    stride = max(1, tile_size - tile_pad * 2)
    h_idx = list(range(0, max(1, h - tile_size), stride)) + [max(0, h - tile_size)]
    w_idx = list(range(0, max(1, w - tile_size), stride)) + [max(0, w - tile_size)]

    for y in sorted(set(h_idx)):
        for x in sorted(set(w_idx)):
            tile = img_tensor[:, :, y:y + tile_size, x:x + tile_size]
            ph = (16 - tile.shape[2] % 16) % 16
            pw = (16 - tile.shape[3] % 16) % 16
            tile_p = torch.nn.functional.pad(tile, (0, pw, 0, ph), mode="reflect")
            sr = model(tile_p)
            sr = sr[:, :, : tile.shape[2] * scale, : tile.shape[3] * scale]

            t_h, t_w = sr.shape[2], sr.shape[3]
            mask = torch.ones((1, 1, t_h, t_w), device="cuda")
            pad_s = tile_pad * scale
            for i in range(pad_s):
                mask[:, :, i, :] *= i / pad_s
                mask[:, :, t_h - i - 1, :] *= i / pad_s
                mask[:, :, :, i] *= i / pad_s
                mask[:, :, :, t_w - i - 1] *= i / pad_s
            out[:, :, y * scale:y * scale + t_h, x * scale:x * scale + t_w] += sr * mask
            weight[:, :, y * scale:y * scale + t_h, x * scale:x * scale + t_w] += mask

    out = out / weight.clamp(min=1e-8)
    return tensor2img([out], rgb2bgr=True)


def _to_unit_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
    return t


def compute_paired_metrics(sr_bgr: np.ndarray, gt_bgr: np.ndarray, lpips_fn) -> dict:
    if sr_bgr.shape != gt_bgr.shape:
        sr_bgr = cv2.resize(sr_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))

    out = {
        "psnr": float(psnr_fn(gt_bgr, sr_bgr, data_range=255)),
        "ssim": float(
            ssim_fn(
                cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY),
                data_range=255,
            )
        ),
    }
    if lpips_fn is not None:
        with torch.no_grad():
            t1 = _to_unit_tensor(sr_bgr) * 2 - 1
            t2 = _to_unit_tensor(gt_bgr) * 2 - 1
            out["lpips"] = float(lpips_fn(t1, t2).item())
    return out


def init_iqa_metrics(metric_names: list[str]) -> dict:
    if not PYIQA_OK:
        print("[warn] pyiqa not installed; skipping no-reference metrics")
        return {}
    metrics = {}
    for name in metric_names:
        try:
            metrics[name] = pyiqa.create_metric(name, device="cuda")
        except Exception as e:
            print(f"[warn] pyiqa metric {name} failed to init: {e}")
    return metrics


def compute_iqa(img_bgr: np.ndarray, metrics: dict) -> dict:
    if not metrics:
        return {}
    t = _to_unit_tensor(img_bgr)
    out = {}
    for name, fn in metrics.items():
        try:
            with torch.no_grad():
                v = fn(t)
            out[name] = float(v.item() if hasattr(v, "item") else v)
        except Exception as e:
            out[name] = None
            print(f"[warn] {name} failed: {e}")
    return out


def aggregate(values: list[dict]) -> dict:
    if not values:
        return {}
    keys = sorted({k for d in values for k in d.keys() if d.get(k) is not None})
    summary = {"count": len(values)}
    for k in keys:
        v = [d[k] for d in values if d.get(k) is not None]
        if v:
            summary[f"{k}_mean"] = float(np.mean(v))
            summary[f"{k}_std"] = float(np.std(v))
    return summary


def run_paired(model, paired_dir: Path, lpips_fn) -> tuple[list, dict]:
    gt_dir = paired_dir / "GT"
    lq_dir = paired_dir / "LQ"
    if not gt_dir.exists() or not lq_dir.exists():
        print(f"[skip] paired dir missing GT/LQ: {paired_dir}")
        return [], {}

    rows = []
    for lq_path in sorted(lq_dir.iterdir()):
        if not lq_path.is_file():
            continue
        gt_path = gt_dir / lq_path.name
        if not gt_path.exists():
            continue
        lq = cv2.imread(str(lq_path))
        gt = cv2.imread(str(gt_path))
        if lq is None or gt is None:
            continue
        sr = hat_upscale(model, lq)
        m = compute_paired_metrics(sr, gt, lpips_fn)
        m["filename"] = lq_path.name
        rows.append(m)
        print(f"  [paired] {lq_path.name}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}"
              + (f"  LPIPS={m['lpips']:.4f}" if "lpips" in m else ""))
    return rows, aggregate([{k: v for k, v in r.items() if k != "filename"} for r in rows])


def run_no_ref(model, hist_dir: Path, iqa_metrics: dict) -> tuple[list, dict]:
    if not hist_dir.exists():
        print(f"[skip] historical dir missing: {hist_dir}")
        return [], {}
    rows = []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(hist_dir.iterdir()) if p.suffix.lower() in exts]
    if not paths:
        print(f"[skip] no images in {hist_dir}")
        return [], {}
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        sr = hat_upscale(model, img)
        m = compute_iqa(sr, iqa_metrics)
        m["filename"] = path.name
        rows.append(m)
        kv = "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "filename" and v is not None)
        print(f"  [hist] {path.name}  {kv}")
    return rows, aggregate([{k: v for k, v in r.items() if k != "filename"} for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--params-key", choices=["params", "params_ema"], default=None)
    ap.add_argument("--tag", required=True, help="Output basename, e.g. baseline_180k")
    ap.add_argument("--paired", default=str(REPO_ROOT / "tests" / "synthetic"))
    ap.add_argument("--no-ref", default=str(REPO_ROOT / "tests" / "historical"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--iqa-metrics", default="maniqa,musiq,clipiqa")
    args = ap.parse_args()

    print(f"[load] {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, args.params_key)

    lpips_fn = None
    if LPIPS_OK:
        print("[load] LPIPS (alex)")
        lpips_fn = lpips_pkg.LPIPS(net="alex").cuda()

    iqa_names = [s.strip() for s in args.iqa_metrics.split(",") if s.strip()]
    iqa_metrics = init_iqa_metrics(iqa_names)

    t0 = time.time()
    print("\n=== Paired (synthetic) ===")
    paired_rows, paired_summary = run_paired(model, Path(args.paired), lpips_fn)

    print("\n=== No-reference (historical) ===")
    hist_rows, hist_summary = run_no_ref(model, Path(args.no_ref), iqa_metrics)

    out = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "params_key": args.params_key,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "paired": {"summary": paired_summary, "rows": paired_rows},
        "no_ref": {"summary": hist_summary, "rows": hist_rows},
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] {out_path}")
    print("paired:", paired_summary)
    print("no-ref:", hist_summary)


if __name__ == "__main__":
    main()
