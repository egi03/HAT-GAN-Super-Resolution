"""
SOTA comparison harness — produces a publication-ready CSV.

Methods are registered in METHOD_REGISTRY below. Each method takes a BGR uint8
LR image and returns a BGR uint8 SR image. Methods that fail to load print a
warning and are excluded from the CSV (so a missing checkpoint doesn't tank
the whole run).

Metrics (all optional except PSNR/SSIM):
  paired:   PSNR, SSIM, LPIPS, DISTS
  no-ref:   MANIQA, MUSIQ, CLIPIQA  (via pyiqa)

Usage:
    python compare_sota.py --input tests/synthetic --output results/sota_synth.csv
    python compare_sota.py --input tests/historical --output results/sota_hist.csv --no-ref-only
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "BasicSR"))
sys.path.insert(0, str(REPO_ROOT))

from basicsr.archs.hat_arch import HAT  # noqa: E402
from basicsr.utils import img2tensor, tensor2img  # noqa: E402

from skimage.metrics import peak_signal_noise_ratio as psnr_fn  # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402

# ---------------------------------------------------------------------------
# Metric loaders
# ---------------------------------------------------------------------------

try:
    import lpips as lpips_pkg
except ImportError:
    lpips_pkg = None

try:
    import pyiqa
except ImportError:
    pyiqa = None


def make_lpips():
    if lpips_pkg is None:
        return None
    return lpips_pkg.LPIPS(net="alex").cuda()


def make_pyiqa(name):
    if pyiqa is None:
        return None
    try:
        return pyiqa.create_metric(name, device="cuda")
    except Exception as e:
        print(f"[warn] pyiqa metric {name} unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# HAT loaders
# ---------------------------------------------------------------------------

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


def load_hat(ckpt_path: str, params_key: str | None = None):
    if not Path(ckpt_path).exists():
        print(f"[skip] checkpoint missing: {ckpt_path}")
        return None
    model = build_hat_l()
    state = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    key = params_key or ("params_ema" if "params_ema" in state else "params")
    if key not in state:
        print(f"[skip] {ckpt_path}: no key '{key}'")
        return None
    model.load_state_dict(state[key], strict=True)
    return model.eval().cuda()


@torch.no_grad()
def hat_upscale(model, img_bgr, tile_size=400, tile_pad=32):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = img2tensor([img_rgb / 255.0], bgr2rgb=False, float32=True)[0].unsqueeze(0).cuda()
    b, c, h, w = img_tensor.size()
    scale = 4
    out = torch.zeros((b, c, h * scale, w * scale), device="cuda")
    weight = torch.zeros_like(out)
    stride = max(1, tile_size - tile_pad * 2)
    h_idx = sorted(set(list(range(0, max(1, h - tile_size), stride)) + [max(0, h - tile_size)]))
    w_idx = sorted(set(list(range(0, max(1, w - tile_size), stride)) + [max(0, w - tile_size)]))
    for y in h_idx:
        for x in w_idx:
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


# ---------------------------------------------------------------------------
# Method registry — each entry is a callable (loaded once) returning SR BGR
# ---------------------------------------------------------------------------

def method_bicubic():
    def f(lr_bgr):
        h, w = lr_bgr.shape[:2]
        return cv2.resize(lr_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    return f


def method_hat(ckpt_path, params_key=None):
    model = load_hat(ckpt_path, params_key)
    if model is None:
        return None
    return lambda lr: hat_upscale(model, lr)


def method_realesrgan():
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model=net, tile=400, tile_pad=10, pre_pad=0, half=False,
        )

        def f(lr):
            sr, _ = upsampler.enhance(lr, outscale=4)
            return sr
        return f
    except Exception as e:
        print(f"[skip] Real-ESRGAN unavailable: {e}")
        return None


def method_swinir(ckpt_path):
    """SwinIR-Real-GAN. User must download the checkpoint first."""
    if not Path(ckpt_path).exists():
        print(f"[skip] SwinIR checkpoint missing: {ckpt_path}")
        return None
    try:
        # The official SwinIR test script uses its own SwinIR class. To avoid
        # bundling another large arch into BasicSR, we rely on it being on
        # PYTHONPATH (clone JingyunLiang/SwinIR) and the user wiring it up.
        # Falling back to no-op until user provides a working loader.
        print("[skip] SwinIR loader not implemented — clone SwinIR repo and edit method_swinir")
        return None
    except Exception as e:
        print(f"[skip] SwinIR unavailable: {e}")
        return None


def method_bsrgan(ckpt_path):
    if not Path(ckpt_path).exists():
        print(f"[skip] BSRGAN checkpoint missing: {ckpt_path}")
        return None
    try:
        # BSRGAN uses the same RRDB arch as ESRGAN — 23 blocks, sf=4
        from basicsr.archs.rrdbnet_arch import RRDBNet
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(ckpt_path, map_location="cuda")
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        net.load_state_dict(state, strict=True)
        net.eval().cuda()

        @torch.no_grad()
        def f(lr_bgr):
            rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            sr = net(t).clamp(0, 1)
            sr_np = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        return f
    except Exception as e:
        print(f"[skip] BSRGAN unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def to_unit_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()


def compute_metrics(sr, gt, nr_metrics, lpips_fn, dists_fn):
    """gt may be None (no-reference mode)."""
    out = {}
    if gt is not None:
        if sr.shape != gt.shape:
            sr_for_paired = cv2.resize(sr, (gt.shape[1], gt.shape[0]))
        else:
            sr_for_paired = sr
        out["psnr"] = float(psnr_fn(gt, sr_for_paired, data_range=255))
        out["ssim"] = float(ssim_fn(
            cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(sr_for_paired, cv2.COLOR_BGR2GRAY),
            data_range=255,
        ))
        if lpips_fn is not None:
            t1 = to_unit_tensor(sr_for_paired) * 2 - 1
            t2 = to_unit_tensor(gt) * 2 - 1
            with torch.no_grad():
                out["lpips"] = float(lpips_fn(t1, t2).item())
        if dists_fn is not None:
            t1 = to_unit_tensor(sr_for_paired)
            t2 = to_unit_tensor(gt)
            with torch.no_grad():
                out["dists"] = float(dists_fn(t1, t2).item())

    sr_t = to_unit_tensor(sr)
    for name, fn in nr_metrics.items():
        if fn is None:
            continue
        try:
            with torch.no_grad():
                v = fn(sr_t)
            out[name] = float(v.item() if hasattr(v, "item") else v)
        except Exception as e:
            print(f"[warn] {name} failed: {e}")
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    return {k: float(np.mean([r[k] for r in rows if isinstance(r.get(k), (int, float))])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dir with LR images. If paired, expects subdirs LQ/ and GT/.")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--no-ref-only", action="store_true",
                    help="Treat --input as a flat dir of historical LR images (no GT).")
    ap.add_argument("--hat-v1", default="BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_180000.pth")
    ap.add_argument("--hat-v1-late", default="BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_240000.pth")
    ap.add_argument("--hat-v2", default="BasicSR/experiments/HAT_GAN_Historical_v2/models/net_g_latest.pth")
    ap.add_argument("--hat-psnr", default="experiments/pretrained_models/HAT_L_PSNR.pth")
    ap.add_argument("--swinir-ckpt", default="experiments/pretrained_models/SwinIR_x4.pth")
    ap.add_argument("--bsrgan-ckpt", default="experiments/pretrained_models/BSRGAN.pth")
    args = ap.parse_args()

    print("[load] methods")
    method_specs = [
        ("Bicubic", method_bicubic()),
        ("Real-ESRGAN", method_realesrgan()),
        ("BSRGAN", method_bsrgan(args.bsrgan_ckpt)),
        ("SwinIR-Real", method_swinir(args.swinir_ckpt)),
        ("HAT-PSNR", method_hat(args.hat_psnr)),
        ("HAT-GAN-v1-180k", method_hat(args.hat_v1)),
        ("HAT-GAN-v1-240k", method_hat(args.hat_v1_late)),
        ("HAT-GAN-v2", method_hat(args.hat_v2)),
    ]
    methods = [(n, f) for n, f in method_specs if f is not None]
    print(f"[load] active methods: {[n for n, _ in methods]}")

    print("[load] metrics")
    lpips_fn = None if args.no_ref_only else make_lpips()
    dists_fn = None if args.no_ref_only else make_pyiqa("dists")
    nr_metrics = {
        "maniqa": make_pyiqa("maniqa"),
        "musiq": make_pyiqa("musiq"),
        "clipiqa": make_pyiqa("clipiqa"),
    }

    in_dir = Path(args.input)
    if args.no_ref_only:
        lq_dir, gt_dir = in_dir, None
    else:
        lq_dir = in_dir / "LQ"
        gt_dir = in_dir / "GT"
        if not lq_dir.exists():
            # treat input as flat dir — must be no-ref
            lq_dir = in_dir
            gt_dir = None
            print("[info] no LQ/GT subdirs; running in no-reference mode")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    lq_paths = sorted(p for p in lq_dir.iterdir() if p.suffix.lower() in exts)
    if not lq_paths:
        raise SystemExit(f"No images under {lq_dir}")

    rows = []
    t0 = time.time()
    for path in lq_paths:
        lr = cv2.imread(str(path))
        if lr is None:
            continue
        gt = None
        if gt_dir is not None:
            gt_path = gt_dir / path.name
            if gt_path.exists():
                gt = cv2.imread(str(gt_path))
        for name, fn in methods:
            try:
                sr = fn(lr)
            except Exception as e:
                print(f"[err] {name} on {path.name}: {e}")
                continue
            metrics = compute_metrics(sr, gt, nr_metrics, lpips_fn, dists_fn)
            metrics.update({"method": name, "filename": path.name})
            rows.append(metrics)
            print(f"  {name:18s} {path.name}  " +
                  "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                            if isinstance(v, (int, float))))

    # Aggregate per method
    methods_seen = []
    for r in rows:
        if r["method"] not in methods_seen:
            methods_seen.append(r["method"])

    summary = []
    for m in methods_seen:
        method_rows = [r for r in rows if r["method"] == m]
        agg = aggregate(method_rows)
        agg["method"] = m
        agg["count"] = len(method_rows)
        summary.append(agg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = ["psnr", "ssim", "lpips", "dists", "maniqa", "musiq", "clipiqa"]
    cols = ["method", "count"] + [m for m in metric_cols if any(m in r for r in summary)]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in summary:
            writer.writerow({c: round(r[c], 4) if isinstance(r.get(c), (int, float)) else r.get(c, "")
                             for c in cols})

    detail_path = out_path.with_suffix(".per_image.csv")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        all_keys = sorted({k for r in rows for k in r.keys()})
        ordered = ["method", "filename"] + [k for k in all_keys if k not in {"method", "filename"}]
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[done] {out_path}  ({len(summary)} methods, {len(rows)} rows, {time.time() - t0:.1f}s)")
    print(f"       per-image -> {detail_path}")


if __name__ == "__main__":
    main()
