"""
Canonical inference pipeline for HAT-GAN Historical.

Stages (each toggleable):
  1. HAT-GAN upscaling (RGB, tiled).
  2. Optional DDColor colorization (B&W / sepia -> color).
  3. Face restoration: GFPGAN | CodeFormer | none.
  4. Optional unsharp mask + CLAHE post-processing.

Usage:
    python ultimate_restore.py \
        --input photos_and_restaured/old_photo_99.webp \
        --output final_result.png \
        --checkpoint BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_180000.pth \
        --params-key params_ema \
        --face-method gfpgan \
        --colorize
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Make `from basicsr...` work whether or not BasicSR is pip-installed.
_REPO = Path(__file__).resolve().parent
if (_REPO / "BasicSR").exists() and str(_REPO / "BasicSR") not in sys.path:
    sys.path.insert(0, str(_REPO / "BasicSR"))

from basicsr.archs.hat_arch import HAT
from basicsr.utils import img2tensor, tensor2img

DEFAULT_CHECKPOINT = "BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_180000.pth"

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

try:
    from codeformer import CodeFormer  # codeformer-pip
    CODEFORMER_AVAILABLE = True
except ImportError:
    CODEFORMER_AVAILABLE = False


class UltimateRestorer:
    def __init__(
        self,
        hat_model_path: str = DEFAULT_CHECKPOINT,
        params_key: str | None = None,
        face_method: str = "gfpgan",
        codeformer_fidelity: float = 0.7,
        colorize: bool = False,
    ):
        print("[init] UltimateRestorer")
        print(f"  checkpoint = {hat_model_path}")
        print(f"  params_key = {params_key or 'auto'}")
        print(f"  face_method = {face_method}")
        print(f"  colorize    = {colorize}")

        self.hat_model = self._load_hat_model(hat_model_path, params_key)
        self.face_method = face_method
        self.gfpgan = None
        self.codeformer = None
        self.colorizer = None
        if colorize:
            from colorizer import Colorizer
            self.colorizer = Colorizer()

        if face_method == "gfpgan":
            if not GFPGAN_AVAILABLE:
                print("[warn] GFPGAN not installed; falling back to face_method=none")
                self.face_method = "none"
            else:
                self.gfpgan = GFPGANer(
                    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
        elif face_method == "codeformer":
            if not CODEFORMER_AVAILABLE:
                print("[warn] codeformer-pip not installed; falling back to face_method=none")
                self.face_method = "none"
            else:
                self.codeformer = CodeFormer(fidelity_weight=codeformer_fidelity)
        elif face_method != "none":
            raise ValueError(f"Unknown face_method: {face_method}")

    def _load_hat_model(self, model_path: str, params_key: str | None) -> HAT:
        model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=2, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5,
            depths=[6] * 12,
            embed_dim=180,
            num_heads=[6] * 12,
            upsampler="pixelshuffle",
        )
        ckpt = torch.load(model_path, map_location="cuda", weights_only=True)
        key = params_key or ("params_ema" if "params_ema" in ckpt else "params")
        if key not in ckpt:
            raise KeyError(f"checkpoint has no key '{key}'; keys = {list(ckpt.keys())}")
        model.load_state_dict(ckpt[key], strict=True)
        return model.eval().cuda()

    def _hat_upscale(self, img_bgr: np.ndarray, tile_size: int = 400, tile_pad: int = 32) -> np.ndarray:
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
                with torch.no_grad():
                    sr = self.hat_model(tile_p)
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

    def _enhance_faces(self, img: np.ndarray) -> np.ndarray:
        if self.face_method == "gfpgan" and self.gfpgan is not None:
            _, _, restored = self.gfpgan.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
            return restored
        if self.face_method == "codeformer" and self.codeformer is not None:
            return self.codeformer.enhance(img)
        return img

    def _post_process(self, img: np.ndarray, sharpen: float = 0.3, clahe_clip: float = 1.5) -> np.ndarray:
        gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
        sharpened = cv2.addWeighted(img, 1 + sharpen, gaussian, -sharpen, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def restore(
        self,
        input_path: str,
        output_path: str,
        use_postprocess: bool = True,
        sharpen: float = 0.3,
        clahe_clip: float = 1.5,
    ) -> np.ndarray:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(input_path)
        print(f"[in]  {input_path}  {img.shape[1]}x{img.shape[0]}")

        upscaled = self._hat_upscale(img)
        print(f"[sr]  {upscaled.shape[1]}x{upscaled.shape[0]}")

        if self.colorizer is not None:
            upscaled = self.colorizer.colorize(upscaled)
            print("[color] DDColor")

        if self.face_method != "none":
            upscaled = self._enhance_faces(upscaled)
            print(f"[face] {self.face_method}")

        if use_postprocess:
            upscaled = self._post_process(upscaled, sharpen, clahe_clip)
            print("[post] sharpen + CLAHE")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        cv2.imwrite(output_path, upscaled)
        print(f"[out] {output_path}")
        return upscaled


def main():
    ap = argparse.ArgumentParser(description="HAT-GAN Historical canonical inference")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--params-key", choices=["params", "params_ema"], default=None)
    ap.add_argument(
        "--face-method", choices=["none", "gfpgan", "codeformer"], default="gfpgan",
        help="Face restoration method. 'none' = pure model output (use for ablations)."
    )
    ap.add_argument("--codeformer-fidelity", type=float, default=0.7,
                    help="0.0=max quality, 1.0=max fidelity. 0.7-0.9 good for historical photos.")
    ap.add_argument("--no-faces", action="store_true",
                    help="Backwards-compat alias for --face-method none.")
    ap.add_argument("--colorize", action="store_true",
                    help="Run DDColor colorization between SR and face restoration. "
                         "First use downloads ~600 MB of weights via modelscope.")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--sharpen", type=float, default=0.3)
    ap.add_argument("--clahe", type=float, default=1.5)
    args = ap.parse_args()

    face_method = "none" if args.no_faces else args.face_method

    restorer = UltimateRestorer(
        hat_model_path=args.checkpoint,
        params_key=args.params_key,
        face_method=face_method,
        codeformer_fidelity=args.codeformer_fidelity,
        colorize=args.colorize,
    )
    restorer.restore(
        input_path=args.input,
        output_path=args.output,
        use_postprocess=not args.no_postprocess,
        sharpen=args.sharpen,
        clahe_clip=args.clahe,
    )


if __name__ == "__main__":
    main()
