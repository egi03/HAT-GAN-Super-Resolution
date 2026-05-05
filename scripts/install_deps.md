# Extra dependencies for the new pipeline

Install once before running Phase 0/4/6 scripts:

```powershell
pip install pyiqa DISTS-pytorch lpips
# Phase 4 (CX-loss). The maintained PyTorch port is on GitHub:
pip install git+https://github.com/S-aiueo32/contextual_loss_pytorch
# Phase 5 (CodeFormer comparison)
pip install codeformer-pip
```

Then download the third-party SOTA checkpoints into
`experiments/pretrained_models/`:

| File                     | URL                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `BSRGAN.pth`             | https://github.com/cszn/BSRGAN/releases/download/v0.1.5/BSRGAN.pth                                 |
| `SwinIR_x4.pth`          | https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth |
| `RealESRGAN_x4plus.pth`  | auto-downloaded by realesrgan package on first use                                                  |
| `HAT_L_PSNR.pth`         | already present in experiments/pretrained_models/                                                   |

## Phase 4 contextual loss — note

The PyPI package `contextual-loss` is unmaintained and the upstream
`roimehrez/contextualLoss` repo is TF1. Use the PyTorch port
`S-aiueo32/contextual_loss_pytorch` (above). When wiring it into BasicSR,
register a new loss type analogous to `PerceptualLoss` so it picks up
`gan_gt_usm`-style options cleanly.
