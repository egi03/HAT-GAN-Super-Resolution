# Historical Photo Restoration with HAT-GAN

4x super-resolution and end-to-end restoration of historical photographs (19th
and early-20th century) using a Hybrid Attention Transformer (HAT-L) generator
fine-tuned with a GAN objective on a domain-specific degradation pipeline.

The model is trained to **reconstruct missing detail while preserving authentic
historical patina** (sepia, film grain, chemical fading) instead of treating
those cues as noise.

## Pipeline

```
input (low-res, aged scan)
  -> HAT-GAN  4x super-resolution (RGB, tiled)
  -> DDColor  colorization                  (optional)
  -> GFPGAN / CodeFormer  face restoration  (optional)
  -> unsharp mask + CLAHE                   (optional)
output (high-res, restored)
```

All stages are toggleable through CLI flags so each component's contribution
can be ablated independently.

## Architecture

- **Generator:** HAT-L
  (`upscale=4`, `embed_dim=180`, 12 residual hybrid-attention groups,
  `compress_ratio=2`, `squeeze_factor=30`, `window_size=16`).
- **Discriminator:** UNet with spectral normalization.
- **Losses:** Charbonnier (pixel) + VGG19 perceptual + LSGAN.
- **Training data:** HR images degraded on-the-fly by a custom
  `HistoricalDegradationDataset` (vignetting, chemical fading, foxing,
  scratches, sensor noise, JPEG artifacts; sepia tone applied stochastically
  to preserve color statistics of the original).
- **Hardware:** trained on a single NVIDIA RTX 4070 Super (12 GB VRAM).

## Results

50-image held-out synthetic test set, 4x super-resolution, full-reference
metrics (mean +/- std):

| Method   |    PSNR (dB)   |       SSIM      |      LPIPS      |
|----------|----------------|-----------------|-----------------|
| Bicubic  | 20.69 +/- 3.23 | 0.663 +/- 0.154 | 0.588 +/- 0.165 |
| HAT-GAN  | 19.30 +/- 2.31 | 0.615 +/- 0.141 | 0.546 +/- 0.152 |

Lower PSNR/SSIM is expected for GAN-based restoration: the model trades
pixel-wise fidelity for perceptual quality (lower LPIPS, sharper textures,
recovered face detail). Raw per-image numbers are in `results/`.

## Repository layout

```
.
├── BasicSR/                   # customized BasicSR fork (model, training framework)
│   └── basicsr/
│       ├── archs/hat_arch.py                         # HAT-L architecture
│       ├── data/historical_degradation_dataset.py    # custom degradation dataset
│       └── degradation_machine.py                    # physics-based aging pipeline
├── ultimate_restore.py        # canonical inference CLI (SR + color + face + post)
├── colorizer.py               # DDColor wrapper used by the pipeline
├── evaluate.py                # PSNR / SSIM / LPIPS evaluation
├── results/                   # measured baseline metrics (JSON)
├── tests/                     # held-out test-set manifests
└── LICENSE
```

## Inference

```bash
python ultimate_restore.py \
    --input  path/to/old_photo.jpg \
    --output path/to/restored.png \
    --checkpoint path/to/net_g.pth \
    --face-method gfpgan \
    --colorize
```

Flags:

| flag | purpose |
|------|---------|
| `--checkpoint`        | path to a trained HAT-L generator (`.pth`) |
| `--params-key`        | `params` or `params_ema` (auto-detected if omitted) |
| `--face-method`       | `gfpgan`, `codeformer`, or `none` |
| `--codeformer-fidelity` | 0.0 (max quality) -- 1.0 (max fidelity); 0.7-0.9 recommended for historical photos |
| `--colorize`          | enable DDColor B&W -> color stage |
| `--no-postprocess`    | disable unsharp mask + CLAHE |
| `--sharpen` / `--clahe` | tune post-processing strength |

The first `--colorize` call downloads ~600 MB of DDColor weights to
`~/.cache/modelscope`.

## Evaluation

```bash
# directory mode
python evaluate.py --restored_dir restored/ --gt_dir ground_truth/ --output results.json

# single pair
python evaluate.py --image1 restored.png --image2 gt.png
```

## Dependencies

Python 3.11+ with CUDA-enabled PyTorch. Core packages:

```
torch torchvision opencv-python numpy scikit-image lpips
basicsr (vendored under BasicSR/)
gfpgan        # optional, for --face-method gfpgan
codeformer-pip # optional, for --face-method codeformer
modelscope    # optional, for --colorize (DDColor)
```

## Acknowledgements

This work builds on:

- HAT (Hybrid Attention Transformer) -- https://github.com/XPixelGroup/HAT
- BasicSR -- https://github.com/XPixelGroup/BasicSR
- GFPGAN -- https://github.com/TencentARC/GFPGAN
- CodeFormer -- https://github.com/sczhou/CodeFormer
- DDColor -- https://github.com/piddnad/DDColor
