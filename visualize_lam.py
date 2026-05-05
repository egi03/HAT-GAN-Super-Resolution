"""
LAM (Local Attribution Maps) Vizualizacija za HAT-L model

Generira heatmapu koja pokazuje koje piksele ulazne slike
model koristi za rekonstrukciju određenog izlaznog piksela.

Korištenje:
    python visualize_lam.py --input photos_and_restaured/old_photo_18.jpg --output lam_result.png
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

from basicsr.archs.hat_arch import HAT


def create_hat_model(model_path):
    """Inicijalizira HAT-L model s 12 grupa."""
    model = HAT(
        upscale=4, 
        in_chans=3, 
        img_size=64, 
        window_size=16,
        compress_ratio=2, 
        squeeze_factor=30, 
        conv_scale=0.01,
        overlap_ratio=0.5, 
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # 12 grupa za HAT-L
        embed_dim=180, 
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],  # 12 grupa
        upsampler='pixelshuffle'
    )
    
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
    key = 'params_ema' if 'params_ema' in checkpoint else 'params'
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval().cuda()
    
    return model


def compute_lam(model, img_tensor, target_y, target_x, channel=1):
    """
    Izračunava Local Attribution Map za specifičan piksel.
    
    Args:
        model: HAT model
        img_tensor: Ulazni tensor [1, 3, H, W]
        target_y, target_x: Koordinate ciljnog piksela u HR slici
        channel: RGB kanal za analizu (0=R, 1=G, 2=B)
    
    Returns:
        attribution_map: Normalizirana mapa doprinosa [H, W]
    """
    img_tensor = img_tensor.clone().requires_grad_(True)
    
    # Forward pass
    output = model(img_tensor)
    
    # Odabir ciljnog piksela
    target_value = output[0, channel, target_y, target_x]
    
    # Backward pass za gradijente
    model.zero_grad()
    target_value.backward()
    
    # Gradijenti kao mjera doprinosa
    grad = img_tensor.grad.data.abs().cpu().numpy()[0]
    
    # Agregacija preko kanala
    attribution_map = np.sum(grad, axis=0)
    
    # Normalizacija
    if attribution_map.max() > attribution_map.min():
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
    
    return attribution_map


def create_visualization(img_lr, attribution_map, target_y, target_x, scale=4):
    """
    Stvara vizualizaciju s originalnom slikom i LAM heatmapom.
    """
    # Crop attribution map da odgovara slici (zbog paddinga)
    h, w = img_lr.shape[:2]
    attribution_map = attribution_map[:h, :w]
    
    # Custom colormap (plavo -> crveno)
    colors = ['#000033', '#0000ff', '#00ffff', '#ffff00', '#ff0000', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('lam', colors, N=256)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Originalna LR slika s označenom metom
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB))
    # Meta u LR rezoluciji
    lr_y, lr_x = target_y // scale, target_x // scale
    ax1.scatter([lr_x], [lr_y], c='red', s=200, marker='x', linewidths=3, label='Target pixel (LR)')
    ax1.set_title('Ulazna LR slika\n(crveni X = ciljni piksel)', fontsize=14)
    ax1.legend()
    ax1.axis('off')
    
    # 2. LAM Heatmapa
    ax2 = axes[1]
    im = ax2.imshow(attribution_map, cmap=cmap)
    ax2.scatter([lr_x], [lr_y], c='white', s=200, marker='x', linewidths=3)
    ax2.set_title('LAM Heatmapa\n(svijetlo = jači doprinos)', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Overlay
    ax3 = axes[2]
    img_rgb = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    # Normaliziraj sliku za overlay
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # Heatmapa kao overlay
    heatmap_colored = plt.cm.jet(attribution_map)[:, :, :3]
    overlay = img_norm * 0.5 + heatmap_colored * 0.5
    
    ax3.imshow(overlay)
    ax3.scatter([lr_x], [lr_y], c='white', s=200, marker='x', linewidths=3)
    ax3.set_title('Overlay\n(LAM na originalnoj slici)', fontsize=14)
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


def run_lam_visualization(model_path, input_path, output_path, target_coords=None):
    """
    Glavna funkcija za LAM vizualizaciju.
    
    Args:
        model_path: Put do .pth checkpointa
        input_path: Put do ulazne slike
        output_path: Put za spremanje vizualizacije
        target_coords: (y, x) koordinate u HR slici, ili None za sredinu
    """
    print(f"Učitavam model: {model_path}")
    model = create_hat_model(model_path)
    
    print(f"Učitavam sliku: {input_path}")
    img_lr_orig = cv2.imread(input_path)
    if img_lr_orig is None:
        raise FileNotFoundError(f"Ne mogu učitati: {input_path}")
    
    # Resize za LAM (backward pass zahtijeva puno memorije)
    max_size = 128  # Smanjen zbog VRAM ograničenja
    h_orig, w_orig = img_lr_orig.shape[:2]
    if max(h_orig, w_orig) > max_size:
        scale_factor = max_size / max(h_orig, w_orig)
        new_w = int(w_orig * scale_factor)
        new_h = int(h_orig * scale_factor)
        img_lr = cv2.resize(img_lr_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Resize: {w_orig}x{h_orig} -> {new_w}x{new_h} (za memoriju)")
    else:
        img_lr = img_lr_orig
    
    h, w = img_lr.shape[:2]
    scale = 4
    
    # Konverzija u tensor
    img_rgb = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.
    
    # Padding za HAT window size (mora biti djeljivo s 16)
    _, _, h_t, w_t = img_tensor.shape
    pad_h = (16 - h_t % 16) % 16
    pad_w = (16 - w_t % 16) % 16
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Ako koordinate nisu specificirane, koristi sredinu HR slike
    if target_coords is None:
        target_y = (h * scale) // 2
        target_x = (w * scale) // 2
    else:
        target_y, target_x = target_coords
    
    print(f"Računam LAM za piksel ({target_y}, {target_x}) u HR slici...")
    attribution_map = compute_lam(model, img_tensor, target_y, target_x)
    
    print("Generiram vizualizaciju...")
    fig = create_visualization(img_lr, attribution_map, target_y, target_x, scale)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ LAM vizualizacija spremljena: {output_path}")
    
    return attribution_map


def compute_receptive_field_stats(attribution_map, threshold=0.1):
    """
    Izračunava statistike receptivnog polja.
    
    Returns:
        dict: Statistike o veličini i obliku receptivnog polja
    """
    # Pikseli iznad praga
    active_pixels = attribution_map > threshold
    active_count = np.sum(active_pixels)
    total_pixels = attribution_map.size
    
    # Pronađi bounding box aktivnog područja
    rows = np.any(active_pixels, axis=1)
    cols = np.any(active_pixels, axis=0)
    
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox_height = ymax - ymin + 1
        bbox_width = xmax - xmin + 1
    else:
        bbox_height = bbox_width = 0
    
    return {
        "active_pixels": int(active_count),
        "total_pixels": int(total_pixels),
        "coverage_percent": float(active_count / total_pixels * 100),
        "bbox_height": int(bbox_height),
        "bbox_width": int(bbox_width),
        "effective_receptive_field": f"{bbox_width}x{bbox_height}"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAM Vizualizacija za HAT-L model")
    parser.add_argument("--input", type=str, default="photos_and_restaured/old_photo_18.jpg",
                        help="Put do ulazne slike")
    parser.add_argument("--output", type=str, default="lam_visualization.png",
                        help="Put za izlaznu vizualizaciju")
    parser.add_argument("--model", type=str, 
                        default="BasicSR/experiments/HAT_GAN_Historical_v1/models/net_g_180000.pth",
                        help="Put do modela")
    parser.add_argument("--target_y", type=int, default=None, help="Y koordinata cilja u HR")
    parser.add_argument("--target_x", type=int, default=None, help="X koordinata cilja u HR")
    
    args = parser.parse_args()
    
    target = None
    if args.target_y is not None and args.target_x is not None:
        target = (args.target_y, args.target_x)
    
    attribution_map = run_lam_visualization(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        target_coords=target
    )
    
    # Ispiši statistike
    stats = compute_receptive_field_stats(attribution_map)
    print("\n📊 Statistike receptivnog polja:")
    print(f"   Aktivni pikseli: {stats['active_pixels']} / {stats['total_pixels']}")
    print(f"   Pokrivenost: {stats['coverage_percent']:.1f}%")
    print(f"   Efektivno receptivno polje: {stats['effective_receptive_field']}")