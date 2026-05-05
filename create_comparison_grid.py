"""
🖼️ Create Comparison Grid - Vizualne usporedbe za znanstveni rad

Kreira side-by-side figure s zoom-in detaljima

Korištenje:
    python create_comparison_grid.py --input photos_and_restaured/old_photo_99.webp \
        --methods_dir comparison_results/ --output figures/
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import os


def create_zoom_crop(img, center_x, center_y, crop_size=200, scale=2):
    """Kreira zoom-in crop određenog dijela slike."""
    h, w = img.shape[:2]
    
    # Izračunaj koordinate
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    crop = img[y1:y2, x1:x2]
    
    # Uvećaj
    zoomed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    return zoomed, (x1, y1, x2, y2)


def add_label(img, text, position='top', font_scale=0.8, thickness=2):
    """Dodaje tekstualnu oznaku na sliku."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    if position == 'top':
        x = (img.shape[1] - text_size[0]) // 2
        y = 30
    else:
        x = (img.shape[1] - text_size[0]) // 2
        y = img.shape[0] - 10
    
    # Pozadina za bolju čitljivost
    cv2.rectangle(img, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return img


def create_comparison_figure(original_path, methods_dict, output_path, 
                            zoom_coords=None, zoom_size=150):
    """
    Kreira usporednu figuru.
    
    Args:
        original_path: Putanja do originalne slike
        methods_dict: {method_name: image_path}
        output_path: Putanja za spremanje
        zoom_coords: (x, y) - centar za zoom (npr. lice)
        zoom_size: Veličina zoom cropa
    """
    # Učitaj original
    original = cv2.imread(original_path)
    if original is None:
        print(f"❌ Ne mogu učitati: {original_path}")
        return
    
    # Učitaj sve metode
    images = {'Original (LQ)': original}
    for name, path in methods_dict.items():
        img = cv2.imread(path)
        if img is not None:
            images[name] = img
    
    # Nađi zajedničku veličinu (uzmi najveću)
    max_h = max(img.shape[0] for img in images.values())
    max_w = max(img.shape[1] for img in images.values())
    
    # Resize sve na istu veličinu
    resized_images = {}
    for name, img in images.items():
        if img.shape[0] != max_h or img.shape[1] != max_w:
            resized = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = img
        resized_images[name] = resized
    
    # Dodaj labele
    labeled_images = []
    for name, img in resized_images.items():
        labeled = add_label(img, name)
        labeled_images.append(labeled)
    
    # Horizontalna kombinacija
    if len(labeled_images) <= 3:
        grid = np.hstack(labeled_images)
    else:
        # 2 reda
        mid = len(labeled_images) // 2
        row1 = np.hstack(labeled_images[:mid])
        row2 = np.hstack(labeled_images[mid:])
        
        # Pad ako treba
        if row1.shape[1] != row2.shape[1]:
            diff = row1.shape[1] - row2.shape[1]
            if diff > 0:
                row2 = np.pad(row2, ((0, 0), (0, diff), (0, 0)), mode='constant')
            else:
                row1 = np.pad(row1, ((0, 0), (0, -diff), (0, 0)), mode='constant')
        
        grid = np.vstack([row1, row2])
    
    # Spremi
    cv2.imwrite(output_path, grid)
    print(f"✅ Usporedna figura spremljena: {output_path}")
    
    # Zoom figure ako su koordinate zadane
    if zoom_coords:
        create_zoom_comparison(resized_images, zoom_coords, zoom_size, 
                              output_path.replace('.png', '_zoom.png'))
    
    return grid


def create_zoom_comparison(images_dict, center_coords, crop_size, output_path):
    """Kreira usporedbu zoom-in detalja."""
    zoom_images = []
    
    for name, img in images_dict.items():
        # Skaliraj koordinate za ovu sliku
        scale_x = img.shape[1] / list(images_dict.values())[0].shape[1]
        scale_y = img.shape[0] / list(images_dict.values())[0].shape[0]
        
        cx = int(center_coords[0] * scale_x) if scale_x != 1 else center_coords[0]
        cy = int(center_coords[1] * scale_y) if scale_y != 1 else center_coords[1]
        
        zoomed, _ = create_zoom_crop(img, cx, cy, crop_size, scale=3)
        zoomed = add_label(zoomed, name, font_scale=0.6)
        zoom_images.append(zoomed)
    
    # Spoji horizontalno
    grid = np.hstack(zoom_images)
    
    cv2.imwrite(output_path, grid)
    print(f"✅ Zoom usporedba spremljena: {output_path}")


def batch_create_figures(input_dir, methods_dirs, output_dir):
    """Kreira figure za sve slike u direktoriju."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
    
    for img_path in files:
        print(f"\n📸 Kreiram figuru za: {img_path.name}")
        
        # Pronađi sve verzije
        methods = {}
        for method_name, method_dir in methods_dirs.items():
            method_path = Path(method_dir) / f"{img_path.stem}.png"
            if method_path.exists():
                methods[method_name] = str(method_path)
        
        if methods:
            output_path = str(output_dir / f"{img_path.stem}_comparison.png")
            create_comparison_figure(str(img_path), methods, output_path)


def main():
    parser = argparse.ArgumentParser(description="Create Comparison Figures")
    parser.add_argument("--input", type=str, required=True, help="Originalna slika ili direktorij")
    parser.add_argument("--hat_gan", type=str, help="HAT-GAN rezultat")
    parser.add_argument("--realesrgan", type=str, help="Real-ESRGAN rezultat")
    parser.add_argument("--swinir", type=str, help="SwinIR rezultat")
    parser.add_argument("--ultimate", type=str, help="Ultimate pipeline rezultat")
    parser.add_argument("--output", type=str, required=True, help="Izlazna slika ili direktorij")
    parser.add_argument("--zoom_x", type=int, help="X koordinata za zoom")
    parser.add_argument("--zoom_y", type=int, help="Y koordinata za zoom")
    
    args = parser.parse_args()
    
    # Skupi metode
    methods = {}
    if args.hat_gan:
        methods['HAT-GAN'] = args.hat_gan
    if args.realesrgan:
        methods['Real-ESRGAN'] = args.realesrgan
    if args.swinir:
        methods['SwinIR'] = args.swinir
    if args.ultimate:
        methods['Ultimate'] = args.ultimate
    
    zoom_coords = None
    if args.zoom_x and args.zoom_y:
        zoom_coords = (args.zoom_x, args.zoom_y)
    
    create_comparison_figure(args.input, methods, args.output, zoom_coords)


if __name__ == "__main__":
    main()
