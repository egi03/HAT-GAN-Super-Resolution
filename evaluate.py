"""
📊 Evaluation Script - PSNR, SSIM, LPIPS metrike

Korištenje:
    python evaluate.py --restored_dir results/ --gt_dir datasets/val/GT
    python evaluate.py --image1 result.png --image2 gt.png
"""

import torch
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

# Metrike
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️ LPIPS nije instaliran. Instaliraj: pip install lpips")


class ImageMetrics:
    def __init__(self, use_lpips=True):
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        
        if self.use_lpips:
            print("📊 Učitavam LPIPS model...")
            self.lpips_fn = lpips.LPIPS(net='alex').cuda()
            print("  ✓ LPIPS spreman")
    
    def calculate_psnr(self, img1, img2):
        """Peak Signal-to-Noise Ratio (veće = bolje)"""
        return psnr(img1, img2, data_range=255)
    
    def calculate_ssim(self, img1, img2):
        """Structural Similarity Index (veće = bolje)"""
        # Konvertiraj u grayscale za SSIM
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        return ssim(gray1, gray2, data_range=255)
    
    def calculate_lpips(self, img1, img2):
        """Learned Perceptual Image Patch Similarity (manje = bolje)"""
        if not self.use_lpips:
            return None
        
        # Konvertiraj u tensor
        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
            return img
        
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        
        with torch.no_grad():
            lpips_val = self.lpips_fn(t1, t2)
        
        return lpips_val.item()
    
    def evaluate_pair(self, img1_path, img2_path):
        """Evaluira par slika."""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        # Osiguraj istu veličinu
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        results = {
            'psnr': self.calculate_psnr(img1, img2),
            'ssim': self.calculate_ssim(img1, img2),
        }
        
        if self.use_lpips:
            results['lpips'] = self.calculate_lpips(img1, img2)
        
        return results
    
    def evaluate_directory(self, restored_dir, gt_dir, pattern='*'):
        """Evaluira sve slike u direktoriju."""
        restored_dir = Path(restored_dir)
        gt_dir = Path(gt_dir)
        
        all_results = []
        
        # Pronađi parove
        for restored_path in restored_dir.glob(pattern):
            if not restored_path.is_file():
                continue
            
            # Pronađi GT
            gt_path = gt_dir / restored_path.name
            if not gt_path.exists():
                # Pokušaj bez sufiksa
                stem = restored_path.stem.replace('_restored', '').replace('_ultimate', '')
                for ext in ['.png', '.jpg', '.jpeg']:
                    gt_path = gt_dir / f"{stem}{ext}"
                    if gt_path.exists():
                        break
            
            if not gt_path.exists():
                print(f"  ⚠️ GT ne postoji za: {restored_path.name}")
                continue
            
            print(f"  📊 Evaluiram: {restored_path.name}")
            results = self.evaluate_pair(str(restored_path), str(gt_path))
            
            if results:
                results['filename'] = restored_path.name
                all_results.append(results)
        
        return all_results
    
    def summarize(self, results):
        """Izračunava prosjeke."""
        if not results:
            return {}
        
        summary = {
            'count': len(results),
            'psnr_mean': np.mean([r['psnr'] for r in results]),
            'psnr_std': np.std([r['psnr'] for r in results]),
            'ssim_mean': np.mean([r['ssim'] for r in results]),
            'ssim_std': np.std([r['ssim'] for r in results]),
        }
        
        if self.use_lpips and results[0].get('lpips') is not None:
            summary['lpips_mean'] = np.mean([r['lpips'] for r in results])
            summary['lpips_std'] = np.std([r['lpips'] for r in results])
        
        return summary
    
    def print_results(self, summary, title="Rezultati"):
        """Lijepo formatira rezultate."""
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        print(f"  Broj slika: {summary.get('count', 0)}")
        print(f"  PSNR:  {summary.get('psnr_mean', 0):.2f} ± {summary.get('psnr_std', 0):.2f} dB")
        print(f"  SSIM:  {summary.get('ssim_mean', 0):.4f} ± {summary.get('ssim_std', 0):.4f}")
        if 'lpips_mean' in summary:
            print(f"  LPIPS: {summary.get('lpips_mean', 0):.4f} ± {summary.get('lpips_std', 0):.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluacija kvalitete restauracije")
    
    # Batch mode
    parser.add_argument("--restored_dir", type=str, help="Direktorij s restauriranim slikama")
    parser.add_argument("--gt_dir", type=str, help="Direktorij s GT slikama")
    
    # Single pair mode
    parser.add_argument("--image1", type=str, help="Prva slika (restaurirana)")
    parser.add_argument("--image2", type=str, help="Druga slika (GT)")
    
    # Options
    parser.add_argument("--no-lpips", action="store_true", help="Preskoči LPIPS")
    parser.add_argument("--output", type=str, help="Spremi rezultate u JSON")
    
    args = parser.parse_args()
    
    metrics = ImageMetrics(use_lpips=not args.no_lpips)
    
    if args.image1 and args.image2:
        # Single pair
        results = metrics.evaluate_pair(args.image1, args.image2)
        print(f"\n📊 Rezultati:")
        print(f"  PSNR:  {results['psnr']:.2f} dB")
        print(f"  SSIM:  {results['ssim']:.4f}")
        if results.get('lpips'):
            print(f"  LPIPS: {results['lpips']:.4f}")
    
    elif args.restored_dir and args.gt_dir:
        # Batch
        print(f"\n📂 Evaluiram direktorij: {args.restored_dir}")
        results = metrics.evaluate_directory(args.restored_dir, args.gt_dir)
        summary = metrics.summarize(results)
        metrics.print_results(summary)
        
        if args.output:
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'restored_dir': args.restored_dir,
                'gt_dir': args.gt_dir,
                'summary': summary,
                'details': results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Rezultati spremljeni: {args.output}")
    
    else:
        print("Korištenje:")
        print("  python evaluate.py --restored_dir results/ --gt_dir datasets/val/GT")
        print("  python evaluate.py --image1 result.png --image2 gt.png")


if __name__ == "__main__":
    main()
