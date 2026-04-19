"""
scripts/compress.py
Compress / decompress a single image with QUIBC.

Usage:
    python scripts/compress.py --input photo.jpg --weights checkpoints/best_model.weights.h5
    python scripts/compress.py --input photo.jpg --weights checkpoints/best_model.weights.h5 \
        --output_dir results/
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from quibc.deployment import load_quibc
from quibc.inference import load_image, save_image, evaluate_image


def parse_args():
    parser = argparse.ArgumentParser(description="Compress an image with QUIBC")
    parser.add_argument("--input",      type=str, required=True,  help="Input image path")
    parser.add_argument("--weights",    type=str, required=True,  help="Model weights (.h5)")
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--latent_ch",  type=int, default=96)
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save reconstructed image")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model … ({args.weights})")
    model = load_quibc(args.weights, img_size=args.img_size, latent_ch=args.latent_ch)

    print(f"Compressing {args.input} …")
    metrics = evaluate_image(args.input, model, args.img_size)

    print(f"\n  PSNR    : {metrics['psnr']:.2f} dB")
    print(f"  MS-SSIM : {metrics['ms_ssim']:.4f}")
    print(f"  bpp     : {metrics['bpp']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output_dir, f"{base}_quibc_recon.png")
    save_image(metrics["reconstructed"], out_path)
    print(f"\nReconstructed image saved → {out_path}")


if __name__ == "__main__":
    main()
