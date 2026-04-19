"""
scripts/generate_xai.py
Generate XAI visualisations (Grad-CAM, Integrated Gradients, LRP, latent plots).

Usage:
    python scripts/generate_xai.py --weights checkpoints/best_model.weights.h5 \
        --input test_images/ --output xai_results/
"""

import argparse
import glob
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quibc.deployment import load_quibc
from quibc.inference import load_image
from quibc.xai import (
    grad_cam,
    overlay_grad_cam,
    integrated_gradients,
    plot_integrated_gradients,
    lrp_attribution,
    plot_lrp,
    visualise_encoder_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate QUIBC XAI visualisations")
    parser.add_argument("--weights",   type=str, required=True)
    parser.add_argument("--input",     type=str, required=True,
                        help="Image file or folder of images")
    parser.add_argument("--output",    type=str, default="xai_results")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--latent_ch", type=int, default=96)
    parser.add_argument("--max_imgs",  type=int, default=5,
                        help="Maximum number of images to process")
    return parser.parse_args()


def collect_paths(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    paths = (
        glob.glob(os.path.join(input_path, "*.jpg"))
        + glob.glob(os.path.join(input_path, "*.jpeg"))
        + glob.glob(os.path.join(input_path, "*.png"))
    )
    return sorted(paths)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Loading model … ({args.weights})")
    model = load_quibc(args.weights, img_size=args.img_size, latent_ch=args.latent_ch)

    paths = collect_paths(args.input)[: args.max_imgs]
    print(f"Processing {len(paths)} image(s) …\n")

    for img_path in paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.join(args.output, base)
        os.makedirs(img_dir, exist_ok=True)

        image = load_image(img_path, args.img_size)
        bits, _ = model.encoder(image, training=False)
        recon = model.decoder(bits, training=False)

        print(f"  [{base}]")

        # ── Grad-CAM ───────────────────────────────────────────────────────────
        try:
            heatmap = grad_cam(model, image, layer_name="dec_trans2")
            overlay = overlay_grad_cam(image, heatmap)

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            axes[0].imshow(image.numpy()[0]);       axes[0].set_title("Original");    axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet");    axes[1].set_title("Grad-CAM");    axes[1].axis("off")
            axes[2].imshow(overlay);                axes[2].set_title("Overlay");     axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, "grad_cam.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    ✓ Grad-CAM saved")
        except Exception as e:
            print(f"    ✗ Grad-CAM failed: {e}")

        # ── Integrated Gradients ───────────────────────────────────────────────
        try:
            ig = integrated_gradients(model, image, steps=30)
            plot_integrated_gradients(
                image, ig,
                save_path=os.path.join(img_dir, "integrated_gradients.png"),
            )
            print(f"    ✓ Integrated Gradients saved")
        except Exception as e:
            print(f"    ✗ Integrated Gradients failed: {e}")

        # ── LRP ────────────────────────────────────────────────────────────────
        try:
            relevance = lrp_attribution(model, image)
            plot_lrp(
                image, relevance, recon=recon,
                save_path=os.path.join(img_dir, "lrp.png"),
            )
            print(f"    ✓ LRP saved")
        except Exception as e:
            print(f"    ✗ LRP failed: {e}")

        # ── Encoder feature maps ───────────────────────────────────────────────
        try:
            visualise_encoder_features(
                model, image,
                save_path=os.path.join(img_dir, "encoder_features.png"),
            )
            print(f"    ✓ Encoder features saved")
        except Exception as e:
            print(f"    ✗ Encoder features failed: {e}")

    print(f"\nAll XAI outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
