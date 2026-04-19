"""
scripts/evaluate.py
Evaluate QUIBC on CLIC validation set or a custom folder of images.

Usage:
    python scripts/evaluate.py --weights checkpoints/best_model.weights.h5
    python scripts/evaluate.py --weights checkpoints/best_model.weights.h5 \
        --dataset custom --images_dir /path/to/images
    python scripts/evaluate.py --weights checkpoints/best_model.weights.h5 \
        --datasets div2k eurosat
"""

import argparse
import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from quibc.deployment import load_quibc
from quibc.losses import psnr_metric, ms_ssim_metric, entropy_bits
from quibc.train import build_clic_datasets


TFDS_DATASET_LOADERS = {
    "div2k": lambda: tfds.load("div2k/bicubic_x2", split="validation", as_supervised=False),
    "eurosat": lambda: tfds.load("eurosat/rgb", split="train[-1000:]", as_supervised=True),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QUIBC compression quality")
    parser.add_argument("--weights",     type=str, required=True)
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--latent_ch",   type=int, default=96)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--dataset",     type=str, default="clic",
                        choices=["clic", "div2k", "eurosat", "custom"])
    parser.add_argument("--images_dir",  type=str, default=None,
                        help="Path to folder of images (used when --dataset custom)")
    parser.add_argument("--max_batches", type=int, default=50)
    return parser.parse_args()


def evaluate_dataset(model, dataset, max_batches=50, img_size=256):
    psnrs, ssims, bpps = [], [], []
    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch

        # Normalise and resize
        if x.dtype != tf.float32:
            x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.resize(x, (img_size, img_size))

        bits, probs = model.encoder(x, training=False)
        recon = model.decoder(bits, training=False)

        psnrs.extend(psnr_metric(x, recon).numpy())
        ssims.extend(ms_ssim_metric(x, recon).numpy())
        bpp = float((entropy_bits(probs) * (img_size // 8) ** 2 / img_size ** 2).numpy())
        bpps.append(bpp)

    return {
        "psnr_mean":    np.mean(psnrs),
        "psnr_std":     np.std(psnrs),
        "ms_ssim_mean": np.mean(ssims),
        "ms_ssim_std":  np.std(ssims),
        "bpp_mean":     np.mean(bpps),
        "bpp_std":      np.std(bpps),
        "n_images":     len(psnrs),
    }


def print_results(name, results):
    print(f"\n{'─'*55}")
    print(f"  Dataset : {name}")
    print(f"  Images  : {results['n_images']}")
    print(f"  PSNR    : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  MS-SSIM : {results['ms_ssim_mean']:.4f} ± {results['ms_ssim_std']:.4f}")
    print(f"  bpp     : {results['bpp_mean']:.4f} ± {results['bpp_std']:.4f}")
    print(f"{'─'*55}")


def main():
    args = parse_args()

    print(f"\nLoading model from {args.weights} …")
    model = load_quibc(args.weights, img_size=args.img_size, latent_ch=args.latent_ch)

    if args.dataset == "clic":
        _, val_ds = build_clic_datasets(
            img_size=args.img_size,
            batch_size=args.batch_size,
            cache=False,
        )
        results = evaluate_dataset(model, val_ds, args.max_batches, args.img_size)
        print_results("CLIC validation", results)

    elif args.dataset in TFDS_DATASET_LOADERS:
        raw = TFDS_DATASET_LOADERS[args.dataset]()
        ds = raw.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        results = evaluate_dataset(model, ds, args.max_batches, args.img_size)
        print_results(args.dataset.upper(), results)

    elif args.dataset == "custom":
        if not args.images_dir:
            raise ValueError("--images_dir is required for --dataset custom")
        paths = sorted(glob.glob(os.path.join(args.images_dir, "**", "*.jpg"), recursive=True)
                     + glob.glob(os.path.join(args.images_dir, "**", "*.png"), recursive=True))
        print(f"Found {len(paths)} images in {args.images_dir}")

        from quibc.inference import batch_evaluate
        results = batch_evaluate(paths, model, args.img_size)
        print_results(f"Custom ({args.images_dir})", results)


if __name__ == "__main__":
    main()
