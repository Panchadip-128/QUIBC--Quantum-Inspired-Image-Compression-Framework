"""
quibc/inference.py
Image compression and decompression utilities for QUIBC.
"""

from __future__ import annotations

import io
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

from .model import QUIBCModel


# ── Image I/O helpers ────────────────────────────────────────────────────────────

def load_image(path: str, img_size: int = 256) -> tf.Tensor:
    """
    Load an image file and preprocess it for QUIBC.

    Returns:
        Float32 tensor of shape (1, img_size, img_size, 3) in [0, 1].
    """
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Center-crop to square then resize
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    s = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, s, s)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)

    return tf.expand_dims(img, 0)   # (1, H, W, 3)


def tensor_to_pil(tensor: tf.Tensor) -> PILImage.Image:
    """Convert a float32 image tensor (1, H, W, 3) → PIL Image."""
    arr = np.clip(tensor.numpy()[0] * 255, 0, 255).astype(np.uint8)
    return PILImage.fromarray(arr)


def save_image(tensor: tf.Tensor, path: str):
    """Save a float32 image tensor to disk as PNG/JPEG (inferred from extension)."""
    tensor_to_pil(tensor).save(path)


# ── Core compression / decompression ────────────────────────────────────────────

def compress_image(
    input_path: str,
    model: QUIBCModel,
    img_size: int = 256,
) -> dict:
    """
    Compress a single image using QUIBC.

    Args:
        input_path: Path to the source image.
        model:      Trained QUIBCModel instance.
        img_size:   Spatial resolution expected by the model.

    Returns:
        Dictionary with keys:
            'bits'      – binary latent codes (numpy array, float32 {-1, +1}),
            'shape'     – latent shape tuple,
            'original'  – original image tensor (1, H, W, 3).
    """
    original = load_image(input_path, img_size)
    bits, _ = model.encoder(original, training=False)
    bits_np = bits.numpy()
    return {
        "bits": bits_np,
        "shape": bits_np.shape,
        "original": original,
    }


def decompress_image(
    compressed: dict,
    model: QUIBCModel,
) -> tf.Tensor:
    """
    Reconstruct an image from a compressed representation.

    Args:
        compressed: Output dict from `compress_image`.
        model:      Trained QUIBCModel instance.

    Returns:
        Reconstructed image tensor (1, H, W, 3).
    """
    bits = tf.constant(compressed["bits"], dtype=tf.float32)
    return model.decoder(bits, training=False)


def compress_tensor(image: tf.Tensor, model: QUIBCModel) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compress a pre-loaded image tensor.

    Args:
        image: Float32 tensor (1, H, W, 3) in [0, 1].
        model: Trained QUIBCModel.

    Returns:
        (bits, probs) tensors.
    """
    return model.encoder(image, training=False)


def decompress_tensor(bits: tf.Tensor, model: QUIBCModel) -> tf.Tensor:
    """Reconstruct image from binary latent codes tensor."""
    return model.decoder(bits, training=False)


# ── Quality evaluation ───────────────────────────────────────────────────────────

def evaluate_image(
    input_path: str,
    model: QUIBCModel,
    img_size: int = 256,
) -> dict:
    """
    Compress and decompress one image, returning quality metrics.

    Returns:
        Dictionary with 'psnr', 'ms_ssim', 'bpp', 'original', 'reconstructed'.
    """
    from .losses import psnr_metric, ms_ssim_metric, entropy_bits

    original = load_image(input_path, img_size)
    bits, probs = model.encoder(original, training=False)
    recon = model.decoder(bits, training=False)

    psnr    = float(tf.reduce_mean(psnr_metric(original, recon)).numpy())
    ms_ssim = float(tf.reduce_mean(ms_ssim_metric(original, recon)).numpy())
    bpp     = float((entropy_bits(probs) * (img_size // 8) ** 2 / (img_size ** 2)).numpy())

    return {
        "psnr": psnr,
        "ms_ssim": ms_ssim,
        "bpp": bpp,
        "original": original,
        "reconstructed": recon,
    }


def batch_evaluate(
    image_paths: list[str],
    model: QUIBCModel,
    img_size: int = 256,
) -> dict:
    """
    Evaluate QUIBC over a list of images and return aggregated statistics.

    Returns:
        Dict with mean/std for psnr, ms_ssim, bpp.
    """
    from .losses import psnr_metric, ms_ssim_metric, entropy_bits

    psnrs, ms_ssims, bpps = [], [], []
    for path in image_paths:
        r = evaluate_image(path, model, img_size)
        psnrs.append(r["psnr"])
        ms_ssims.append(r["ms_ssim"])
        bpps.append(r["bpp"])

    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std":  float(np.std(psnrs)),
        "ms_ssim_mean": float(np.mean(ms_ssims)),
        "ms_ssim_std":  float(np.std(ms_ssims)),
        "bpp_mean": float(np.mean(bpps)),
        "bpp_std":  float(np.std(bpps)),
    }
