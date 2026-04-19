"""
QUIBC: Quantum-Inspired Image Binarization Compressor
======================================================
A quantum-inspired deep learning framework for efficient image compression
on resource-constrained edge devices and IoT systems.

Published at IEEE INDICON 2025.
"""

from .layers import CayleyOrthogonal1x1, STEBinarizer, StandardQuantizer
from .losses import (
    distortion_mse,
    entropy_bits,
    rate_distortion_loss,
    psnr_metric,
    ms_ssim_metric,
    bits_per_pixel,
)
from .model import (
    QUIBCModel,
    QUIBCModelFixedLambda,
    make_encoder,
    make_decoder,
    make_encoder_no_unitary,
    make_encoder_no_ste,
)
from .train import train_model, build_clic_datasets
from .inference import (
    compress_image,
    decompress_image,
    evaluate_image,
    batch_evaluate,
    load_image,
    save_image,
)
from .deployment import (
    convert_to_tflite,
    load_quibc,
    save_quibc,
    simulate_edge_benchmark,
    print_device_comparison,
)

# Expose as top-level alias for convenience
QUIBC = QUIBCModel

__version__ = "1.0.0"
__author__ = (
    "Panchadip Bhattacharjee, Somyajeet Arukh, Arya Abnish Singh, "
    "Jonath Jimmi, Gururaj H L"
)
__all__ = [
    # Layers
    "CayleyOrthogonal1x1",
    "STEBinarizer",
    "StandardQuantizer",
    # Losses / metrics
    "distortion_mse",
    "entropy_bits",
    "rate_distortion_loss",
    "psnr_metric",
    "ms_ssim_metric",
    "bits_per_pixel",
    # Models
    "QUIBC",
    "QUIBCModel",
    "QUIBCModelFixedLambda",
    "make_encoder",
    "make_decoder",
    "make_encoder_no_unitary",
    "make_encoder_no_ste",
    # Training
    "train_model",
    "build_clic_datasets",
    # Inference
    "compress_image",
    "decompress_image",
    "evaluate_image",
    "batch_evaluate",
    "load_image",
    "save_image",
    # Deployment
    "convert_to_tflite",
    "load_quibc",
    "save_quibc",
    "simulate_edge_benchmark",
    "print_device_comparison",
]
