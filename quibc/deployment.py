"""
quibc/deployment.py
Edge device deployment utilities for QUIBC.

Supports TFLite conversion (FP32 / FP16 / INT8) and simulated benchmarking
for Coral Edge TPU, Jetson Xavier NX, Jetson Nano, and Raspberry Pi 4.
"""

from __future__ import annotations

import os
import time
import numpy as np
import tensorflow as tf

from .model import QUIBCModel, make_encoder, make_decoder


# ── Device profiles (based on paper measurements) ──────────────────────────────

DEVICE_PROFILES = {
    "coral_tpu": {
        "fps": 26.19,
        "latency_ms": 38,
        "power_w": 2.0,
        "energy_per_image_j": 0.08,
        "images_per_kwh": 46_800,
        "notes": "INT8 quantization required; use Edge TPU compiler.",
    },
    "jetson_xavier_nx": {
        "fps": 7.20,
        "latency_ms": 139,
        "power_w": None,
        "energy_per_image_j": None,
        "images_per_kwh": None,
        "notes": "FP16 or INT8 quantization recommended.",
    },
    "jetson_nano": {
        "fps": 3.09,
        "latency_ms": 323,
        "power_w": None,
        "energy_per_image_j": None,
        "images_per_kwh": None,
        "notes": "Suitable for batch/offline compression.",
    },
    "raspberry_pi_4": {
        "fps": 0.31,
        "latency_ms": 3200,
        "power_w": None,
        "energy_per_image_j": None,
        "images_per_kwh": None,
        "notes": "CPU-only; use for offline/low-frequency tasks.",
    },
}

# Memory footprint per quantization mode (MB, from paper)
MEMORY_FOOTPRINT_MB = {
    "int8":  5.38,
    "fp16": 10.76,
    "fp32": 20.52,
}


# ── TFLite conversion ───────────────────────────────────────────────────────────

def _build_inference_graph(model: QUIBCModel, img_size: int = 256):
    """Return a concrete tf.function for end-to-end inference."""

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, img_size, img_size, 3], dtype=tf.float32)])
    def inference_fn(image):
        bits, _ = model.encoder(image, training=False)
        return model.decoder(bits, training=False)

    return inference_fn


def convert_to_tflite(
    model: QUIBCModel,
    quantization: str = "fp32",
    img_size: int = 256,
    representative_dataset=None,
    output_path: str = None,
) -> bytes:
    """
    Convert QUIBC to a TFLite FlatBuffer.

    Args:
        model:                  Trained QUIBCModel.
        quantization:           One of 'fp32', 'fp16', 'int8'.
        img_size:               Expected input spatial size.
        representative_dataset: Generator yielding (1, H, W, 3) float32 tensors.
                                Required for INT8 calibration.
        output_path:            If given, save the .tflite file here.

    Returns:
        TFLite model as bytes.
    """
    concrete_fn = _build_inference_graph(model, img_size)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_fn.get_concrete_function()]
    )

    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is None:
            raise ValueError("representative_dataset is required for INT8 quantization.")
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved → {output_path}")
        size_mb = len(tflite_model) / (1024 ** 2)
        print(f"Model size: {size_mb:.2f} MB  (expected ~{MEMORY_FOOTPRINT_MB.get(quantization, '?')} MB)")

    return tflite_model


# ── Benchmark simulation ────────────────────────────────────────────────────────

def simulate_edge_benchmark(
    model: QUIBCModel,
    device: str = "coral_tpu",
    n_images: int = 100,
    img_size: int = 256,
    quantization: str = "int8",
) -> dict:
    """
    Simulate edge device performance using local CPU timing.

    The simulated latency is scaled from the measured baseline in the paper.
    For real on-device numbers, deploy the exported TFLite model.

    Args:
        model:         Trained QUIBCModel.
        device:        Target device key (see DEVICE_PROFILES).
        n_images:      Number of images to simulate.
        img_size:      Spatial resolution.
        quantization:  Quantization scheme used for memory estimate.

    Returns:
        Dictionary with fps, latency_ms, memory_mb and device profile.
    """
    if device not in DEVICE_PROFILES:
        raise ValueError(f"Unknown device '{device}'. Choose from {list(DEVICE_PROFILES)}")

    profile = DEVICE_PROFILES[device]

    # Dummy inference to measure local throughput
    dummy = tf.zeros((1, img_size, img_size, 3))
    # Warm-up
    for _ in range(5):
        _ = model(dummy, training=False)

    start = time.perf_counter()
    for _ in range(n_images):
        _ = model(dummy, training=False)
    elapsed = time.perf_counter() - start

    local_fps     = n_images / elapsed
    local_lat_ms  = (elapsed / n_images) * 1000

    mem_mb = MEMORY_FOOTPRINT_MB.get(quantization, None)

    return {
        "device": device,
        "quantization": quantization,
        "local_fps": round(local_fps, 2),
        "local_latency_ms": round(local_lat_ms, 1),
        "expected_fps_on_device": profile["fps"],
        "expected_latency_ms": profile["latency_ms"],
        "memory_mb": mem_mb,
        "images_per_kwh": profile["images_per_kwh"],
        "notes": profile["notes"],
    }


def print_device_comparison():
    """Print a summary table of all device profiles."""
    print(f"\n{'Device':<22} {'FPS':>8} {'Latency (ms)':>14} {'Power (W)':>10} {'J/image':>10}")
    print("-" * 68)
    for name, p in DEVICE_PROFILES.items():
        fps     = f"{p['fps']}"      if p['fps']     else "—"
        lat     = f"{p['latency_ms']}" if p['latency_ms'] else "—"
        pwr     = f"{p['power_w']}"  if p['power_w'] else "—"
        energy  = f"{p['energy_per_image_j']}" if p['energy_per_image_j'] else "—"
        print(f"{name:<22} {fps:>8} {lat:>14} {pwr:>10} {energy:>10}")
    print()


# ── Model saving / loading ───────────────────────────────────────────────────────

def save_quibc(model: QUIBCModel, path: str):
    """Save QUIBC weights to an HDF5 file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_weights(path)
    print(f"Weights saved → {path}")


def load_quibc(
    path: str,
    img_size: int = 256,
    latent_ch: int = 96,
    lambda_init: float = 1e-3,
) -> QUIBCModel:
    """
    Reconstruct a QUIBCModel and load weights from file.

    Args:
        path:         Path to saved .h5 weights.
        img_size:     Spatial resolution used during training.
        latent_ch:    Latent channel count used during training.
        lambda_init:  Starting λ (overwritten by loaded weights).

    Returns:
        Loaded QUIBCModel.
    """
    encoder = make_encoder(img_size, latent_ch)
    decoder = make_decoder(img_size, latent_ch)
    model = QUIBCModel(encoder, decoder, lam=lambda_init)

    # Build graph before loading weights
    _ = model(tf.zeros((1, img_size, img_size, 3)), training=False)
    model.load_weights(path)
    print(f"Weights loaded ← {path}")
    return model


def deploy_edge(tflite_model: bytes, device: str = "coral_tpu"):
    """
    Placeholder for on-device deployment.
    In production, replace with platform-specific PyCoral / TensorRT APIs.
    """
    print(f"[deploy_edge] Deploying {len(tflite_model) / 1024:.1f} KB TFLite model to '{device}'.")
    print("  → For Coral TPU:    use PyCoral (pycoral.utils.edgetpu)")
    print("  → For Jetson:       use TensorRT via tf2onnx + onnx2trt")
    print("  → For Raspberry Pi: use tf.lite.Interpreter directly")
