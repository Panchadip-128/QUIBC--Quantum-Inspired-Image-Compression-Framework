"""
scripts/edge_benchmark.py
Simulate QUIBC performance on edge devices and export TFLite models.

Usage:
    python scripts/edge_benchmark.py --weights checkpoints/best_model.weights.h5
    python scripts/edge_benchmark.py --weights checkpoints/best_model.weights.h5 \
        --device coral_tpu --quantization int8 --export
"""

import argparse
import os
import tensorflow as tf

from quibc.deployment import (
    load_quibc,
    convert_to_tflite,
    simulate_edge_benchmark,
    print_device_comparison,
    DEVICE_PROFILES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="QUIBC edge device benchmark")
    parser.add_argument("--weights",      type=str, required=True)
    parser.add_argument("--img_size",     type=int, default=256)
    parser.add_argument("--latent_ch",    type=int, default=96)
    parser.add_argument("--device",       type=str, default="all",
                        help="'all' or one of: coral_tpu, jetson_xavier_nx, jetson_nano, raspberry_pi_4")
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("--n_images",     type=int, default=100)
    parser.add_argument("--export",       action="store_true",
                        help="Export TFLite model to tflite_models/")
    parser.add_argument("--output_dir",   type=str, default="tflite_models")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading model from {args.weights} …")
    model = load_quibc(args.weights, img_size=args.img_size, latent_ch=args.latent_ch)

    # Paper reference table
    print("\n── Paper-reported device performance ──")
    print_device_comparison()

    # Local simulation
    devices = list(DEVICE_PROFILES.keys()) if args.device == "all" else [args.device]

    print("── Local simulation results ──")
    for dev in devices:
        result = simulate_edge_benchmark(
            model,
            device=dev,
            n_images=args.n_images,
            img_size=args.img_size,
            quantization=args.quantization,
        )
        print(f"\n  [{dev}]")
        print(f"    Local FPS        : {result['local_fps']}")
        print(f"    Local latency    : {result['local_latency_ms']} ms")
        print(f"    Expected FPS     : {result['expected_fps_on_device']} (on real hardware)")
        print(f"    Expected latency : {result['expected_latency_ms']} ms")
        print(f"    Memory ({args.quantization}): {result['memory_mb']} MB")
        if result["images_per_kwh"]:
            print(f"    Images/kWh       : {result['images_per_kwh']:,}")
        print(f"    Notes            : {result['notes']}")

    # Optional TFLite export
    if args.export:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"quibc_{args.quantization}.tflite")
        print(f"\nExporting TFLite model ({args.quantization}) → {out_path} …")
        convert_to_tflite(
            model,
            quantization=args.quantization,
            img_size=args.img_size,
            output_path=out_path,
        )


if __name__ == "__main__":
    main()
