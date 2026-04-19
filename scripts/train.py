"""
scripts/train.py
Command-line entry point for training QUIBC.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --epochs 50 --latent_ch 128 --checkpoint_dir runs/exp1
"""

import argparse
import yaml
import tensorflow as tf

from quibc.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train QUIBC model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to train_config.yaml (overrides individual flags)")
    parser.add_argument("--img_size",       type=int,   default=256)
    parser.add_argument("--latent_ch",      type=int,   default=96)
    parser.add_argument("--epochs",         type=int,   default=38)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--learning_rate",  type=float, default=1e-4)
    parser.add_argument("--lambda_init",    type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--no_iot_augment", action="store_true",
                        help="Disable IoT noise/blur augmentation")
    parser.add_argument("--no_cache",       action="store_true",
                        help="Disable dataset caching (saves RAM, slower training)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Base kwargs from CLI
    kwargs = dict(
        img_size=args.img_size,
        latent_ch=args.latent_ch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_init=args.lambda_init,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        iot_augment=not args.no_iot_augment,
        cache=not args.no_cache,
    )

    # Override with YAML config if provided
    if args.config:
        cfg = load_config(args.config)
        m = cfg.get("model", {})
        t = cfg.get("training", {})
        o = cfg.get("optimizer", {})
        d = cfg.get("dataset", {})
        c = cfg.get("callbacks", {})

        kwargs.update({
            "img_size":       m.get("img_size",      kwargs["img_size"]),
            "latent_ch":      m.get("latent_ch",     kwargs["latent_ch"]),
            "lambda_init":    m.get("lambda_init",   kwargs["lambda_init"]),
            "epochs":         t.get("epochs",        kwargs["epochs"]),
            "batch_size":     t.get("batch_size",    kwargs["batch_size"]),
            "seed":           t.get("seed",          kwargs["seed"]),
            "learning_rate":  o.get("learning_rate", kwargs["learning_rate"]),
            "iot_augment":    d.get("iot_augment",   kwargs["iot_augment"]),
            "cache":          d.get("cache",         kwargs["cache"]),
            "checkpoint_dir": c.get("checkpoint_dir", kwargs["checkpoint_dir"]),
        })

    print("\n" + "=" * 60)
    print("QUIBC Training")
    print("=" * 60)
    for k, v in kwargs.items():
        print(f"  {k:<20} {v}")
    print("=" * 60 + "\n")

    model, history = train_model(**kwargs)

    # Final metrics
    val_psnr   = max(history.history.get("val_psnr",   [0]))
    val_msssim = max(history.history.get("val_ms_ssim",[0]))
    print(f"\nBest validation PSNR:    {val_psnr:.2f} dB")
    print(f"Best validation MS-SSIM: {val_msssim:.4f}")
    print(f"Effective λ:             {model.effective_lambda:.6f}")


if __name__ == "__main__":
    main()
