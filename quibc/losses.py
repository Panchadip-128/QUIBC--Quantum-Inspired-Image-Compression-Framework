"""
quibc/losses.py
Rate-distortion loss functions and metrics for QUIBC.

The composite loss is:
    L = D + λ · R
where
    D = MSE(x, x̂)              (distortion)
    R = H(p)  [bits/pixel]      (rate, Shannon entropy of binarizer probs)
    λ = softplus(λ_param)       (learnable, always positive)
"""

import tensorflow as tf


# ── Distortion ─────────────────────────────────────────────────────────────────

_mse = tf.keras.losses.MeanSquaredError()


def distortion_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean Squared Error distortion term."""
    return _mse(y_true, y_pred)


# ── Rate (entropy proxy) ────────────────────────────────────────────────────────

def entropy_bits(probs: tf.Tensor, eps: float = 1e-9) -> tf.Tensor:
    """
    Shannon binary entropy in bits, averaged over all latent dimensions.

        H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    Args:
        probs: Probabilities in (0, 1) from STEBinarizer.sigmoid output.
        eps:   Clipping value to avoid log(0).

    Returns:
        Scalar mean entropy (bits).
    """
    p = tf.clip_by_value(probs, eps, 1.0 - eps)
    H = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p)) / tf.math.log(2.0)
    return tf.reduce_mean(H)


# ── Composite loss ──────────────────────────────────────────────────────────────

def rate_distortion_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    probs: tf.Tensor,
    lambda_param: tf.Variable,
) -> tf.Tensor:
    """
    Adaptive rate-distortion loss:
        L = MSE(x, x̂) + softplus(λ_param) · H(probs)

    Args:
        y_true:       Original image batch.
        y_pred:       Reconstructed image batch.
        probs:        Binarizer probability outputs (for entropy proxy).
        lambda_param: Trainable scalar (log-space); softplus ensures positivity.

    Returns:
        Scalar loss value.
    """
    d = distortion_mse(y_true, y_pred)
    r = entropy_bits(probs)
    lam_eff = tf.nn.softplus(lambda_param)
    return d + lam_eff * r


# ── Evaluation metrics ──────────────────────────────────────────────────────────

def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Peak Signal-to-Noise Ratio (dB), assuming images in [0, 1]."""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ms_ssim_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Multi-Scale Structural Similarity Index, images in [0, 1]."""
    return tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0)


def bits_per_pixel(probs: tf.Tensor, img_size: int = 256) -> tf.Tensor:
    """
    Estimate bits-per-pixel from entropy of binarizer probabilities.

    Args:
        probs:    Probability tensor of shape (B, H', W', C).
        img_size: Spatial size of the original image (assumed square).

    Returns:
        Scalar bpp estimate.
    """
    H = entropy_bits(probs)
    latent_spatial = (img_size // 8) ** 2   # encoder downsamples by 8×
    # total bits = H (bits/latent) × latent_size, normalised by image pixels
    return H * latent_spatial / (img_size * img_size)
