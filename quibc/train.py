"""
quibc/train.py
Full training pipeline for QUIBC: dataset loading, preprocessing, and training loop.
"""

import os
import math
import tensorflow as tf
import tensorflow_datasets as tfds

from .model import QUIBCModel, make_encoder, make_decoder

# ── Default hyperparameters ─────────────────────────────────────────────────────
DEFAULTS = dict(
    img_size=256,
    latent_ch=96,
    batch_size=16,
    epochs=38,
    learning_rate=1e-4,
    lambda_init=1e-3,
    seed=42,
)


# ── Preprocessing ───────────────────────────────────────────────────────────────

def center_square_resize(x: tf.Tensor, size: int) -> tf.Tensor:
    """Center-crop to square then resize to (size × size)."""
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    s = tf.minimum(h, w)
    x = tf.image.resize_with_crop_or_pad(x, s, s)
    x = tf.image.resize(x, (size, size), antialias=True)
    return x


def degrade_iot(x: tf.Tensor, seed: int = 42) -> tf.Tensor:
    """Simulate IoT camera degradation: mild Gaussian noise + average blur."""
    noise = tf.random.normal(tf.shape(x), stddev=0.01, seed=seed)
    x = x + noise
    x = tf.nn.avg_pool2d(x[None, ...], ksize=2, strides=1, padding="SAME")[0]
    return tf.clip_by_value(x, 0.0, 1.0)


def build_clic_datasets(
    img_size: int = 256,
    batch_size: int = 16,
    seed: int = 42,
    iot_augment: bool = True,
    cache: bool = True,
):
    """
    Load and preprocess the CLIC dataset from TensorFlow Datasets.

    Args:
        img_size:     Target spatial resolution (square).
        batch_size:   Mini-batch size.
        seed:         Random seed for shuffling / noise.
        iot_augment:  Apply IoT noise+blur to training images.
        cache:        Cache dataset in memory after first epoch.

    Returns:
        (train_ds, val_ds) as tf.data.Dataset objects.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tfds.load("clic", as_supervised=False)
    train_raw = ds["train"]
    val_raw   = ds["validation"]

    def _preprocess_train(sample):
        img = tf.image.convert_image_dtype(sample["image"], tf.float32)
        img = center_square_resize(img, img_size)
        if iot_augment:
            img = degrade_iot(img, seed)
        return img, img

    def _preprocess_val(sample):
        img = tf.image.convert_image_dtype(sample["image"], tf.float32)
        img = center_square_resize(img, img_size)
        return img, img

    train_ds = train_raw.map(_preprocess_train, num_parallel_calls=AUTOTUNE)
    if cache:
        train_ds = train_ds.cache()
    train_ds = (
        train_ds
        .shuffle(2048, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = val_raw.map(_preprocess_val, num_parallel_calls=AUTOTUNE)
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds


# ── Learning-rate schedule ──────────────────────────────────────────────────────

def exponential_decay_schedule(
    initial_lr: float = 1e-4,
    decay_rate: float = 0.96,
    decay_steps: int = 1000,
):
    """Exponential decay: η(t) = lr₀ · decay_rate^(t / decay_steps)."""
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )


# ── Callbacks ───────────────────────────────────────────────────────────────────

def build_callbacks(checkpoint_dir: str = "checkpoints", patience: int = 8):
    """Standard callback suite: checkpointing, early stopping, NaN guard."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.weights.h5"),
            monitor="val_psnr",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_psnr",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs"),
            histogram_freq=0,
        ),
    ]


# ── High-level training entry point ────────────────────────────────────────────

def train_model(
    model: QUIBCModel = None,
    train_path: str = None,       # unused when using TFDS; reserved for custom datasets
    val_path: str = None,
    img_size: int = DEFAULTS["img_size"],
    latent_ch: int = DEFAULTS["latent_ch"],
    batch_size: int = DEFAULTS["batch_size"],
    epochs: int = DEFAULTS["epochs"],
    learning_rate: float = DEFAULTS["learning_rate"],
    lambda_init: float = DEFAULTS["lambda_init"],
    seed: int = DEFAULTS["seed"],
    checkpoint_dir: str = "checkpoints",
    iot_augment: bool = True,
    cache: bool = True,
):
    """
    Train a QUIBC model end-to-end.

    If `model` is None, a fresh model is built from the defaults.
    Returns (model, history).

    Example::

        from quibc import QUIBC, train_model
        model, history = train_model(epochs=38)
    """
    tf.keras.utils.set_random_seed(seed)

    # Build model if not provided
    if model is None:
        encoder = make_encoder(img_size, latent_ch)
        decoder = make_decoder(img_size, latent_ch)
        model = QUIBCModel(encoder, decoder, lam=lambda_init)

    # Compile with exponential-decay Adam
    lr_schedule = exponential_decay_schedule(learning_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule))

    # Warm up model graph
    _ = model(tf.zeros((1, img_size, img_size, 3)), training=False)

    # Data
    train_ds, val_ds = build_clic_datasets(
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
        iot_augment=iot_augment,
        cache=cache,
    )

    callbacks = build_callbacks(checkpoint_dir=checkpoint_dir)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    return model, history
