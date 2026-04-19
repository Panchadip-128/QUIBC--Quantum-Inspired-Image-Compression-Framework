"""
quibc/layers.py
Custom layers for QUIBC: CayleyOrthogonal1x1 (unitary transform) and STEBinarizer.
"""

import tensorflow as tf
from tensorflow.keras import layers


class CayleyOrthogonal1x1(layers.Layer):
    """
    Parameterises a unitary (orthogonal) 1×1 convolution via the Cayley map.

    Given a skew-symmetric matrix A (A^T = -A), the Cayley map produces:
        W = (I + A + εI)^{-1} (I - A)
    which satisfies W^T W ≈ I, preserving spectral norm (||Wz||₂ = ||z||₂).
    This mirrors the reversibility of quantum gates.

    Args:
        channels: Number of input/output channels.
        eps: Small regularisation constant for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.eps = eps

    def build(self, input_shape):
        C = self.channels
        self.M = self.add_weight(
            name="M",
            shape=(C, C),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        C = self.channels
        A = self.M - tf.transpose(self.M)          # skew-symmetric
        I = tf.eye(C, dtype=x.dtype)
        W = tf.linalg.solve(I + A + self.eps * I, I - A)
        return tf.tensordot(x, W, axes=[[3], [0]])

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "eps": self.eps})
        return config


class STEBinarizer(layers.Layer):
    """
    Quantum-inspired binarizer using the Straight-Through Estimator (STE).

    Forward pass:  ẑ = sign(z) ∈ {-1, +1}  (mirrors quantum measurement
                   collapse to basis states |0⟩, |1⟩).
    Backward pass: gradient flows as if the sign were the identity
                   (straight-through approximation), enabling stable training.

    Also returns `probs = sigmoid(z) ∈ (0,1)` for Shannon-entropy rate
    estimation.

    Args:
        superposition: If True, use soft rounding instead of hard sign
                       (useful for exploring "superposition" semantics).
        noise_std:     Std-dev of optional Gaussian noise injected during
                       training (quantum-style uncertainty simulation).
    """

    def __init__(self, superposition: bool = False, noise_std: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.superposition = superposition
        self.noise_std = noise_std

    def call(self, inputs, training=None):
        probs = tf.sigmoid(inputs)          # probability proxy for entropy

        if self.superposition:
            hard = tf.round(probs)
            codes = probs + tf.stop_gradient(hard - probs)
        else:
            forward = tf.sign(inputs)
            codes = inputs + tf.stop_gradient(forward - inputs)

        if training and self.noise_std > 0.0:
            codes = codes + tf.random.normal(tf.shape(codes), stddev=self.noise_std)

        return codes, probs

    def get_config(self):
        config = super().get_config()
        config.update({"superposition": self.superposition, "noise_std": self.noise_std})
        return config


class StandardQuantizer(layers.Layer):
    """
    Multi-level uniform quantizer (used in ablation studies as STE alternative).

    Args:
        levels: Number of quantization levels (default 256).
    """

    def __init__(self, levels: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels

    def call(self, inputs, training=None):
        normalized = tf.sigmoid(inputs)
        quantized = tf.round(normalized * (self.levels - 1)) / (self.levels - 1)
        # Straight-through so gradients still flow
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        probs = tf.sigmoid(inputs)
        return quantized, probs

    def get_config(self):
        config = super().get_config()
        config.update({"levels": self.levels})
        return config


# Registry for easy custom-object loading
CUSTOM_OBJECTS = {
    "CayleyOrthogonal1x1": CayleyOrthogonal1x1,
    "STEBinarizer": STEBinarizer,
    "StandardQuantizer": StandardQuantizer,
}
