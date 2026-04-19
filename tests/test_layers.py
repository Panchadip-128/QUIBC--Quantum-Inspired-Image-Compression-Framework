"""
tests/test_layers.py
Unit tests for CayleyOrthogonal1x1 and STEBinarizer layers.
"""

import pytest
import numpy as np
import tensorflow as tf

from quibc.layers import CayleyOrthogonal1x1, STEBinarizer, StandardQuantizer

CHANNELS = 16
SPATIAL  = 8
BATCH    = 4


@pytest.fixture
def random_input():
    return tf.random.normal((BATCH, SPATIAL, SPATIAL, CHANNELS))


# ── CayleyOrthogonal1x1 ──────────────────────────────────────────────────────────

class TestCayleyOrthogonal:
    def test_output_shape(self, random_input):
        layer = CayleyOrthogonal1x1(CHANNELS)
        out = layer(random_input)
        assert out.shape == random_input.shape

    def test_spectral_norm_preservation(self, random_input):
        """||Wz||₂ ≈ ||z||₂ for each sample."""
        layer = CayleyOrthogonal1x1(CHANNELS)
        out = layer(random_input)
        # Compare L2-norm across channel dimension
        in_norm  = tf.norm(tf.reshape(random_input, (BATCH, -1)), axis=-1).numpy()
        out_norm = tf.norm(tf.reshape(out,          (BATCH, -1)), axis=-1).numpy()
        np.testing.assert_allclose(in_norm, out_norm, rtol=0.05)

    def test_get_config(self):
        layer = CayleyOrthogonal1x1(channels=32, eps=1e-3)
        cfg = layer.get_config()
        assert cfg["channels"] == 32
        assert cfg["eps"] == 1e-3

    def test_trainable_weights(self):
        layer = CayleyOrthogonal1x1(CHANNELS)
        _ = layer(tf.zeros((1, 4, 4, CHANNELS)))
        assert len(layer.trainable_weights) == 1
        assert layer.trainable_weights[0].name == "M:0"


# ── STEBinarizer ─────────────────────────────────────────────────────────────────

class TestSTEBinarizer:
    def test_output_shapes(self, random_input):
        layer = STEBinarizer()
        codes, probs = layer(random_input)
        assert codes.shape == random_input.shape
        assert probs.shape == random_input.shape

    def test_probs_in_unit_interval(self, random_input):
        layer = STEBinarizer()
        _, probs = layer(random_input)
        assert float(tf.reduce_min(probs)) >= 0.0
        assert float(tf.reduce_max(probs)) <= 1.0

    def test_ste_gradient_flow(self):
        """Gradient must flow through the non-differentiable sign."""
        x = tf.Variable(tf.random.normal((2, 4, 4, 8)))
        layer = STEBinarizer(superposition=False)
        with tf.GradientTape() as tape:
            codes, _ = layer(x, training=True)
            loss = tf.reduce_sum(codes)
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert not tf.reduce_all(tf.equal(grads, 0.0)).numpy(), "All gradients are zero!"

    def test_superposition_mode(self, random_input):
        layer = STEBinarizer(superposition=True)
        codes, _ = layer(random_input, training=False)
        # In superposition mode codes are derived from sigmoid → (0,1)
        assert float(tf.reduce_min(codes)) >= -0.01
        assert float(tf.reduce_max(codes)) <= 1.01

    def test_get_config(self):
        layer = STEBinarizer(superposition=True, noise_std=0.1)
        cfg = layer.get_config()
        assert cfg["superposition"] is True
        assert cfg["noise_std"] == 0.1


# ── StandardQuantizer ────────────────────────────────────────────────────────────

class TestStandardQuantizer:
    def test_output_shapes(self, random_input):
        layer = StandardQuantizer(levels=256)
        codes, probs = layer(random_input)
        assert codes.shape == random_input.shape

    def test_probs_in_range(self, random_input):
        layer = StandardQuantizer()
        _, probs = layer(random_input)
        assert float(tf.reduce_min(probs)) >= 0.0
        assert float(tf.reduce_max(probs)) <= 1.0
