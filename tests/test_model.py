"""
tests/test_model.py
Unit tests for the core QUIBC model architecture.
"""

import pytest
import numpy as np
import tensorflow as tf

from quibc import (
    QUIBCModel,
    QUIBCModelFixedLambda,
    make_encoder,
    make_decoder,
    make_encoder_no_unitary,
    make_encoder_no_ste,
)

IMG_SIZE   = 64    # use small size for fast tests
LATENT_CH  = 16
BATCH_SIZE = 2


@pytest.fixture(scope="module")
def encoder():
    return make_encoder(IMG_SIZE, LATENT_CH)


@pytest.fixture(scope="module")
def decoder():
    return make_decoder(IMG_SIZE, LATENT_CH)


@pytest.fixture(scope="module")
def model(encoder, decoder):
    return QUIBCModel(encoder, decoder, lam=1e-3)


# ── Architecture tests ───────────────────────────────────────────────────────────

class TestEncoderDecoder:
    def test_encoder_output_shapes(self, encoder):
        x = tf.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, probs = encoder(x, training=False)
        spatial = IMG_SIZE // 8
        assert bits.shape  == (BATCH_SIZE, spatial, spatial, LATENT_CH)
        assert probs.shape == (BATCH_SIZE, spatial, spatial, LATENT_CH)

    def test_bits_are_binary(self, encoder):
        x = tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, _ = encoder(x, training=False)
        # In superposition mode bits ∈ (0,1); in sign mode ∈ {-1, +1}.
        # Either way the range should be bounded.
        assert float(tf.reduce_max(tf.abs(bits)).numpy()) <= 1.0 + 1e-5

    def test_probs_in_range(self, encoder):
        x = tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        _, probs = encoder(x, training=False)
        assert float(tf.reduce_min(probs).numpy()) >= 0.0
        assert float(tf.reduce_max(probs).numpy()) <= 1.0

    def test_decoder_output_shape(self, encoder, decoder):
        x = tf.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, _ = encoder(x, training=False)
        recon = decoder(bits, training=False)
        assert recon.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

    def test_decoder_output_range(self, encoder, decoder):
        x = tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, _ = encoder(x, training=False)
        recon = decoder(bits, training=False)
        assert float(tf.reduce_min(recon).numpy()) >= 0.0 - 1e-5
        assert float(tf.reduce_max(recon).numpy()) <= 1.0 + 1e-5


class TestQUIBCModel:
    def test_model_call(self, model):
        x = tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        recon = model(x, training=False)
        assert recon.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

    def test_lambda_param_exists(self, model):
        assert hasattr(model, "lambda_param")
        assert model.lambda_param.trainable

    def test_effective_lambda_positive(self, model):
        assert model.effective_lambda > 0.0

    def test_compress_decompress_roundtrip(self, model):
        x = tf.random.uniform((1, IMG_SIZE, IMG_SIZE, 3))
        bits = model.compress(x)
        recon = model.decompress(bits)
        assert recon.shape == x.shape


class TestFixedLambdaModel:
    def test_lambda_not_trainable(self, encoder, decoder):
        m = QUIBCModelFixedLambda(encoder, decoder, lam=0.01)
        assert not m.lambda_param.trainable


class TestAblationVariants:
    def test_no_unitary_encoder(self):
        enc = make_encoder_no_unitary(IMG_SIZE, LATENT_CH)
        x = tf.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, probs = enc(x, training=False)
        assert bits.shape[-1] == LATENT_CH

    def test_no_ste_encoder(self):
        enc = make_encoder_no_ste(IMG_SIZE, LATENT_CH)
        x = tf.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        bits, probs = enc(x, training=False)
        assert bits.shape[-1] == LATENT_CH
