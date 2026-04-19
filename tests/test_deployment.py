"""
tests/test_deployment.py
Tests for deployment utilities: TFLite conversion, edge benchmarking, save/load.
"""

import os
import tempfile
import pytest
import numpy as np
import tensorflow as tf

from quibc import make_encoder, make_decoder, QUIBCModel, save_quibc, load_quibc
from quibc.deployment import (
    convert_to_tflite,
    simulate_edge_benchmark,
    print_device_comparison,
    DEVICE_PROFILES,
    MEMORY_FOOTPRINT_MB,
)

IMG_SIZE  = 64
LATENT_CH = 16


@pytest.fixture(scope="module")
def model():
    enc = make_encoder(IMG_SIZE, LATENT_CH)
    dec = make_decoder(IMG_SIZE, LATENT_CH)
    m = QUIBCModel(enc, dec, lam=1e-3)
    _ = m(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)), training=False)
    return m


# ── TFLite conversion ────────────────────────────────────────────────────────────

class TestTFLiteConversion:
    def test_fp32_conversion(self, model):
        tflite_bytes = convert_to_tflite(model, quantization="fp32", img_size=IMG_SIZE)
        assert isinstance(tflite_bytes, bytes)
        assert len(tflite_bytes) > 1000   # non-trivial model

    def test_fp16_conversion(self, model):
        tflite_bytes = convert_to_tflite(model, quantization="fp16", img_size=IMG_SIZE)
        assert isinstance(tflite_bytes, bytes)

    def test_int8_requires_representative_dataset(self, model):
        with pytest.raises(ValueError, match="representative_dataset"):
            convert_to_tflite(model, quantization="int8", img_size=IMG_SIZE)

    def test_output_file_written(self, model, tmp_path):
        out = str(tmp_path / "model.tflite")
        convert_to_tflite(model, quantization="fp32", img_size=IMG_SIZE, output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# ── Edge benchmark simulation ────────────────────────────────────────────────────

class TestEdgeBenchmark:
    def test_known_device(self, model):
        results = simulate_edge_benchmark(
            model, device="coral_tpu", n_images=5, img_size=IMG_SIZE
        )
        assert "local_fps" in results
        assert results["local_fps"] > 0
        assert results["expected_fps_on_device"] == DEVICE_PROFILES["coral_tpu"]["fps"]

    def test_unknown_device_raises(self, model):
        with pytest.raises(ValueError):
            simulate_edge_benchmark(model, device="unknown_device")

    def test_all_devices_have_profiles(self):
        for device in ["coral_tpu", "jetson_xavier_nx", "jetson_nano", "raspberry_pi_4"]:
            assert device in DEVICE_PROFILES

    def test_memory_footprint_table(self):
        assert MEMORY_FOOTPRINT_MB["int8"]  < MEMORY_FOOTPRINT_MB["fp32"]
        assert MEMORY_FOOTPRINT_MB["fp16"]  < MEMORY_FOOTPRINT_MB["fp32"]

    def test_print_device_comparison(self, capsys):
        print_device_comparison()
        captured = capsys.readouterr()
        assert "coral_tpu" in captured.out


# ── Model save / load ────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_reload(self, model, tmp_path):
        path = str(tmp_path / "quibc.weights.h5")
        save_quibc(model, path)
        assert os.path.exists(path)

        loaded = load_quibc(path, img_size=IMG_SIZE, latent_ch=LATENT_CH)
        # Check reconstructions are identical after reload
        x = tf.random.uniform((1, IMG_SIZE, IMG_SIZE, 3), seed=0)
        orig_out   = model(x, training=False).numpy()
        loaded_out = loaded(x, training=False).numpy()
        np.testing.assert_allclose(orig_out, loaded_out, atol=1e-5)
