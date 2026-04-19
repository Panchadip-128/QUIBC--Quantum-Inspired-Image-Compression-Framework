# QUIBC: Quantum-Inspired Image Binarization Compressor for Edge Devices

[![Paper](https://img.shields.io/badge/Paper-IEEE%20INDICON%202025-blue)](https://github.com/Panchadip-128/QUIBC)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17%2B-orange)](https://www.tensorflow.org/)

**A quantum-inspired deep learning framework for efficient image compression on resource-constrained edge devices and IoT systems.**

## Overview

QUIBC addresses the critical challenge of image compression for edge devices by introducing a novel quantum-inspired neural network architecture. Traditional codecs like JPEG fail to preserve visual quality at ultra-low bitrates, while existing deep learning models are too computationally expensive for edge deployment.

Our solution achieves **26.58 dB PSNR** and **0.914 MS-SSIM** at only **0.454 bpp** (bits per pixel) using just **1.45M parameters** and **50.9 GFLOPs**, enabling practical deployment on edge hardware.

## Key Features

- **Quantum-Inspired Design**: Leverages unitary transformations and quantum-style binarization for efficient compression
- **Edge-Optimized**: Real-time performance on Coral TPU (26.19 FPS), Jetson Xavier NX (7.2 FPS), and Jetson Nano (3.09 FPS)
- **Adaptive Rate-Distortion**: Learnable Lagrange multiplier (λ) achieves 8.6% bitrate reduction
- **Cross-Domain Robustness**: Validated on medical imaging (32.98 dB), satellite imagery (31.02 dB), and natural images
- **Interpretable**: Comprehensive XAI analysis with Grad-CAM, LRP, and Integrated Gradients
- **Memory Efficient**: 5.38–20.52 MB footprint (INT8–FP32 quantization)

## Architecture

QUIBC consists of three core components:

1. **Encoder**: Strided convolutions (3×3, stride 2) with CayleyOrthogonal 1×1 unitary transformations
2. **Binarization Layer**: STEBinarizer for quantum-style discretization to {-1, +1} states
3. **Decoder**: Transposed convolutions with residual blocks for reconstruction

### Mathematical Foundation

The model optimizes adaptive rate-distortion loss:

```
L = E[d(x, x̂) + λR(x, x̂)]
```

Where:
- `d(x, x̂) = ||x - x̂||²` (MSE distortion)
- `R(z)` is Shannon entropy
- `λ` is a learnable multiplier (log-parameterized, projected via softplus)

Unitary transformations preserve spectral norm:
```
W = (I + A)⁻¹(I - A), where Aᵀ = -A
Ensures: WᵀW = I  (||Wz||₂ = ||z||₂)
```

## Performance

### Compression Quality (CLIC Dataset)

| Split | PSNR (dB) | MS-SSIM | Bitrate (bpp) |
|-------|-----------|---------|---------------|
| Training | 26.58 | 0.914 | 0.454 |
| Validation | 25.05 | 0.909 | 0.449 |

### Comparison with Baselines

| Model | Bitrate (bpp) | PSNR (dB) | Key Feature |
|-------|---------------|-----------|-------------|
| JPEG2000 | ~0.45 | ~24.1 | Wavelet Transform |
| Ballé et al. (2017) | ~0.50 | ~26.5 | End-to-End Learned |
| **QUIBC (Ours)** | **~0.45** | **26.69** | **Quantum-Inspired** |

QUIBC outperforms JPEG2000 by over 1 dB at similar bitrates.

### Cross-Dataset Generalization

| Dataset | Avg PSNR (dB) | Avg MS-SSIM | Avg Rate (bpp) |
|---------|---------------|-------------|----------------|
| DIV2K | 22.72 ± 2.47 | 0.8865 ± 0.0440 | 0.63 ± 0.05 |
| Food-101 | 25.93 ± 2.08 | 0.9159 ± 0.0204 | 0.01 ± 0.00 |
| Oxford Flowers 102 | 24.69 ± 2.27 | 0.8847 ± 0.0466 | 0.01 ± 0.00 |
| Pneumonia MNIST | 32.98 ± 2.10 | 0.9643 ± 0.0180 | 0.45 ± 0.05 |
| EuroSAT | 31.02 ± 2.35 | 0.9404 ± 0.0220 | 0.48 ± 0.06 |

### Edge Device Performance

| Device | FPS | Latency (ms) | Power (W) | Energy/Image (J) |
|--------|-----|--------------|-----------|------------------|
| Coral Edge TPU | 26.19 | 38 | 2.0 | 0.08 |
| Jetson Xavier NX | 7.20 | 139 | — | — |
| Jetson Nano | 3.09 | 323 | — | — |
| Raspberry Pi 4 | 0.31 | 3200 | — | — |

Coral TPU achieves **46,800 images/kWh** — the most energy-efficient option.

### Ablation Study

| Component Removed | PSNR Impact | MS-SSIM Impact | Notes |
|-------------------|-------------|----------------|-------|
| Unitary Transformations | -1.71 dB | -0.0502 | Severe quality degradation |
| STE (Straight-Through Estimator) | Training collapse | NaN | Cannot train without STE |
| Adaptive λ | +0.01 dB | -0.0049 | 8.6% bitrate reduction with adaptive |

## Project Structure

```
QUIBC/
├── quibc/
│   ├── __init__.py           # Public API
│   ├── model.py              # Core QUIBC architecture + ablation variants
│   ├── layers.py             # CayleyOrthogonal1x1, STEBinarizer, StandardQuantizer
│   ├── losses.py             # Rate-distortion loss, PSNR, MS-SSIM, entropy
│   ├── train.py              # Training pipeline, CLIC dataset loader, callbacks
│   ├── inference.py          # Compress / decompress / evaluate images
│   ├── deployment.py         # TFLite export, edge benchmarking, save/load
│   └── xai.py                # Grad-CAM, Integrated Gradients, LRP, latent viz
├── notebooks/
│   ├── training.ipynb        # End-to-end training demonstration
│   ├── xai_analysis.ipynb    # XAI visualisations (Grad-CAM, IG, LRP, t-SNE)
│   └── ablation_study.ipynb  # Component ablation comparisons
├── tests/
│   ├── test_model.py         # Architecture & forward-pass tests
│   ├── test_layers.py        # CayleyOrthogonal & STEBinarizer unit tests
│   └── test_deployment.py    # TFLite conversion & save/load tests
├── scripts/
│   ├── train.py              # CLI training script
│   ├── evaluate.py           # CLI evaluation on CLIC / custom datasets
│   ├── compress.py           # CLI single-image compression
│   ├── edge_benchmark.py     # CLI edge device simulation & TFLite export
│   └── generate_xai.py       # CLI XAI report generation
├── configs/
│   ├── train_config.yaml     # Training hyperparameters
│   └── deploy_config.yaml    # Deployment & device configurations
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Panchadip-128/QUIBC.git
cd QUIBC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .
```

### Requirements

```
tensorflow>=2.17.0
tensorflow-datasets>=4.9.4
numpy>=1.26.0
opencv-python>=4.5.0
Pillow>=9.0.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
PyYAML>=6.0
```

## Usage

### Python API

```python
from quibc import QUIBC, train_model

# Train from scratch (downloads CLIC automatically via tensorflow-datasets)
model, history = train_model(epochs=38, latent_ch=96, checkpoint_dir='checkpoints')

# Load pre-trained weights
from quibc import load_quibc
model = load_quibc('checkpoints/best_model.weights.h5')

# Compress an image
from quibc import compress_image, decompress_image
compressed = compress_image('input.jpg', model)
reconstructed = decompress_image(compressed, model)

# Evaluate quality
from quibc import evaluate_image
metrics = evaluate_image('input.jpg', model)
print(f"PSNR: {metrics['psnr']:.2f} dB  |  MS-SSIM: {metrics['ms_ssim']:.4f}  |  bpp: {metrics['bpp']:.4f}")
```

### Command-Line Interface

```bash
# Train
python scripts/train.py --config configs/train_config.yaml

# Evaluate on CLIC validation set
python scripts/evaluate.py --weights checkpoints/best_model.weights.h5

# Compress a single image
python scripts/compress.py --input photo.jpg --weights checkpoints/best_model.weights.h5

# Edge device benchmark + TFLite export
python scripts/edge_benchmark.py \
    --weights checkpoints/best_model.weights.h5 \
    --device coral_tpu --quantization int8 --export

# Generate XAI report
python scripts/generate_xai.py \
    --weights checkpoints/best_model.weights.h5 \
    --input test_images/ --output xai_results/
```

### Edge Deployment

```python
from quibc import convert_to_tflite, load_quibc

model = load_quibc('checkpoints/best_model.weights.h5')

# FP16 (Jetson devices)
tflite_fp16 = convert_to_tflite(model, quantization='fp16', output_path='quibc_fp16.tflite')

# INT8 (Coral TPU — requires calibration data)
def representative_dataset():
    for x, _ in val_ds.take(50):
        yield [x]

tflite_int8 = convert_to_tflite(
    model, quantization='int8',
    representative_dataset=representative_dataset,
    output_path='quibc_int8.tflite'
)
```

### Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=quibc --cov-report=term-missing
```

## Explainability (XAI)

QUIBC incorporates comprehensive interpretability analysis accessible via `quibc.xai`:

- **Grad-CAM** — Decoder attention on perceptually significant regions
- **Integrated Gradients** — Pixel-level contributions to reconstruction
- **LRP** (Layer-wise Relevance Propagation) — Spatial reconstruction importance
- **t-SNE / PCA / UMAP** — Structured latent space organisation

See `notebooks/xai_analysis.ipynb` for interactive visualisations.

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning Rate | 1e-4 with exponential decay (0.96^(step/1000)) |
| Batch Size | 16 |
| Epochs | 38 |
| Image Size | 256×256 |
| Latent Channels | 96 |
| λ (initial) | 1e-3 (adaptive, learned) |

### Hardware Requirements

**Training:**
- GPU: NVIDIA T4 or better
- RAM: 16 GB minimum
- Storage: 50 GB for datasets

**Inference (Edge):**
- Coral TPU, Jetson Xavier NX/Nano, Raspberry Pi 4
- See `configs/deploy_config.yaml` for device-specific settings

## Citation

If you use QUIBC in your research, please cite:

```bibtex
@inproceedings{jimmi2025quibc,
  title     = {QUIBC: A Quantum-Inspired Image Binarization Compressor for Resource-Constrained Edge Devices},
  author    = {Jimmi, Jonath and Arukh, Somyajeet and Singh, Arya Abnish and Bhattacharjee, Panchadip and H L, Gururaj},
  booktitle = {IEEE INDICON 2025},
  year      = {2025},
  organization = {IEEE}
}
```

## Main Contributors

| Name | Affiliation | Email |
|------|------------|-------|
| **Panchadip Bhattacharjee** | MIT Bengaluru, MAHE | panchadip.mitblr2023@learner.manipal.edu |
| **Somyajeet Arukh** | MIT Bengaluru, MAHE | somyajeet.mitblr2023@learner.manipal.edu |
| **Arya Abnish Singh** | MIT Bengaluru, MAHE | arya.mitblr2024@learner.manipal.edu |
| **Jonath Jimmi** | MIT Bengaluru, MAHE | jonath.mitblr2024@learner.manipal.edu |
| **Dr. Gururaj H L** (Supervisor) | MIT Bengaluru, MAHE | gururaj.hl@manipal.edu |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- CLIC (Challenge on Learned Image Compression) dataset and organizers
- TensorFlow and Keras teams
- IEEE INDICON 2025 reviewers and organizers
- Manipal Institute of Technology Bengaluru
- Manipal Academy of Higher Education

## Future Work

- Extension to video compression and 3D point clouds
- Hardware-specific optimizations for FPGA/ASIC implementations
- Real-time streaming compression for IoT camera networks
- Integration with federated learning for privacy-preserving compression
- Post-quantum cryptographic extensions
- Multi-modal compression (combining images with sensor data)

## FAQ

**Q: What makes QUIBC different from traditional compression?**
QUIBC uses quantum-inspired techniques (unitary transformations, binarization) for 1+ dB better quality at the same bitrate, with edge-friendly compute (1.45M params, 50.9 GFLOPs).

**Q: Can I use QUIBC for real-time applications?**
Yes! On Coral TPU, QUIBC achieves 26.19 FPS, suitable for real-time IoT camera streams.

**Q: Does it work on grayscale images?**
Yes — adjust `model.encoder` input channels. The architecture handles both RGB and grayscale.

**Q: How do I fine-tune for my specific domain?**
Load pretrained weights with `load_quibc()` and continue training on your domain-specific dataset.

**Q: What is the minimum hardware for inference?**
Raspberry Pi 4 works at 0.31 FPS (offline use). For real-time, use Coral TPU or Jetson devices.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Run the test suite (`pytest tests/ -v`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Open a Pull Request

---

**Keywords**: Quantum ML · Image Compression · Edge AI · IoT · Rate-Distortion · Binarization · Unitary Transformations · TFLite · Explainable AI

**Status**: ✅ IEEE INDICON 2025 Published · 🚀 Active Development · 📊 Benchmarked on 5+ Datasets
