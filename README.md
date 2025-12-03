# QUIBC: Quantum-Inspired Image Binarization Compressor for Edge Devices

[![Paper](https://img.shields.io/badge/Paper-IEEE%20INDICON%202025-blue)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

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
- **Memory Efficient**: 5.38-20.52 MB footprint (INT8-FP32 quantization)

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
- `λ` is a learnable multiplier

Unitary transformations preserve spectral norm:
```
W = (I + A)⁻¹(I - A), where Aᵀ = -A
Ensures: WᵀW = I (||Wz||₂ = ||z||₂)
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
| Jetson Xavier NX | 7.20 | 139 | - | - |
| Jetson Nano | 3.09 | 323 | - | - |
| Raspberry Pi 4 | 0.31 | 3200 | - | - |

Coral TPU achieves **46,800 images/kWh** - the most energy-efficient option.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QUIBC.git
cd QUIBC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.8.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
pillow>=8.3.0
scipy>=1.7.0
```

## Usage

### Training

```python
from quibc import QUIBC, train_model

# Initialize model
model = QUIBC(latent_dim=128, lambda_init=0.01)

# Train on CLIC dataset
train_model(
    model=model,
    train_path='path/to/clic/train',
    val_path='path/to/clic/val',
    epochs=38,
    batch_size=16,
    learning_rate=1e-4
)
```

### Compression

```python
from quibc import compress_image, decompress_image

# Compress an image
compressed = compress_image('input.jpg', model)

# Decompress
reconstructed = decompress_image(compressed, model)
```

### Edge Deployment

```python
# Convert to TensorFlow Lite for edge devices
from quibc.deployment import convert_to_tflite

tflite_model = convert_to_tflite(model, quantization='int8')

# Deploy to Coral TPU, Jetson, or Raspberry Pi
from quibc.deployment import deploy_edge
deploy_edge(tflite_model, device='coral_tpu')
```

## Project Structure

```
QUIBC/
├── quibc/
│   ├── __init__.py
│   ├── model.py              # Core QUIBC architecture
│   ├── layers.py             # Custom layers (CayleyOrthogonal, STEBinarizer)
│   ├── losses.py             # Rate-distortion loss functions
│   ├── train.py              # Training pipeline
│   ├── inference.py          # Compression/decompression
│   ├── deployment.py         # Edge device deployment utilities
│   └── xai.py                # Explainability tools
├── notebooks/
│   ├── training.ipynb        # Training demonstration
│   ├── xai_analysis.ipynb    # XAI visualizations
│   └── ablation_study.ipynb  # Component ablation studies
├── tests/
│   ├── test_model.py
│   ├── test_layers.py
│   └── test_deployment.py
├── configs/
│   ├── train_config.yaml     # Training configurations
│   └── deploy_config.yaml    # Deployment configurations
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Ablation Studies

Critical component contributions:

| Component Removed | PSNR Impact | MS-SSIM Impact | Notes |
|-------------------|-------------|----------------|-------|
| Unitary Transformations | -1.71 dB | -0.0502 | Severe quality degradation |
| STE (Straight-Through Estimator) | Training collapse | NaN | Cannot train without STE |
| Adaptive λ | +0.01 dB | -0.0049 | 8.6% bitrate reduction with adaptive |

## Explainability (XAI)

QUIBC incorporates comprehensive interpretability analysis:

- **Grad-CAM**: Visualizes decoder attention on perceptually significant regions
- **Integrated Gradients**: Identifies pixel-level contributions to reconstruction
- **LRP (Layer-wise Relevance Propagation)**: Attributes reconstruction importance
- **t-SNE/UMAP**: Reveals structured latent space organization

See `notebooks/xai_analysis.ipynb` for detailed visualizations.

## Training Details

### Dataset Preprocessing

- **Input Size**: 256×256 pixels
- **Normalization**: [0, 1] range
- **Augmentation**: IoT simulation (blur, noise on training set)
- **Dataset**: CLIC (Challenge on Learned Image Compression)

### Hyperparameters

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Learning Rate**: 1e-4 with exponential decay (0.96^(step/1000))
- **Batch Size**: 16
- **Epochs**: 38
- **Loss**: Adaptive rate-distortion (MSE + λ·Shannon Entropy)

### Hardware Requirements

**Training:**
- GPU: NVIDIA T4 or better
- RAM: 16 GB minimum
- Storage: 50 GB for datasets

**Inference:**
- Edge Devices: Coral TPU, Jetson Xavier/Nano, Raspberry Pi 4
- CPU: Any modern processor for offline compression

## Benchmark Scripts

```bash
# Run full evaluation on CLIC dataset
python scripts/evaluate.py --dataset clic --checkpoint checkpoints/best_model.h5

# Cross-dataset generalization test
python scripts/cross_dataset_eval.py --datasets div2k food101 eurosat

# Edge device simulation
python scripts/edge_benchmark.py --device coral_tpu --quantization int8

# XAI analysis
python scripts/generate_xai.py --input test_images/ --output xai_results/
```

## Citation

If you use QUIBC in your research, please cite our paper:

```bibtex
@inproceedings{jimmi2025quibc,
  title={QUIBC: A Quantum-Inspired Image Binarization Compressor for Resource-Constrained Edge Devices},
  author={Jimmi, Jonath and Arukh, Somyajeet and Singh, Arya Abnish and Bhattacharjee, Panchadip and H L, Gururaj},
  booktitle={IEEE INDICON 2025},
  year={2025},
  organization={IEEE}
}
```

## Authors

- **Jonath Jimmi** - Manipal Institute of Technology Bengaluru  
  📧 jonath.mitblr2024@learner.manipal.edu

- **Somyajeet Arukh** - Manipal Institute of Technology Bengaluru  
  📧 somyajeet.mitblr2023@learner.manipal.edu

- **Arya Abnish Singh** - Manipal Institute of Technology Bengaluru  
  📧 arya.mitblr2024@learner.manipal.edu

- **Panchadip Bhattacharjee** - Manipal Institute of Technology Bengaluru  
  📧 panchadip.mitblr2023@learner.manipal.edu

- **Gururaj H L** - Manipal Institute of Technology Bengaluru  
  📧 gururaj.hl@manipal.edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CLIC (Challenge on Learned Image Compression) dataset
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

## Related Publications

1. **QUIBC Paper** - IEEE INDICON 2025
2. **MST-Conformer** - Multi-Scale Temporal Convolution for HAR
3. **FeTA** - Federated Learning with Privacy Compliance

## FAQ

**Q: What makes QUIBC different from traditional compression?**  
A: QUIBC uses quantum-inspired techniques (unitary transformations, binarization) for 1+ dB better quality at same bitrate, with edge-friendly compute.

**Q: Can I use QUIBC for real-time applications?**  
A: Yes! On Coral TPU, QUIBC achieves 26 FPS, suitable for real-time IoT camera streams.

**Q: Does it work on grayscale images?**  
A: Yes, the architecture handles both RGB and grayscale. Adjust input channels accordingly.

**Q: How do I fine-tune for my specific domain?**  
A: Use transfer learning: load pretrained weights and continue training on your domain-specific dataset.

**Q: What's the minimum hardware for inference?**  
A: Raspberry Pi 4 works but at 0.31 FPS (offline use). For real-time, use Coral TPU or Jetson devices.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions, collaborations, or issues:
- 📧 Email: panchadip.mitblr2023@learner.edu
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/QUIBC/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Panchadip-128/QUIBC/discussions)

## Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/QUIBC&type=Date)](https://star-history.com/#yourusername/QUIBC&Date)

---

**Keywords**: Quantum ML, Image Compression, Edge AI, IoT, Rate-Distortion, Binarization, Unitary Transformations, Deep Learning, Resource-Constrained Devices, Explainable AI

**Status**: ✅ IEEE INDICON 2025 Published | 🚀 Active Development | 📊 Benchmarked on 5+ Datasets
