<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CelebA-FF6F00?style=for-the-badge&logo=database&logoColor=white" alt="CelebA"/>
  <img src="https://img.shields.io/badge/MPS-000000?style=for-the-badge&logo=apple&logoColor=white" alt="MPS"/>
</p>

<h1 align="center">
  Vanilla GAN — Face Generation from Scratch
</h1>

<p align="center">
  <b>A PyTorch implementation of the original Generative Adversarial Network (Goodfellow et al., 2014)</b><br>
  Trained on 200K+ celebrity faces to generate photorealistic portraits from random noise.
</p>

---

## Overview

This project implements a **Vanilla GAN** — the foundational architecture that kicked off the generative AI revolution. The network learns to produce convincing face images by pitting two neural networks against each other in a minimax game:

- **Generator (G)**: Takes random noise (z ∼ N(0, I), dim=100) and synthesizes 64×64 RGB images.
- **Discriminator (D)**: Distinguishes real faces from fakes, pushing G to improve iteratively.

No CNNs, no attention — just pure fully-connected determination.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Noise (z)  │────▶│    Generator     │────▶│  Fake Image      │
│  100-dim    │     │  FC 256→512→1024 │     │  64×64×3         │
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Real Image │────▶│   Discriminator  │────▶│  Real / Fake?    │
│  64×64×3    │     │  FC 1024→512→256 │     │  (Binary Prob)   │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

| Component | Layers | Activation | Output |
|-----------|--------|------------|--------|
| **Generator** | Linear(100→256) → Linear(256→512) → Linear(512→1024) → Linear(1024→12288) | ReLU (hidden), Tanh (final) | 64×64×3 |
| **Discriminator** | Linear(12288→1024) → Linear(1024→512) → Linear(512→256) → Linear(256→1) | LeakyReLU(0.2) (hidden), Sigmoid (final) | Scalar [0,1] |

---

## Dataset

**CelebFaces Attributes (CelebA)** — 202,599 celebrity images, preprocessed:

```
Original: 178×218  ──center crop──▶  178×178  ──resize──▶  64×64  ──normalize──▶  [-1, 1]
```

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 128 |
| Latent Dim (z) | 100 |
| Optimizer | Adam (lr=2e-4, β₁=0.5, β₂=0.999) |
| Loss Function | Binary Cross-Entropy |
| Epochs | 5 |
| Device | MPS / CUDA / CPU (auto-detect) |

### Loss Curves (Qualitative)

Training logs show the tug-of-war between G and D — generator loss trends downward from ~3.3 → ~1.5 over 5 epochs while discriminator loss stabilizes around 0.5, indicating healthy competition.

---

## Results

Generated samples improve dramatically across epochs:

<p align="center">
  <b>Epoch 1</b> — Mostly noise, faint structural hints<br>
  <b>Epoch 2</b> — Color blobs and rough face silhouettes<br>
  <b>Epoch 3</b> — Facial features begin to emerge (eyes, mouth regions)<br>
  <b>Epoch 4</b> — More defined face structures with skin tones<br>
  <b>Epoch 5</b> — Recognizable face-like patterns with reasonable symmetry
</p>

> *Note: 5 epochs is a fraction of what SOTA GANs require. Extended training (50–200 epochs) yields significantly sharper results.*

---

## Getting Started

```bash
# Clone
git clone https://github.com/your-username/vanilla-gan.git
cd vanilla-gan

# Install
pip install torch torchvision matplotlib pillow numpy

# Download CelebA (img_align_celeba) and place in project root
# Ensure the directory structure:
# ├── vanilla_gan_main.ipynb
# └── img_align_celeba/
#     ├── 000001.jpg
#     ├── 000002.jpg
#     └── ...

# Run
jupyter notebook vanilla_gan_main.ipynb
```

---

## Key Takeaways

- **First principles**: Implements the original GAN paper with zero convolutional layers — pure MLP-based generation.
- **Scalable design**: Modular `Generator`/`Discriminator` classes make it easy to swap in DCGAN-style architectures.
- **Device-agnostic**: Auto-selects MPS (Apple Silicon), CUDA, or CPU.
- **Visual progress tracking**: Saves generated samples after each epoch for qualitative assessment.

---

## References

- Goodfellow, I. J., et al. *"Generative Adversarial Nets."* NeurIPS, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Liu, Z., et al. *"Deep Learning Face Attributes in the Wild."* ICCV, 2015. (CelebA Dataset)

---

<p align="center">
  <sub>Built with ❤️ and PyTorch — because sometimes the simplest GAN is the best place to start.</sub>
</p>
