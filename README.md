# FaceGen - Vanilla GAN for Face Synthesis

A PyTorch-based **Vanilla Generative Adversarial Network (GAN)** project trained on the **CelebA facial image dataset** to generate realistic human face images from random latent vectors.

---

## Overview

This project implements a classic **Generator–Discriminator adversarial learning framework** to model the distribution of facial images and synthesize new samples.  
The notebook includes:

- custom image loading and preprocessing
- GAN architecture design
- adversarial training loop
- image generation and visualization
- training outputs and generated samples

---

## Features

- **Vanilla GAN architecture** built from scratch in PyTorch
- **Generator** network for synthetic face creation
- **Discriminator** network for real vs fake image classification
- **Binary Cross-Entropy loss** for adversarial training
- **Adam optimizer** for stable convergence
- **Latent space sampling** to generate diverse outputs
- **Image preprocessing pipeline** for CelebA faces
- **64×64 RGB image synthesis**

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PIL

---

## Dataset

The project uses the **CelebA dataset** of facial images.

### Preprocessing applied:
- Center cropping
- Resizing to `64×64`
- Tensor conversion
- Normalization to `[-1, 1]`

> The dataset is not included in this repository.  
> Download it separately and place it in the expected folder before running the notebook.

---

## Model Architecture

### Generator
The Generator takes a random latent vector of size `100` and transforms it into a synthetic `64×64×3` RGB face image using fully connected layers and `ReLU` activations.

### Discriminator
The Discriminator receives an image and outputs a probability score indicating whether the image is real or generated, using fully connected layers and `LeakyReLU` activations.

---

## Training Details

- Loss Function: `BCELoss`
- Optimizer: `Adam`
- Learning Rate: `0.0002`
- Beta Values: `(0.5, 0.999)`
- Batch Size: `128`
- Image Size: `64 × 64`
- Latent Vector Size: `100`

---

## Results

After training, the model generates facial images that gradually improve in realism as the discriminator and generator learn in opposition.

The notebook includes:
- training progress
- loss values
- generated face outputs
