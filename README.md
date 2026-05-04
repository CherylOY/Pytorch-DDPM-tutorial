# Pytorch DDPM Tutorial

A minimal PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** trained on the MNIST dataset.

> Based on: [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103) — Stanley Chan, 2025

---

## Overview

This project implements a DDPM from scratch, including a lightweight U-Net noise predictor and a linear noise schedule. The model learns to generate handwritten digits by iteratively denoising samples starting from pure Gaussian noise.

---

## Project Structure

```
├── model.py        # U-Net architecture (network definition)
├── inference.py    # Diffusion schedule, training loop, and sampling
├── DDPM.ipynb      # Original notebook (end-to-end walkthrough)
└── ddpm_unet_outputs/  # Sample images saved during training
```

---

## Model Architecture

The noise predictor is a small U-Net with sinusoidal timestep embeddings:

```
Input (1, 28, 28)
  └─ init_conv        →  (32, 28, 28)
      ├─ DownBlock 1  →  (64, 14, 14)   + skip connection
      │   └─ DownBlock 2  →  (128, 7, 7)   + skip connection
      │       └─ Bottleneck (mid1 + mid2)
      │   └─ UpBlock 1    →  (64, 14, 14)
      └─ UpBlock 2        →  (32, 28, 28)
  └─ final_conv       →  (1, 28, 28)   ← predicted noise ε̂
```

Each block is conditioned on the diffusion timestep via a sinusoidal embedding projected into the feature maps.

---

## Diffusion Process

**Forward process** — gradually adds Gaussian noise over T steps:

$$q(x_t | x_0) = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

**Reverse process** — the U-Net predicts the noise at each step so we can recover $x_0$:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\!\left(\mu_\theta(x_t, t),\; \sigma_t^2 I\right)$$

**Training objective** — minimise MSE between true and predicted noise:

$$\mathcal{L} = \mathbb{E}_{x_0,\, \varepsilon,\, t}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Timesteps T | 200 |
| Image size | 28 × 28 |
| Batch size | 128 |
| Epochs | 150 |
| Learning rate | 1e-3 |
| Base channels | 32 |
| Time embedding dim | 128 |
| β schedule | Linear (1e-4 → 0.06) |

---

## Getting Started

**Install dependencies**

```bash
pip install torch torchvision matplotlib
```

**Train the model**

```bash
python inference.py
```

Sample grids are saved to `ddpm_unet_outputs/` after every epoch.

**Run the notebook**

```bash
jupyter notebook DDPM.ipynb
```

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- matplotlib

---

## Reference

Stanley H. Chan. *Tutorial on Diffusion Models for Imaging and Vision*. arXiv:2403.18103, 2025.
