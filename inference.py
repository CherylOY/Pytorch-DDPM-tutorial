# =============================================================================
# inference.py — DDPM Training & Sampling Pipeline
# =============================================================================
# This file covers everything outside the network definition:
#   1.  Imports & configuration
#   2.  Dataset preparation
#   3.  Noise schedule (linear beta schedule)
#   4.  Forward diffusion  q(x_t | x_0)   — adding noise
#   5.  Training loss      (MSE on predicted noise)
#   6.  Reverse diffusion  p(x_{t-1} | x_t) — denoising step
#   7.  Full sampling loop (from pure Gaussian noise to a clean image)
#   8.  Training loop
#   9.  Loss curve visualisation
# =============================================================================

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# Import the U-Net architecture defined in model.py
from model import SmallUNet


# =============================================================================
# Step 1: Device & Hyperparameters
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATA_ROOT  = "./data"
BATCH_SIZE = 128
EPOCHS     = 150
LR         = 1e-3
TIMESTEPS  = 200        # Total diffusion steps T (can be increased to 500 / 1000)
IMAGE_SIZE = 28
CHANNELS   = 1
SAVE_DIR   = "ddpm_unet_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Step 2: Dataset Preparation
# =============================================================================
# Load MNIST, rescale pixel values from [0, 1] to [-1, 1] so that the
# data range matches the zero-mean Gaussian prior used in diffusion.

transform = transforms.Compose([
    transforms.ToTensor(),                       # → [0, 1]
    transforms.Lambda(lambda x: x * 2 - 1),     # → [-1, 1]
])

dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)


# =============================================================================
# Step 3: Noise Schedule (Linear Beta Schedule)
# =============================================================================
# β_t increases linearly from β_start to β_end over T steps.
# From β_t we derive:
#   α_t        = 1 − β_t
#   ᾱ_t        = ∏_{s=1}^{t} α_s   (cumulative product)
#   √ᾱ_t       — scale factor for the clean signal in q(x_t | x_0)
#   √(1 − ᾱ_t) — scale factor for the noise in q(x_t | x_0)
# =============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.06):
    """Return a 1-D tensor of β values linearly spaced from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


# Pre-compute all schedule constants and move them to the target device
betas      = linear_beta_schedule(TIMESTEPS).to(device)          # β_t
alphas     = 1.0 - betas                                          # α_t
alpha_bars = torch.cumprod(alphas, dim=0)                         # ᾱ_t
# ᾱ_{t-1}: pad with 1.0 at position 0 (i.e. ᾱ_0 ≡ 1)
alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

sqrt_alpha_bars           = torch.sqrt(alpha_bars)                # √ᾱ_t
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)          # √(1 − ᾱ_t)
sqrt_recip_alphas         = torch.sqrt(1.0 / alphas)              # 1 / √α_t

# Posterior variance σ²_t = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Index schedule tensor `a` at positions `t` and reshape for broadcasting.

    Args:
        a       : 1-D schedule tensor of length T.
        t       : Batch of timestep indices, shape (B,).
        x_shape : Shape of the target tensor, e.g. (B, C, H, W).
    Returns:
        Tensor of shape (B, 1, 1, 1) — broadcasts over C, H, W.
    """
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


# =============================================================================
# Step 4: Model, Optimiser
# =============================================================================

model     = SmallUNet(in_channels=1, base_channels=32, time_emb_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =============================================================================
# Step 5: Forward Diffusion  q(x_t | x_0)
# =============================================================================
# Given a clean image x_0 and a timestep t, produce the noisy image x_t:
#
#   x_t = √ᾱ_t · x_0  +  √(1 − ᾱ_t) · ε,    ε ~ N(0, I)
#
# This is the "reparameterised" closed-form that lets us sample any t directly
# without iterating through the Markov chain.
# =============================================================================

def q_sample(x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Add noise to x_start at diffusion step t.

    Args:
        x_start : Clean images x_0,  shape (B, C, H, W).
        t       : Timestep indices,   shape (B,).
        noise   : Optional pre-sampled Gaussian noise; drawn fresh if None.
    Returns:
        Noisy images x_t,  shape (B, C, H, W).
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_ab_t  = extract(sqrt_alpha_bars,           t, x_start.shape)   # √ᾱ_t
    sqrt_omb_t = extract(sqrt_one_minus_alpha_bars, t, x_start.shape)   # √(1 − ᾱ_t)

    return sqrt_ab_t * x_start + sqrt_omb_t * noise


# =============================================================================
# Step 6: Training Loss
# =============================================================================
# The DDPM objective is to minimise the MSE between the true noise ε and the
# noise predicted by the network:
#
#   L = E_{x_0, ε, t} [ ‖ε − ε̂_θ(x_t, t)‖² ]
#
# At each iteration we:
#   1. Sample a random noise tensor ε.
#   2. Construct x_t via q_sample.
#   3. Ask the model to predict ε given (x_t, t).
#   4. Compute MSE.
# =============================================================================

def p_losses(model: nn.Module, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the denoising loss for a batch.

    Args:
        model   : The U-Net noise predictor.
        x_start : Clean images x_0,  shape (B, C, H, W).
        t       : Randomly sampled timesteps, shape (B,).
    Returns:
        Scalar MSE loss.
    """
    noise           = torch.randn_like(x_start)            # Sample ε ~ N(0, I)
    x_noisy         = q_sample(x_start, t, noise)          # x_t via closed form
    predicted_noise = model(x_noisy, t)                    # ε̂_θ(x_t, t)
    return F.mse_loss(predicted_noise, noise)              # ‖ε − ε̂‖²


# =============================================================================
# Step 7: Reverse Diffusion — Single Denoising Step  p(x_{t-1} | x_t)
# =============================================================================
# Given x_t and the predicted noise, compute the posterior mean and sample
# x_{t-1}:
#
#   μ_θ(x_t, t) = (1/√α_t) · (x_t  −  β_t/√(1−ᾱ_t) · ε̂_θ)
#   x_{t-1}     = μ_θ  +  σ_t · z,    z ~ N(0, I)   (z = 0 when t = 0)
# =============================================================================

@torch.no_grad()
def p_sample(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Perform one reverse diffusion step: sample x_{t-1} given x_t.

    Args:
        model : Trained U-Net.
        x     : Current noisy samples x_t, shape (B, C, H, W).
        t     : Current timestep indices,  shape (B,).
    Returns:
        Denoised samples x_{t-1}, shape (B, C, H, W).
    """
    # Retrieve schedule values for this batch of timesteps
    betas_t                  = extract(betas,                      t, x.shape)
    sqrt_one_minus_ab_t      = extract(sqrt_one_minus_alpha_bars,  t, x.shape)
    sqrt_recip_alphas_t      = extract(sqrt_recip_alphas,          t, x.shape)

    # Predict noise and compute the posterior mean μ_θ
    predicted_noise = model(x, t)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_ab_t
    )

    # Add stochastic noise only for t > 0 (at t = 0 we take the mean directly)
    posterior_var_t = extract(posterior_variance, t, x.shape)
    noise           = torch.randn_like(x)
    nonzero_mask    = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))

    return model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var_t, min=1e-20)) * noise


# =============================================================================
# Step 8: Full Sampling Loop
# =============================================================================
# Start from pure Gaussian noise x_T ~ N(0, I) and iteratively apply the
# reverse step from t = T-1 down to t = 0, recovering a clean image.
# Finally, rescale from [-1, 1] back to [0, 1] for saving/displaying.
# =============================================================================

@torch.no_grad()
def sample(model: nn.Module, n: int = 16) -> torch.Tensor:
    """
    Generate n images from scratch using the trained DDPM.

    Args:
        model : Trained U-Net.
        n     : Number of images to generate.
    Returns:
        Generated images in [0, 1], shape (n, C, H, W).
    """
    model.eval()

    # Start from pure Gaussian noise  x_T ~ N(0, I)
    x = torch.randn((n, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device)

    # Iteratively denoise: t = T-1, T-2, …, 1, 0
    for i in reversed(range(TIMESTEPS)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t)

    # Rescale from [-1, 1] → [0, 1] and clip to valid range
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x


# =============================================================================
# Step 9: Training Loop
# =============================================================================
# For each epoch:
#   - Iterate over all mini-batches.
#   - Sample random timesteps t ~ Uniform[0, T).
#   - Compute p_losses and back-propagate.
#   - After each epoch, generate a sample grid and save it to SAVE_DIR.
# =============================================================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for step, (x, _) in enumerate(dataloader):
        x = x.to(device)
        # Sample a random diffusion timestep for each image in the batch
        t = torch.randint(0, TIMESTEPS, (x.shape[0],), device=device).long()

        loss = p_losses(model, x, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  Step {step:4d}  Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} finished — avg loss = {avg_loss:.4f}")

    # Generate and save a 4×4 sample grid after every epoch
    samples = sample(model, n=16).cpu()
    grid    = utils.make_grid(samples, nrow=4)
    utils.save_image(grid, os.path.join(SAVE_DIR, f"epoch_{epoch+1}.png"))

print("Training finished.")


# =============================================================================
# Step 10: Final Sample Visualisation
# =============================================================================
# Generate 16 images and display them in a 4×4 grid using matplotlib.

samples = sample(model, n=16).cpu()
grid    = utils.make_grid(samples, nrow=4).permute(1, 2, 0).squeeze()

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap="gray")
plt.axis("off")
plt.title("DDPM Generated Samples")
plt.tight_layout()
plt.show()


# =============================================================================
# Step 11: Loss Curve Visualisation  (paste recorded loss_history here)
# =============================================================================
# Replace the list below with the actual per-epoch average losses logged
# during your training run.

loss_history = [
    0.0650, 0.0402, 0.0359, 0.0340, 0.0333, 0.0323, 0.0319, 0.0314, 0.0310, 0.0310,
    0.0309, 0.0304, 0.0304, 0.0301, 0.0300, 0.0301, 0.0296, 0.0295, 0.0295, 0.0292,
    0.0290, 0.0289, 0.0290, 0.0290, 0.0290, 0.0287, 0.0286, 0.0287, 0.0287, 0.0289,
    # ... (continue for all 150 epochs)
]

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.title("DDPM Training Loss Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
