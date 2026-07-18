"""
Shared utilities for controlled DiT training + diagnostics experiment.
Goal: same DiT-S/2 architecture trained with eps-prediction (DDPM) vs flow matching.
"""
from __future__ import annotations

import copy
import math
import os
import random
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

from diffusers import DiTTransformer2DModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train_controlled"

# DiT-S/2 config
DIT_CONFIG = dict(
    sample_size=32,         # latent/image spatial dim
    patch_size=2,           # 2x2 patches → 32² = 1024 tokens for 64x64 input
    in_channels=3,          # RGB
    out_channels=3,         # RGB prediction
    num_layers=12,
    num_attention_heads=6,
    attention_head_dim=64,  # hidden_dim = 6*64 = 384
)

# Training
BATCH_SIZE = 16
TOTAL_STEPS = 50_000
WARMUP_STEPS = 1_000
LEARNING_RATE = 1e-4
EMA_DECAY = 0.9999
GRAD_CLIP = 1.0
LOG_EVERY = 100
SAMPLE_EVERY = 5_000
CHECKPOINT_EVERY = 10_000

# DDPM schedule
DDPM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Diagnostics
DIAG_NUM_STEPS = 50
TEST_IMAGE_SIZE = 64

# Data roots
DATA_ROOTS = [
    PROJECT_ROOT / "data" / "coco_val100",
    PROJECT_ROOT / "data" / "basetest",
    PROJECT_ROOT / "data" / "architecture",
    PROJECT_ROOT / "data" / "portraits",
    PROJECT_ROOT / "data" / "typography",
]
TEST_DATA_ROOT = PROJECT_ROOT / "data" / "coco_val"


# ═══════════════════════════════════════════════════════
# Seed
# ═══════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════
def get_dit_s2_model() -> DiTTransformer2DModel:
    """Create a fresh DiT-S/2 with pixel-space config."""
    set_seed(42)
    model = DiTTransformer2DModel(**DIT_CONFIG)
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════
def _collect_images(roots: list[Path], exclude_names: set | None = None) -> list[Path]:
    paths = []
    for root in roots:
        if root.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in root.glob(ext):
                    if exclude_names is None or p.name not in exclude_names:
                        paths.append(p)
    return sorted(paths)


def make_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def make_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def tensor_to_np_uint8(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor → uint8 numpy for SSIM/LPIPS saving."""
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = ((t * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return t.permute(1, 2, 0).numpy()


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_train_loader() -> DataLoader:
    # Exclude test images from training
    test_names = {p.name for p in _collect_images([TEST_DATA_ROOT])}
    paths = _collect_images(DATA_ROOTS, exclude_names=test_names)
    print(f"[Data] Training images: {len(paths)} (excluded {len(test_names)} test images)")
    ds = ImageDataset(paths, make_train_transform())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                       drop_last=False, pin_memory=True)


def get_test_loader() -> DataLoader:
    paths = _collect_images([TEST_DATA_ROOT])
    print(f"[Data] Test images: {len(paths)}")
    ds = ImageDataset(paths, make_test_transform())
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


# ═══════════════════════════════════════════════════════
# Noise schedules
# ═══════════════════════════════════════════════════════
class NoiseScheduleDDPM:
    """Precompute DDPM betas/alphas/alpha_bars for 1000 steps."""
    def __init__(self, T: int = DDPM_TIMESTEPS):
        self.T = T
        self.betas = torch.linspace(BETA_START, BETA_END, T, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise"""
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1mab = (1.0 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise


class FlowPath:
    """Straight-line trajectory: x_t = (1-t)*x0 + t*noise, velocity = noise - x0."""
    @staticmethod
    def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """t ∈ [0, 1]"""
        t_r = t.view(-1, 1, 1, 1)
        return (1.0 - t_r) * x0 + t_r * noise

    @staticmethod
    def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """v = d(x_t)/dt = noise - x0"""
        return noise - x0


# ═══════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════
class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_decay = min(self.decay, (1.0 + self._step) / (10.0 + self._step)) \
            if hasattr(self, '_step') else self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# ═══════════════════════════════════════════════════════
# Feature hooks (for diagnostics)
# ═══════════════════════════════════════════════════════
def discover_dit_hook_targets(model: DiTTransformer2DModel) -> list[str]:
    targets = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
            targets.append(name)
    return sorted(targets, key=lambda x: int(x.split(".")[1]))


class TransformerFeatureHooker:
    def __init__(self, model: DiTTransformer2DModel):
        self.model = model
        self.features: dict[str, torch.Tensor] = {}
        self.handles = []

    def _find_module(self, name: str):
        mod = self.model
        for part in name.split("."):
            try:
                mod = getattr(mod, part)
            except AttributeError:
                return None
        return mod

    def register(self):
        for name in discover_dit_hook_targets(self.model):
            mod = self._find_module(name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self.features.__setitem__(n, out)
                )
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ═══════════════════════════════════════════════════════
# DDIM Inversion/Reconstruction (eps-prediction)
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def ddim_inversion_eps(
    model: DiTTransformer2DModel,
    x0: torch.Tensor,
    num_steps: int = DIAG_NUM_STEPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DDIM inversion for eps-prediction model. Returns (z_T, features_at_turnaround)."""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler(num_train_timesteps=DDPM_TIMESTEPS,
                          beta_start=BETA_START, beta_end=BETA_END,
                          prediction_type="epsilon")
    sched.set_timesteps(num_steps, device=DEVICE)
    timesteps = sched.timesteps

    hooker = TransformerFeatureHooker(model)
    hooker.register()

    z = x0.clone()
    # Collect features at each step; keep only last step
    all_features = []
    extended_ts = timesteps.tolist() + [0]

    for i in range(len(extended_ts) - 1, 0, -1):
        t_cur = extended_ts[i]
        t_next = extended_ts[i - 1]
        cl = torch.zeros(z.shape[0], dtype=torch.long, device=DEVICE)
        eps_pred = model(z, timestep=torch.tensor([t_cur], device=DEVICE),
                         class_labels=cl).sample

        alpha_cur = sched.alphas_cumprod[t_cur]
        alpha_next = sched.alphas_cumprod[t_next]
        sigma_cur = (1.0 - alpha_cur).sqrt()
        sigma_next = (1.0 - alpha_next).sqrt()

        x0_pred = (z - sigma_cur * eps_pred) / alpha_cur.sqrt().clamp(min=1e-8)
        z = alpha_next.sqrt() * x0_pred + sigma_next * eps_pred

    # Collect features after last step
    features = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}
    hooker.remove()
    return z, features


@torch.no_grad()
def ddim_reconstruction_eps(
    model: DiTTransformer2DModel,
    noise: torch.Tensor,
    num_steps: int = DIAG_NUM_STEPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DDIM reconstruction for eps-prediction model."""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler(num_train_timesteps=DDPM_TIMESTEPS,
                          beta_start=BETA_START, beta_end=BETA_END,
                          prediction_type="epsilon")
    sched.set_timesteps(num_steps, device=DEVICE)
    timesteps = sched.timesteps

    hooker = TransformerFeatureHooker(model)
    hooker.register()

    z = noise.clone()
    for t in timesteps:
        cl = torch.zeros(z.shape[0], dtype=torch.long, device=DEVICE)
        eps_pred = model(z, timestep=torch.tensor([t], device=DEVICE),
                         class_labels=cl).sample
        z = sched.step(eps_pred, t, z).prev_sample

    features = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}
    hooker.remove()
    return z, features


# ═══════════════════════════════════════════════════════
# Flow Euler Inversion/Reconstruction
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def flow_inversion(
    model: DiTTransformer2DModel,
    x0: torch.Tensor,
    num_steps: int = DIAG_NUM_STEPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward Euler ODE: dz/dt = v(z,t), from t=0 to t=1."""
    dt = 1.0 / num_steps

    hooker = TransformerFeatureHooker(model)
    hooker.register()

    z = x0.clone()
    for i in range(num_steps):
        t_val = i * dt
        t_tensor = torch.tensor([int(t_val * 999)], device=DEVICE)
        cl = torch.zeros(z.shape[0], dtype=torch.long, device=DEVICE)
        v_pred = model(z, timestep=t_tensor, class_labels=cl).sample
        z = z + v_pred * dt

    features = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}
    hooker.remove()
    return z, features


@torch.no_grad()
def flow_reconstruction(
    model: DiTTransformer2DModel,
    noise: torch.Tensor,
    num_steps: int = DIAG_NUM_STEPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backward Euler ODE: z_0 from noise by integrating v reversed."""
    dt = 1.0 / num_steps

    hooker = TransformerFeatureHooker(model)
    hooker.register()

    z = noise.clone()
    for i in range(num_steps, 0, -1):
        t_val = i * dt
        t_tensor = torch.tensor([int(t_val * 999)], device=DEVICE)
        cl = torch.zeros(z.shape[0], dtype=torch.long, device=DEVICE)
        v_pred = model(z, timestep=t_tensor, class_labels=cl).sample
        z = z - v_pred * dt

    features = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}
    hooker.remove()
    return z, features


# ═══════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════
def compute_image_metrics(orig: torch.Tensor, recon: torch.Tensor,
                          lpips_fn=None) -> dict:
    """PSNR, SSIM, LPIPS between [-1,1] tensors."""
    o = orig.detach()
    r = recon.detach()
    o01 = (o * 0.5 + 0.5).clamp(0, 1)
    r01 = (r * 0.5 + 0.5).clamp(0, 1)

    mse_val = F.mse_loss(o01, r01).item()
    psnr = 20.0 * math.log10(1.0) - 10.0 * math.log10(max(mse_val, 1e-12))

    o_np = o01.squeeze(0).permute(1, 2, 0).cpu().numpy()
    r_np = r01.squeeze(0).permute(1, 2, 0).cpu().numpy()
    s_val = ssim(o_np, r_np, channel_axis=2, data_range=1.0)

    lpips_val = None
    if lpips_fn is not None:
        lpips_dev = next(lpips_fn.parameters()).device
        lpips_val = lpips_fn(o01.to(lpips_dev), r01.to(lpips_dev)).item()

    return {"PSNR": psnr, "SSIM": s_val, "LPIPS": lpips_val}


# ═══════════════════════════════════════════════════════
# Structural distance (same 4-feature as paper)
# ═══════════════════════════════════════════════════════
def extract_structural_features(drift: np.ndarray) -> np.ndarray:
    """Extract 4 features from per-layer drift vector [L].
    Features: peak_position(relative), n_peaks, concentration(gini), spread(std/mean).
    """
    L = len(drift)
    d = np.maximum(drift, 0.0)  # non-negative
    d_max = d.max()
    if d_max < 1e-12:
        return np.array([0.5, 0.0, 0.0, 0.0])

    # Peak position (0–1 relative)
    peak_pos = float(np.argmax(d)) / max(L - 1, 1)

    # Number of peaks (exceeding 70% of max)
    n_peaks = float(np.sum(d > 0.7 * d_max))

    # Gini concentration (0=uniform, 1=one layer)
    d_sorted = np.sort(d)
    n = len(d_sorted)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * d_sorted)) / (n * np.sum(d_sorted) + 1e-12) - (n + 1) / n

    # Spread: std / mean
    spread = float(np.std(d) / (np.mean(d) + 1e-12))

    return np.array([peak_pos, n_peaks, gini, spread])


def structural_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    f1 = extract_structural_features(d1)
    f2 = extract_structural_features(d2)
    return float(np.linalg.norm(f1 - f2))


# ═══════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════
def plot_drift_comparison(eps_drift: dict, flow_drift: dict, out_path: str):
    layers = sorted(eps_drift.keys(), key=lambda x: int(x.split(".")[-1]))
    eps_vals = [eps_drift[k] for k in layers]
    flow_vals = [flow_drift[k] for k in layers]
    labels = [f"b.{k.split('.')[-1]}" for k in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, eps_vals, w, label="eps-prediction (DDPM)", color="#4472C4", alpha=0.85)
    ax.bar(x + w/2, flow_vals, w, label="flow matching", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Feature Drift (MSE)")
    ax.set_title("DiT-S/2 Layer-wise Drift: eps vs flow matching")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(eps_losses: list[float], flow_losses: list[float], out_path: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(eps_losses, label="eps-prediction (DDPM)", color="#4472C4", alpha=0.8, lw=0.6)
    ax.plot(flow_losses, label="flow matching", color="#ED7D31", alpha=0.8, lw=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: eps-prediction vs flow matching")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
