#!/usr/bin/env python
"""
Quick extraction of per-image per-layer drift for SD 1.5 on 19 coco_val images.
Needed for bootstrap CI (Task 4).

Uses same layer discovery + hook pattern as phase1_diagnostics.py.

Saves: outputs/phase9_task4/sd15_per_image_drift.json
Format: {layer_name: {image_name: drift_value}}

Usage:
    python scripts/phase9_task4_sd15_perimage.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_common import load_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_task4"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
NUM_STEPS = 50


# ---------------------------------------------------------------------------
# Layer discovery (mirrors phase1_diagnostics.py)
# ---------------------------------------------------------------------------

def discover_hook_targets(unet):
    """Discover ResNet blocks and attention blocks in UNet."""
    targets = []
    for name, module in unet.named_modules():
        parts = name.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(name)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(name)
    return sorted(targets)


class UNetFeatureHooker:
    """Hook UNet intermediate features for drift analysis."""

    def __init__(self, unet, targets):
        self.unet = unet
        self.features = {}
        self.handles = []
        for name in targets:
            mod = self._find_module(name)
            if mod is not None:
                handle = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook_fn(n, out)
                )
                self.handles.append(handle)

    def _find_module(self, name):
        tokens = name.split(".")
        mod = self.unet
        for t in tokens:
            try:
                mod = getattr(mod, t)
            except AttributeError:
                return None
        return mod

    def _hook_fn(self, name, output):
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 3:
            output = output.mean(dim=1, keepdim=True)
        self.features[name] = output.detach().cpu()

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# Drift measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_per_layer_drift(pipe, img_tensor, prompt_emb, targets):
    """DDIM inversion -> reconstruction, measure per-layer MSE drift at turnaround."""
    vae_dtype = next(pipe.vae.parameters()).dtype

    img_tensor = img_tensor.to(device=DEVICE, dtype=vae_dtype)
    latent = pipe.vae.encode(img_tensor).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    latent = latent.to(dtype=pipe.unet.dtype)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps

    # ---- DDIM inversion ----
    z = latent.clone()
    for i, t in enumerate(timesteps.flip(0)):
        prev_t = timesteps[-2 - i] if i < len(timesteps) - 1 else torch.tensor(-1, device=DEVICE)
        npred = pipe.unet(z, t, encoder_hidden_states=prompt_emb).sample
        ac = scheduler.alphas_cumprod[t]
        an_val = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=DEVICE)
        z = (an_val / ac).sqrt() * z + ((1 - an_val).sqrt() - (an_val / ac).sqrt() * (1 - ac).sqrt()) * npred
    noise = z.clone()

    # Capture inversion features at turnaround (t=1)
    hooker = UNetFeatureHooker(pipe.unet, targets)
    t1 = torch.tensor([1], device=DEVICE, dtype=torch.long)
    pipe.unet(noise, t1, encoder_hidden_states=prompt_emb)
    inv_features = hooker.features.copy()
    hooker.clear()

    # ---- DDIM reconstruction ----
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    z = noise.clone()
    for t in scheduler.timesteps[:-1]:
        npred = pipe.unet(z, t, encoder_hidden_states=prompt_emb).sample
        z = scheduler.step(npred, t, z).prev_sample

    # Capture reconstruction features at same timestep (t=1)
    pipe.unet(z, t1, encoder_hidden_states=prompt_emb)
    recon_features = hooker.features.copy()
    hooker.remove()

    # Compute per-layer drift: MSE between inv and recon features
    drift = {}
    for name in targets:
        if name not in inv_features or name not in recon_features:
            drift[name] = 0.0
            continue
        f_inv = inv_features[name].float()
        f_rec = recon_features[name].float()
        drift[name] = float((f_inv - f_rec).pow(2).mean().item())

    return drift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SD 1.5 pipeline...")
    pipe = load_pipeline()
    empty_emb = pipe.text_encoder(
        pipe.tokenizer("", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    )[0].to(pipe.unet.dtype)

    targets = discover_hook_targets(pipe.unet)
    print(f"Tracking {len(targets)} layers")
    for t in targets[:5]:
        print(f"  {t}")

    paths = sorted(DATA_DIR.glob("coco_*.jpg"))
    print(f"Processing {len(paths)} images...")

    # per_image: {layer_name: {image_name: drift_value}}
    per_image = {t: {} for t in targets}

    for p in tqdm(paths):
        img_name = p.stem
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        img_tensor = transform(Image.open(str(p)).convert("RGB")).unsqueeze(0)

        drift = measure_per_layer_drift(pipe, img_tensor, empty_emb, targets)
        for t in targets:
            per_image[t][img_name] = drift.get(t, 0.0)

        del img_tensor
        torch.cuda.empty_cache()

    # Save
    out_path = OUT_DIR / "sd15_per_image_drift.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "SD 1.5",
            "steps": NUM_STEPS,
            "n_images": len(paths),
            "n_layers": len(targets),
            "images": [p.stem for p in paths],
            "layers": targets,
            "per_image": per_image,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
