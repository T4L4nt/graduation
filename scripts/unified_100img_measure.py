#!/usr/bin/env python
"""
Unified 100-image Architecture Fingerprint Measurement.
======================================================

Measures per-layer feature drift for all 6 ICLR architectures on 104 coco_val100
images using a consistent protocol (50 DDIM/Euler steps, empty prompt, fp16).

Motivation: reviewer audit — core fingerprint claims were based on 19 images;
scaling to 100+ is the #1 priority to address evidence-scale criticism.

Architectures:
  SD 1.5    — UNet (38 layers), DDIM 50-step, 512×512
  SDXL      — UNet (28 layers), DDIM 50-step, 1024×1024
  HunyuanDiT — DiT single-stream (40 layers), DDIM v_pred 50-step, 1024×1024
  FLUX      — MM-DiT dual-stream (57 layers), Euler 50-step, variable
  SD 3.5    — MM-DiT-X (24 layers), Euler 50-step, 1024×1024
  PixArt-Σ  — DiT cross-attn (28 layers), DDIM 50-step, 1024×1024

Usage:
    python scripts/unified_100img_measure.py                    # all 6 architectures
    python scripts/unified_100img_measure.py --arch SD1.5       # single architecture
    python scripts/unified_100img_measure.py --images 20        # subset for testing
    python scripts/unified_100img_measure.py --resume           # skip completed

Output: outputs/unified_100img/{arch}_drift.json
        outputs/unified_100img/cross_arch_v2_matrix.json
"""

import argparse, json, copy, sys, time, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
OUT_DIR = PROJECT_ROOT / "outputs" / "unified_100img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
N_STEPS = 50  # consistent 50-step protocol across all architectures

# ─── v2 Feature Extraction ───────────────────────────────────────────

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    return float((2 * np.sum(np.arange(1, n+1) * x) - (n+1) * s) / (n * s)) if s > 0 else 0.0

def extract_v2_features(profile, layer_names=None):
    """v2 continuous features from drift profile (no peak_count)."""
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p - dmin) / (dmax - dmin) if dmax > dmin else p.copy()
    pp = float(np.argmax(pn)) / L
    k = max(1, int(np.ceil(0.2 * L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top]) / np.sum(pn))
    sp = float(gini(pn))
    peaks, props = find_peaks(pn, prominence=0.1)
    n_peaks = len(peaks)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks) > 0 else 0.0
    peak_layer = layer_names[int(np.argmax(pn))] if layer_names else f"layer_{int(np.argmax(pn))}"
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "n_peaks": n_peaks, "top_prominence": top_prom,
            "L": L, "peak_layer": peak_layer}

def distance_v2(fa, fb):
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"] - fb["concentration"],
                            fa["spread"] - fb["spread"]])
    d_total = float(np.linalg.norm([d_pp, d_mag]))
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_mag": d_mag}

# ─── Generic Hook Infrastructure ─────────────────────────────────────

class FeatureExtractor:
    """Register hooks, capture intermediate features during forward pass."""
    def __init__(self, model, targets):
        self.model = model
        self.targets = set(targets)
        self.features = {}
        self.handles = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0].detach().float().cpu()
            else:
                self.features[name] = output.detach().float().cpu()
        return fn

    def register(self):
        self.remove()
        for n, m in self.model.named_modules():
            if n in self.targets:
                self.handles.append(m.register_forward_hook(self._hook_fn(n)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.features.clear()

def discover_unet_targets(unet):
    """Discover UNet hook targets: resnets (last numeric suffix) and transformer_blocks[0]."""
    targets = []
    for n, m in unet.named_modules():
        parts = n.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(n)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(n)
    return sorted(targets)

def discover_transformer_targets(model, prefix="", block_attr="transformer_blocks"):
    """Discover transformer block hook targets."""
    targets = []
    for n, m in model.named_modules():
        if block_attr in n and n.count(".") == n.count(f".{block_attr}") + n.split(f".{block_attr}")[-1].count("."):
            # Heuristic: select one representative layer per block
            if "attn" in n or "ff" in n or "proj" in n:
                continue
            targets.append(n)
    if not targets:
        # Fallback: grab all named modules at a certain depth
        for n, m in model.named_modules():
            if block_attr in n:
                targets.append(n)
    return sorted(targets)[:200]  # safety cap

# ─── DDIM Inversion / Reconstruction ─────────────────────────────────

def ddim_invert(pipe, latent, conditioning, steps):
    """DDIM inversion: x_0 → x_T."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = latent.clone()
    extended = timesteps.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended) - 1, 0, -1):
            t_curr, t_next = extended[i], extended[i-1]
            noise_pred = pipe.unet(z, t_curr, **conditioning).sample
            ac, an = scheduler.alphas_cumprod[t_curr], scheduler.alphas_cumprod[t_next]
            c1 = (an / ac).sqrt()
            sc = (1 - ac).sqrt()
            sn = (1 - an).sqrt()
            z = c1 * z + (sn - c1 * sc) * noise_pred
    return z

def ddim_reconstruct(pipe, noise, conditioning, steps):
    """DDIM reconstruction: x_T → x_0."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=DEVICE)
    z = noise.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = pipe.unet(z, t, **conditioning).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z

def encode_image_sd(pipe, img_path, size=512):
    """Load image, encode to VAE latent. Returns latent + (C,H,W) tensor."""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.LANCZOS)
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=pipe.dtype)
    t = 2 * t - 1
    with torch.no_grad():
        lat = pipe.vae.encode(t).latent_dist.sample()
    return lat * pipe.vae.config.scaling_factor, t

# ─── Per-Architecture Measurement ────────────────────────────────────

def measure_sd15(images, steps=N_STEPS):
    """SD 1.5: UNet, DDIM, 512×512, empty prompt."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"  Loading {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    # Encode empty prompt
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        pe = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
    conditioning = {"encoder_hidden_states": pe}

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  SD1.5"):
        latent, _ = encode_image_sd(pipe, img_path, size=512)
        extractor.features.clear()
        z_inv = ddim_invert(pipe, latent, conditioning, steps)
        inv_f = {k: v.clone() for k, v in extractor.features.items()}
        extractor.features.clear()
        _ = ddim_reconstruct(pipe, z_inv, conditioning, steps)
        recon_f = {k: v.clone() for k, v in extractor.features.items()}
        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift
    extractor.remove()
    del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets

def measure_sdxl(images, steps=N_STEPS):
    """SDXL: UNet, DDIM, 1024×1024, empty prompt.
    SDXL uses text_encoder_2 (CLIP-G) hidden states for cross-attention,
    plus pooled embeddings from both encoders as added_cond_kwargs."""
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"  Loading {model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(torch.float32)  # SDXL VAE produces NaN in fp16
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    # Encode empty prompt using pipeline's built-in method (handles all SDXL nuances)
    (prompt_embeds, _, pooled_prompt_embeds, _) = pipe.encode_prompt(
        prompt="", prompt_2="", device=DEVICE, num_images_per_prompt=1,
        do_classifier_free_guidance=False)
    add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=DEVICE, dtype=torch.float16)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
    conditioning = {"encoder_hidden_states": prompt_embeds,
                    "added_cond_kwargs": added_cond_kwargs}

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  SDXL"):
        # SDXL VAE is fp32, must encode in fp32 to avoid NaN
        img = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor
        latent = latent.to(torch.float16)
        extractor.features.clear()
        z_inv = ddim_invert(pipe, latent, conditioning, steps)
        inv_f = {k: v.clone() for k, v in extractor.features.items()}
        extractor.features.clear()
        _ = ddim_reconstruct(pipe, z_inv, conditioning, steps)
        recon_f = {k: v.clone() for k, v in extractor.features.items()}
        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift
    extractor.remove()
    del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


def measure_hunyuandit(images, steps=N_STEPS):
    """HunyuanDiT: single-stream Transformer, DDIM v_pred, 1024×1024.
    v_prediction: v = sqrt(alpha)*eps - sqrt(1-alpha)*x_0
    → x_0 = sqrt(alpha)*z - sqrt(1-alpha)*v
    → eps = sqrt(1-alpha)*z + sqrt(alpha)*v
    Output: 8 channels (learn_sigma=True), first 4 = prediction."""
    from diffusers import HunyuanDiTPipeline, DDIMScheduler
    model_id = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    print(f"  Loading {model_id}...")
    pipe = HunyuanDiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, prediction_type="v_prediction")
    pipe.vae.to(torch.float32)
    # Discover transformer blocks
    targets = []
    for n, m in pipe.transformer.named_modules():
        if n.startswith("blocks.") and n.count(".") == 1:
            targets.append(n)
    targets = sorted(targets, key=lambda x: int(x.replace("blocks.", "")))
    extractor = FeatureExtractor(pipe.transformer, targets)
    extractor.register()
    # Encode empty prompt
    prompt_embeds, _, prompt_attn, _ = pipe.encode_prompt(
        prompt="", device=DEVICE, dtype=torch.float16,
        num_images_per_prompt=1, do_classifier_free_guidance=False)
    prompt_embeds_2, _, prompt_attn_2, _ = pipe.encode_prompt(
        prompt="", device=DEVICE, dtype=torch.float16,
        num_images_per_prompt=1, do_classifier_free_guidance=False, text_encoder_index=1)

    def _forward(z, t_int):
        """Call transformer with tensor timestep (not int), taking first 4 channels."""
        t_tensor = torch.tensor([t_int], device=DEVICE, dtype=torch.float16)
        out = pipe.transformer(
            hidden_states=z,
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attn,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attn_2,
            timestep=t_tensor,
            return_dict=False)[0]
        return out[:, :4] if out.shape[1] >= 8 else out

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  HunyuanDiT"):
        img = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor
        latent = latent.to(torch.float16)

        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = scheduler.timesteps
        z = latent.clone()
        extended = timesteps.tolist() + [0]

        # Inversion (v_prediction-aware)
        extractor.features.clear()
        with torch.no_grad():
            for i in range(len(extended) - 1, 0, -1):
                tc, tn = extended[i], extended[i-1]
                v_pred = _forward(z, tc)
                alpha_c = scheduler.alphas_cumprod[tc]
                alpha_n = scheduler.alphas_cumprod[tn]
                sigma_c = (1 - alpha_c).sqrt()
                sigma_n = (1 - alpha_n).sqrt()
                # v_pred → x0_pred, eps_pred
                x0 = alpha_c.sqrt() * z - sigma_c * v_pred
                eps = sigma_c * z + alpha_c.sqrt() * v_pred
                z = alpha_n.sqrt() * x0 + sigma_n * eps
        inv_f = {k: v.clone() for k, v in extractor.features.items()}

        # Reconstruction
        extractor.features.clear()
        with torch.no_grad():
            for t_ in timesteps:
                noise_pred = _forward(z, t_.item())
                z = scheduler.step(noise_pred, t_, z).prev_sample
        recon_f = {k: v.clone() for k, v in extractor.features.items()}

        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift

    extractor.remove()
    del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


def measure_flux(images, steps=N_STEPS):
    """FLUX: MM-DiT dual-stream, Euler, variable resolution."""
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flux_common import load_flux_pipeline, flux_invert, compute_block_drift

    print("  Loading FLUX (fp16, offload T5)...")
    pipe = load_flux_pipeline(device=DEVICE, dtype=torch.float16, offload_t5=True)

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  FLUX"):
        img = Image.open(str(img_path)).convert("RGB")
        out = flux_invert(pipe, img, num_steps=steps, extract_features=True)
        drift = compute_block_drift(out["features_inv"], out["features_recon"])
        for name, d in drift.items():
            per_image[f"hidden_{name}"][img_path.name] = d["hidden_drift"]
        del out; torch.cuda.empty_cache()

    del pipe; torch.cuda.empty_cache()
    # FLUX block names: joint_0..joint_18, single_0..single_37
    all_names = sorted(per_image.keys())
    return dict(per_image), all_names


def measure_sd35(images, steps=N_STEPS):
    """SD 3.5: MM-DiT-X, Euler, 1024×1024."""
    from diffusers import StableDiffusion3Pipeline
    model_id = "stabilityai/stable-diffusion-3.5-large"
    print(f"  Loading {model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    # Discover transformer blocks
    targets = []
    for n, m in pipe.transformer.named_modules():
        if n.startswith("transformer_blocks.") and n.count(".") == 1:
            targets.append(n)
    targets = sorted(targets, key=lambda x: int(x.replace("transformer_blocks.", "")))
    extractor = FeatureExtractor(pipe.transformer, targets)
    extractor.register()
    # Encode empty prompt
    (prompt_embeds, _, pooled, _) = pipe.encode_prompt(
        prompt="", prompt_2="", prompt_3="", device=DEVICE,
        num_images_per_prompt=1, do_classifier_free_guidance=False)

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  SD3.5"):
        img = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = scheduler.timesteps
        z = latent.clone()
        extended = timesteps.tolist() + [0]

        # Inversion (Euler: z_{t+1} = z_t + (sigma_{t+1} - sigma_t) * v_pred)
        extractor.features.clear()
        with torch.no_grad():
            for i in range(len(extended) - 1, 0, -1):
                tc, tn = extended[i], extended[i-1]
                npred = pipe.transformer(
                    z, tc,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=False)[0]
                sig_curr = scheduler.sigmas[scheduler.timesteps.tolist().index(tc)] if tc in scheduler.timesteps.tolist() else scheduler.sigmas[0]
                sig_next = scheduler.sigmas[scheduler.timesteps.tolist().index(tn)] if tn in scheduler.timesteps.tolist() else scheduler.sigmas[0]
                z = z + (sig_next - sig_curr) * npred
        inv_f = {k: v.clone() for k, v in extractor.features.items()}

        # Reconstruction
        extractor.features.clear()
        with torch.no_grad():
            for t_ in timesteps:
                npred = pipe.transformer(
                    z, t_,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled,
                    return_dict=False)[0]
                z = scheduler.step(npred, t_, z).prev_sample
        recon_f = {k: v.clone() for k, v in extractor.features.items()}

        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift

    extractor.remove()
    del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


def measure_pixart(images, steps=N_STEPS):
    """PixArt-Σ: DiT with cross-attention, DDIM, 1024×1024."""
    from diffusers import PixArtSigmaPipeline, DDIMScheduler
    model_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    print(f"  Loading {model_id}...")
    pipe = PixArtSigmaPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # Discover transformer blocks
    targets = []
    for n, m in pipe.transformer.named_modules():
        if n.startswith("transformer_blocks.") and n.count(".") == 1:
            targets.append(n)
    targets = sorted(targets, key=lambda x: int(x.replace("transformer_blocks.", "")))
    extractor = FeatureExtractor(pipe.transformer, targets)
    extractor.register()
    # Encode empty prompt
    (prompt_embeds, prompt_attn, _, _) = pipe.encode_prompt(
        prompt="", do_classifier_free_guidance=False, num_images_per_prompt=1,
        device=DEVICE)

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc="  PixArt-Sigma"):
        img = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor

        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = scheduler.timesteps
        z = latent.clone()
        # Keep as tensors, not .tolist() — PixArt transformer needs tensor timesteps
        extended = torch.cat([timesteps, torch.tensor([0], device=DEVICE, dtype=timesteps.dtype)])

        extractor.features.clear()
        with torch.no_grad():
            for i in range(len(extended) - 1, 0, -1):
                tc, tn = extended[i], extended[i-1]
                npred = pipe.transformer(
                    hidden_states=z,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attn,
                    timestep=tc.reshape(1),
                    return_dict=False)[0]
                # PixArt learn_sigma=True outputs 8 channels; take first 4
                if npred.shape[1] >= 8:
                    npred = npred[:, :4]
                ac, an = scheduler.alphas_cumprod[int(tc.item())], scheduler.alphas_cumprod[int(tn.item())]
                c1 = (an / ac).sqrt()
                sc = (1 - ac).sqrt()
                sn = (1 - an).sqrt()
                z = c1 * z + (sn - c1 * sc) * npred
        inv_f = {k: v.clone() for k, v in extractor.features.items()}

        extractor.features.clear()
        with torch.no_grad():
            for t_ in timesteps:
                npred = pipe.transformer(
                    hidden_states=z,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attn,
                    timestep=t_.reshape(1),
                    return_dict=False)[0]
                if npred.shape[1] >= 8:
                    npred = npred[:, :4]
                z = scheduler.step(npred, t_, z).prev_sample
        recon_f = {k: v.clone() for k, v in extractor.features.items()}

        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift

    extractor.remove()
    del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


# ─── Architecture Registry ───────────────────────────────────────────

ARCHITECTURES = {
    "SD1.5":        measure_sd15,
    "SDXL":         measure_sdxl,
    "HunyuanDiT":   measure_hunyuandit,
    "FLUX":         measure_flux,
    "SD3.5":        measure_sd35,
    "PixArt-Sigma": measure_pixart,
}

# ─── Main ────────────────────────────────────────────────────────────

def aggregate_per_image(per_image, layer_order):
    """Compute mean profile from per-image data."""
    profile = []
    for ln in layer_order:
        vals = list(per_image.get(ln, {}).values())
        profile.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(profile)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None,
                    help="Single architecture to measure (default: all 6)")
    ap.add_argument("--images", type=int, default=104,
                    help="Number of images to process (default: 104)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip architectures already measured")
    ap.add_argument("--steps", type=int, default=N_STEPS,
                    help=f"Number of DDIM/Euler steps (default: {N_STEPS})")
    args = ap.parse_args()

    images = sorted(DATA_DIR.glob("coco_*.jpg"))[:args.images]
    print(f"Unified 100-Image Fingerprint Measurement")
    print(f"  Images: {len(images)} from {DATA_DIR}")
    print(f"  Steps: {args.steps}")
    print(f"  Architectures: {list(ARCHITECTURES.keys()) if not args.arch else [args.arch]}")

    arch_list = [args.arch] if args.arch else list(ARCHITECTURES.keys())
    results = {}

    for arch_name in arch_list:
        out_path = OUT_DIR / f"{arch_name}_drift.json"
        if args.resume and out_path.exists():
            print(f"\n[{arch_name}] Already measured, loading cached...")
            with open(out_path) as f:
                results[arch_name] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"[{arch_name}] Measuring...")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            per_image, layer_order = ARCHITECTURES[arch_name](images, steps=args.steps)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        # Aggregate
        profile = aggregate_per_image(per_image, layer_order)
        features = extract_v2_features(profile, layer_order)

        n_layers = len(layer_order)
        print(f"  Done in {elapsed:.0f}s. {n_layers} layers, "
              f"peak={features['peak_layer']} ({features['peak_position']:.3f}), "
              f"conc={features['concentration']:.3f}")

        # Save
        result = {
            "arch": arch_name,
            "n_images": len(images),
            "n_steps": args.steps,
            "n_layers": n_layers,
            "layer_order": layer_order,
            "features": features,
            "aggregated": {ln: {"mean": float(np.mean(list(vals.values()))),
                                "std": float(np.std(list(vals.values())))}
                          for ln, vals in per_image.items() if vals},
            "per_image": {ln: vals for ln, vals in per_image.items()},
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        results[arch_name] = result
        print(f"  Saved to {out_path}")

    # ─── Cross-Architecture v2 Matrix ───
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("Computing cross-architecture v2 matrix...")
        features_dict = {name: r["features"] for name, r in results.items()}
        names = sorted(features_dict.keys())
        matrix = {}
        print(f"{'Pair':>25s}  {'D_total':>10s}  {'D_pp':>8s}  {'D_mag':>8s}")
        print("-" * 60)
        for i, na in enumerate(names):
            for nb in names[i+1:]:
                dd = distance_v2(features_dict[na], features_dict[nb])
                key = f"{na}-{nb}"
                matrix[key] = dd
                print(f"{key:>25s}  {dd['D_total']:10.6f}  {dd['D_peak_pos']:8.6f}  {dd['D_mag']:8.6f}")

        # Noise floor: bootstrap on largest-sample architecture
        ref_arch = max(results, key=lambda n: results[n]["n_images"])
        ref_profile = aggregate_per_image(
            results[ref_arch]["per_image"],
            results[ref_arch]["layer_order"])
        n_boot = 100
        rng = np.random.RandomState(0)
        boot_d = []
        for _ in range(n_boot):
            # Resample images
            ref_per_img = results[ref_arch]["per_image"]
            all_imgs = sorted(set().union(*[v.keys() for v in ref_per_img.values()]))
            ia = rng.choice(all_imgs, len(all_imgs), replace=True)
            ib = rng.choice(all_imgs, len(all_imgs), replace=True)
            pa = np.array([np.mean([ref_per_img[ln].get(img, 0) for img in ia])
                          for ln in results[ref_arch]["layer_order"]])
            pb = np.array([np.mean([ref_per_img[ln].get(img, 0) for img in ib])
                          for ln in results[ref_arch]["layer_order"]])
            fa = extract_v2_features(pa)
            fb = extract_v2_features(pb)
            boot_d.append(distance_v2(fa, fb)["D_total"])
        noise = {"median": float(np.median(boot_d)), "p95": float(np.percentile(boot_d, 95))}

        summary = {
            "metric": "v2 continuous (D_pp + D_mag, no peak_count)",
            "protocol": {"n_images": args.images, "n_steps": args.steps, "architectures": len(results)},
            "features": {n: f["features"] for n, f in results.items()},
            "pairwise": matrix,
            "noise_floor": noise,
        }
        matrix_path = OUT_DIR / "cross_arch_v2_matrix.json"
        with open(matrix_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nNoise floor (B={n_boot}): median={noise['median']:.6f} p95={noise['p95']:.6f}")
        print(f"Matrix saved to {matrix_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
