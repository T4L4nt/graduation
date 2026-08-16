#!/usr/bin/env python
"""
VAE-swap 2×4 factor experiment.
===============================

Tests whether Architecture Fingerprint peak position is determined by VAE
latent space or backbone architecture. Four compatible 4-channel KL-f8 models
are measured with both SD1.5-VAE and SDXL-VAE.

Design (8 configurations, 4 new + 4 baselines already measured):
           SD1.5-VAE (0.18215)     SDXL-VAE (0.13025)
  SD1.5    [baseline: 100img]      NEW
  SDXL     NEW                     [baseline: 100img]
  HunyuanDiT NEW                   [baseline: 100img]
  PixArt-Σ  NEW                    [baseline: 100img]

Critical: VAE latent scaling must match the VAE, not the backbone.
  SD1.5-VAE:  latent = encode(img) * 0.18215
  SDXL-VAE:   latent = encode(img) * 0.13025

Usage:
    python scripts/vae_swap_2x4.py                              # all 4 new configs
    python scripts/vae_swap_2x4.py --config SD1.5_sdxlvae       # single config
    python scripts/vae_swap_2x4.py --images 20                  # quick test

Output: outputs/vae_swap/{config}_drift.json
"""

import argparse, json, copy, sys, time
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
OUT_DIR = PROJECT_ROOT / "outputs" / "vae_swap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
N_STEPS = 50

# ─── v2 features (from unified_100img_measure.py) ───

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x); s = np.sum(x)
    return float((2 * np.sum(np.arange(1, n+1) * x) - (n+1) * s) / (n * s)) if s > 0 else 0.0

def extract_v2_features(profile, layer_names=None):
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

# ─── UNet layer ordering ───

def _unet_sort_key(name):
    parts = name.split(".")
    section_order = {"down_blocks": 0, "mid_block": 1, "up_blocks": 2}
    sec = section_order.get(parts[0], 99)
    blk_idx = 0; type_ord = 0; sub_idx = 0
    for i, p in enumerate(parts):
        if p in ("down_blocks", "mid_block", "up_blocks"):
            if i + 1 < len(parts) and parts[i+1].isdigit():
                blk_idx = int(parts[i+1])
        elif p == "resnets":
            type_ord = 0
            if i + 1 < len(parts) and parts[i+1].isdigit():
                sub_idx = int(parts[i+1])
        elif p == "attentions":
            type_ord = 1
            if i + 1 < len(parts) and parts[i+1].isdigit():
                sub_idx = int(parts[i+1])
    return (sec, blk_idx, type_ord, sub_idx)

def discover_unet_targets(unet):
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
    return sorted(targets, key=_unet_sort_key)

# ─── Hook infrastructure ───

class FeatureExtractor:
    def __init__(self, model, targets):
        self.model = model; self.targets = set(targets)
        self.features = {}; self.handles = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
        return fn

    def register(self):
        self.remove()
        for n, m in self.model.named_modules():
            if n in self.targets:
                self.handles.append(m.register_forward_hook(self._hook_fn(n)))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear(); self.features.clear()

# ─── DDIM inversion / reconstruction ───

def ddim_invert(pipe, latent, conditioning, steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = latent.clone()
    extended = timesteps.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended) - 1, 0, -1):
            tc, tn = extended[i], extended[i-1]
            noise_pred = pipe.unet(z, tc, **conditioning).sample
            ac, an = scheduler.alphas_cumprod[tc], scheduler.alphas_cumprod[tn]
            c1 = (an / ac).sqrt(); sc = (1 - ac).sqrt(); sn = (1 - an).sqrt()
            z = c1 * z + (sn - c1 * sc) * noise_pred
    return z

def ddim_reconstruct(pipe, noise, conditioning, steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=DEVICE)
    z = noise.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = pipe.unet(z, t, **conditioning).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z

# ─── VAE swap helper ───

def load_vae(which="sd15"):
    """Load standalone VAE for swapping."""
    from diffusers import AutoencoderKL
    if which == "sd15":
        return AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",
                                              subfolder="vae", torch_dtype=torch.float32).to(DEVICE)
    elif which == "sdxl":
        return AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                              subfolder="vae", torch_dtype=torch.float32).to(DEVICE)

VAE_SCALES = {"sd15": 0.18215, "sdxl": 0.13025}

# ─── Per-configuration measurement ───

def measure_sd15_with_vae(vae_name, images, steps=N_STEPS):
    """SD 1.5 backbone with swapped VAE."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"  Loading {model_id} with {vae_name} VAE...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = load_vae(vae_name)  # SWAP VAE
    vae_scale = VAE_SCALES[vae_name]
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        pe = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]
    conditioning = {"encoder_hidden_states": pe}

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc=f"  SD1.5+{vae_name}VAE"):
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE).float()
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample() * vae_scale
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
    extractor.remove(); del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


def measure_sdxl_with_vae(vae_name, images, steps=N_STEPS):
    """SDXL backbone with swapped VAE."""
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"  Loading {model_id} with {vae_name} VAE...")
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = load_vae(vae_name)
    vae_scale = VAE_SCALES[vae_name]
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    (prompt_embeds, _, pooled, _) = pipe.encode_prompt(
        prompt="", prompt_2="", device=DEVICE, num_images_per_prompt=1,
        do_classifier_free_guidance=False)
    add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=DEVICE, dtype=torch.float16)
    added = {"text_embeds": pooled, "time_ids": add_time_ids}
    conditioning = {"encoder_hidden_states": prompt_embeds, "added_cond_kwargs": added}

    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc=f"  SDXL+{vae_name}VAE"):
        img = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE).float()
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample() * vae_scale
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
    extractor.remove(); del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


def measure_dit_with_vae(arch, model_id, vae_name, images, steps=N_STEPS):
    """HunyuanDiT or PixArt-Σ backbone with swapped VAE."""
    vae_scale = VAE_SCALES[vae_name]
    img_size = 1024

    if arch == "HunyuanDiT":
        from diffusers import HunyuanDiTPipeline, DDIMScheduler
        print(f"  Loading {model_id} with {vae_name} VAE...")
        pipe = HunyuanDiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, prediction_type="v_prediction")
        pipe.vae = load_vae(vae_name)
        pipe.vae.to(torch.float32)
        targets = sorted([n for n, m in pipe.transformer.named_modules()
                         if n.startswith("blocks.") and n.count(".") == 1],
                        key=lambda x: int(x.replace("blocks.", "")))
        extractor = FeatureExtractor(pipe.transformer, targets)
        extractor.register()
        prompt_embeds, _, prompt_attn, _ = pipe.encode_prompt(
            prompt="", device=DEVICE, dtype=torch.float16,
            num_images_per_prompt=1, do_classifier_free_guidance=False)
        prompt_embeds_2, _, prompt_attn_2, _ = pipe.encode_prompt(
            prompt="", device=DEVICE, dtype=torch.float16,
            num_images_per_prompt=1, do_classifier_free_guidance=False, text_encoder_index=1)

        def _forward(z, t_int):
            t_tensor = torch.tensor([t_int], device=DEVICE, dtype=torch.float16)
            out = pipe.transformer(
                hidden_states=z, encoder_hidden_states=prompt_embeds,
                text_embedding_mask=prompt_attn, encoder_hidden_states_t5=prompt_embeds_2,
                text_embedding_mask_t5=prompt_attn_2, timestep=t_tensor, return_dict=False)[0]
            return out[:, :4] if out.shape[1] >= 8 else out

        use_scheduler_step = True  # v_prediction DDIM step

    else:  # PixArt-Sigma
        from diffusers import PixArtSigmaPipeline, DDIMScheduler
        print(f"  Loading {model_id} with {vae_name} VAE...")
        pipe = PixArtSigmaPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.vae = load_vae(vae_name)
        targets = sorted([n for n, m in pipe.transformer.named_modules()
                         if n.startswith("transformer_blocks.") and n.count(".") == 1],
                        key=lambda x: int(x.replace("transformer_blocks.", "")))
        extractor = FeatureExtractor(pipe.transformer, targets)
        extractor.register()
        (prompt_embeds, prompt_attn, _, _) = pipe.encode_prompt(
            prompt="", do_classifier_free_guidance=False, num_images_per_prompt=1, device=DEVICE)
        use_scheduler_step = False

    per_image = defaultdict(dict)
    desc = f"  {arch}+{vae_name}VAE"
    for img_path in tqdm(images, desc=desc):
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE).float()
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample() * vae_scale
        latent = latent.to(torch.float16)

        scheduler = pipe.scheduler
        scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = scheduler.timesteps
        z = latent.clone()
        extended = timesteps.tolist() + [0]

        # Inversion
        extractor.features.clear()
        with torch.no_grad():
            for i in range(len(extended) - 1, 0, -1):
                tc, tn = extended[i], extended[i-1]
                if arch == "HunyuanDiT":
                    v_pred = _forward(z, tc)
                    alpha_c = scheduler.alphas_cumprod[tc]
                    alpha_n = scheduler.alphas_cumprod[tn]
                    sigma_c = (1 - alpha_c).sqrt()
                    sigma_n = (1 - alpha_n).sqrt()
                    x0 = alpha_c.sqrt() * z - sigma_c * v_pred
                    eps = sigma_c * z + alpha_c.sqrt() * v_pred
                    z = alpha_n.sqrt() * x0 + sigma_n * eps
                else:  # PixArt: epsilon-prediction DDIM
                    tc_t = torch.tensor([tc], device=DEVICE, dtype=timesteps.dtype)
                    npred = pipe.transformer(
                        hidden_states=z, encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attn, timestep=tc_t, return_dict=False)[0]
                    if npred.shape[1] >= 8:
                        npred = npred[:, :4]
                    ac, an = scheduler.alphas_cumprod[tc], scheduler.alphas_cumprod[tn]
                    c1 = (an / ac).sqrt()
                    sc = (1 - ac).sqrt()
                    sn = (1 - an).sqrt()
                    z = c1 * z + (sn - c1 * sc) * npred
        inv_f = {k: v.clone() for k, v in extractor.features.items()}

        # Reconstruction
        extractor.features.clear()
        with torch.no_grad():
            for t_ in timesteps:
                if arch == "HunyuanDiT":
                    noise_pred = _forward(z, t_.item())
                    z = scheduler.step(noise_pred, t_, z).prev_sample
                else:
                    tn_t = torch.tensor([t_.item()], device=DEVICE, dtype=timesteps.dtype)
                    npred = pipe.transformer(
                        hidden_states=z, encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attn, timestep=tn_t, return_dict=False)[0]
                    if npred.shape[1] >= 8:
                        npred = npred[:, :4]
                    z = scheduler.step(npred, t_, z).prev_sample
        recon_f = {k: v.clone() for k, v in extractor.features.items()}

        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln] - recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift

    extractor.remove(); del pipe; torch.cuda.empty_cache()
    return dict(per_image), targets


# ─── Config registry ───

CONFIGS = {
    "SD1.5_sd15vae":   {"fn": measure_sd15_with_vae, "vae": "sd15",   "note": "baseline (already measured)"},
    "SD1.5_sdxlvae":   {"fn": measure_sd15_with_vae, "vae": "sdxl",   "note": "NEW"},
    "SDXL_sd15vae":    {"fn": measure_sdxl_with_vae,  "vae": "sd15",   "note": "NEW"},
    "SDXL_sdxlvae":    {"fn": measure_sdxl_with_vae,  "vae": "sdxl",   "note": "baseline (already measured)"},
    "HunyuanDiT_sd15vae": {"fn": None, "vae": "sd15", "note": "NEW — special measure_dit_with_vae"},
    "HunyuanDiT_sdxlvae": {"fn": None, "vae": "sdxl", "note": "baseline (already measured)"},
    "PixArt-Sigma_sd15vae": {"fn": None, "vae": "sd15", "note": "NEW — special measure_dit_with_vae"},
    "PixArt-Sigma_sdxlvae": {"fn": None, "vae": "sdxl", "note": "baseline (already measured)"},
}

# ─── Main ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--images", type=int, default=104)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    images = sorted(DATA_DIR.glob("coco_*.jpg"))[:args.images]
    print(f"VAE-swap 2×4 Factor Experiment")
    print(f"  Images: {len(images)}, Steps: {args.steps}")

    # Only run the NEW (non-baseline) configs
    to_run = [
        ("SD1.5", "sdxl", measure_sd15_with_vae, "SD1.5 backbone + SDXL-VAE"),
        ("SDXL", "sd15", measure_sdxl_with_vae, "SDXL backbone + SD1.5-VAE"),
        ("HunyuanDiT", "sd15", None, "HunyuanDiT backbone + SD1.5-VAE"),
        ("PixArt-Sigma", "sd15", None, "PixArt-Σ backbone + SD1.5-VAE"),
    ]
    dit_model_ids = {
        "HunyuanDiT": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
        "PixArt-Sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    }

    if args.config:
        to_run = [c for c in to_run if c[0] in args.config]

    results = {}
    for arch, vae, fn, desc in to_run:
        config_name = f"{arch}_{vae}vae"
        out_path = OUT_DIR / f"{config_name}_drift.json"
        if out_path.exists():
            print(f"\n[{config_name}] Already exists, skip")
            with open(out_path) as f:
                results[config_name] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"[{config_name}] {desc}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            if arch in ("HunyuanDiT", "PixArt-Sigma"):
                per_image, layer_order = measure_dit_with_vae(
                    arch, dit_model_ids[arch], vae, images, steps=args.steps)
            else:
                per_image, layer_order = fn(vae, images, steps=args.steps)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        profile = np.array([np.mean(list(per_image[ln].values())) for ln in layer_order])
        features = extract_v2_features(profile, layer_order)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s. "
              f"peak={features['peak_layer']} ({features['peak_position']:.3f}), "
              f"conc={features['concentration']:.3f}")

        result = {"arch": arch, "vae": vae, "config": config_name,
                   "n_images": len(images), "n_steps": args.steps,
                   "n_layers": len(layer_order), "layer_order": layer_order,
                   "features": features,
                   "aggregated": {ln: {"mean": float(np.mean(list(vals.values()))),
                                        "std": float(np.std(list(vals.values())))}
                                  for ln, vals in per_image.items() if vals},
                   "per_image": {ln: vals for ln, vals in per_image.items()}}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        results[config_name] = result

    # Summary table
    print(f"\n{'='*70}")
    print("VAE-Swap Results Summary")
    print(f"{'='*70}")
    print(f"{'Backbone':>14s} {'VAE':>8s} {'Peak Pos':>10s} {'Conc':>8s} {'Peak Layer':<35s}")
    print("-"*70)
    for config_name, r in sorted(results.items()):
        f = r["features"]
        print(f"{r['arch']:>14s} {r['vae']:>8s} {f['peak_position']:10.4f} {f['concentration']:8.4f} {f['peak_layer']:<35s}")

    # Compare with baselines from unified_100img
    print(f"\n--- Baseline comparison (from unified_100img) ---")
    baseline_dir = PROJECT_ROOT / "outputs" / "unified_100img"
    for arch in ["SD1.5", "SDXL", "HunyuanDiT", "PixArt-Sigma"]:
        bp = baseline_dir / f"{arch}_drift.json"
        if bp.exists():
            with open(bp) as f:
                bd = json.load(f)
            bf = bd["features"]
            default_vae = "sd15" if arch == "SD1.5" else "sdxl"
            print(f"  {arch} (default {default_vae}vae): pp={bf['peak_position']:.4f} peak={bf['peak_layer']}")
            # Show swap result if available
            swap_vae = "sdxl" if default_vae == "sd15" else "sd15"
            swap_key = f"{arch}_{swap_vae}vae"
            if swap_key in results:
                sf = results[swap_key]["features"]
                print(f"  {arch} (swap {swap_vae}vae):   pp={sf['peak_position']:.4f} peak={sf['peak_layer']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
