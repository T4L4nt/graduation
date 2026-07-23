"""
P0b: Sampler swap experiment.

Tests C3: Φ(M) organizational structure is invariant to sampler choice
(DDIM vs DPM++ vs Euler vs Euler a for SD 1.5; Euler vs Heun for FLUX).

Protocol:
  - SD 1.5: DDIM inversion (fixed) → reconstruction with target sampler
    Samplers: DDIM(η=0), DDIM(η=1), DPM++ 2M, Euler, Euler a
    Step counts: 20, 50
  - FLUX.1-dev: Euler inversion → reconstruction with target sampler
    Samplers: Euler, Heun
    Step counts: 20, 50
  - 19 coco_val images
  - Pre-registered criterion: peak position invariant across samplers;
    Spearman ρ(cross-sampler) > 0.95

Usage:
  LD_PRELOAD="..." python scripts/p0b_sampler_swap.py [--images N] [--skip-flux]
"""

import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, DDIMScheduler,
    DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
)
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_sampler_swap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COCO_VAL_DIR = PROJECT_ROOT / "data" / "coco_val"
TEST_IMAGES = sorted(COCO_VAL_DIR.glob("*.jpg"))

MODEL_ID_SD = "runwayml/stable-diffusion-v1-5"
MODEL_ID_FLUX = "black-forest-labs/FLUX.1-dev"

# Sampler configs for SD 1.5: (key, scheduler_cls, scheduler_kwargs)
SD_SAMPLERS = [
    ("ddim_eta0", DDIMScheduler, {"set_alpha_to_one": True}),
    ("ddim_eta1", DDIMScheduler, {"set_alpha_to_one": True}),
    ("dpmpp_2m", DPMSolverMultistepScheduler, {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    ("euler", EulerDiscreteScheduler, {}),
    ("euler_a", EulerAncestralDiscreteScheduler, {}),
]

FLUX_SAMPLERS = [
    ("euler", None, {}),      # uses flux default
    ("heun", None, {}),       # Heun via config override
]

STEP_COUNTS = [20, 50]


# ---------------------------------------------------------------------------
# UNet Hook infrastructure (same as p0b_cross_checkpoint.py)
# ---------------------------------------------------------------------------

def discover_hook_targets(unet):
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
    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.handles = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach().float().cpu()
        return fn

    def register(self, targets):
        self.remove()
        for name, module in self.unet.named_modules():
            if name in targets:
                self.handles.append(module.register_forward_hook(self._hook_fn(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.features.clear()


# ---------------------------------------------------------------------------
# Structural features (same as p0b_cross_checkpoint.py)
# ---------------------------------------------------------------------------

def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def extract_structural_features(drift_profile):
    """Extract 4-dim structural feature vector."""
    from scipy.signal import find_peaks

    profile = np.asarray(drift_profile, dtype=np.float64)
    L = len(profile)
    peak_pos = float(np.argmax(profile)) / L
    peaks, _ = find_peaks(profile, prominence=0.1)
    peak_cnt = int(np.sum(profile[peaks] > 0.5)) if len(peaks) > 0 else 0
    k = max(1, int(np.ceil(0.2 * L)))
    top_indices = np.argsort(profile)[-k:]
    concentration = float(np.sum(profile[top_indices]) / np.sum(profile))
    spread = float(gini(profile))
    return {
        "peak_position": peak_pos,
        "peak_count": peak_cnt,
        "concentration": concentration,
        "spread": spread,
    }


# ---------------------------------------------------------------------------
# SD 1.5: DDIM inversion (same for all samplers)
# ---------------------------------------------------------------------------

def encode_empty_prompt(pipe):
    text_input = pipe.tokenizer(
        "", padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        embeds = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]
    return embeds


def ddim_inversion(pipe, latents, prompt_embeds, num_steps):
    """Standard DDIM inversion (deterministic)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
    return z


def ddim_inversion_with_eta(pipe, latents, prompt_embeds, num_steps, eta=1.0):
    """DDIM inversion with stochastic sampling (η > 0)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            sigma_cur = eta * ((1 - alpha_cur) / alpha_cur).sqrt()
            sigma_next = eta * ((1 - alpha_next) / alpha_next).sqrt()
            coeff1 = (alpha_next / alpha_cur).sqrt() * (1 - sigma_cur / sigma_next) + \
                     (1 - alpha_cur) / (1 - alpha_next).sqrt() * sigma_cur / sigma_next
            coeff2 = alpha_cur.sqrt() * sigma_cur / sigma_next
            z = coeff1 * z + coeff2 * noise_pred
    return z


# ---------------------------------------------------------------------------
# Generic reconstruction (works with any diffusers scheduler)
# ---------------------------------------------------------------------------

def reconstruct(pipe, noise, prompt_embeds, num_steps):
    """Reconstruction using pipe.scheduler (set before calling)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            # Scale model input (required for Euler/DPM++ schedulers)
            if hasattr(scheduler, 'scale_model_input'):
                z_scaled = scheduler.scale_model_input(z, t)
            else:
                z_scaled = z
            noise_pred = pipe.unet(z_scaled, t, encoder_hidden_states=prompt_embeds).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_encode_sd(pipe, path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2 * tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent


# ---------------------------------------------------------------------------
# Main diagnostic for one (model, sampler, step_count) combo
# ---------------------------------------------------------------------------

def diagnose_sd_sampler(pipe, sampler_key, scheduler_cls, scheduler_kwargs,
                        prompt_embeds, hooker, targets, images, num_steps, eta=0.0):
    """Run diagnostic for one SD sampler configuration.

    Inversion: always DDIM (p0b_sampler_swap.py keeps a DDIM scheduler reference).
    Reconstruction: uses the target scheduler (scheduler_cls).
    """
    # Keep a DDIM scheduler for inversion
    ddim_sched = DDIMScheduler.from_config(pipe.scheduler.config)

    per_image_drifts = []

    for img_path in tqdm(images, desc=f"    {sampler_key}/{num_steps}st"):
        try:
            latent = load_and_encode_sd(pipe, str(img_path))

            # --- Inversion (always DDIM) ---
            pipe.scheduler = ddim_sched
            if eta > 0:
                z_inv = ddim_inversion_with_eta(pipe, latent, prompt_embeds, num_steps, eta=eta)
            else:
                z_inv = ddim_inversion(pipe, latent, prompt_embeds, num_steps)

            # Capture inversion features at turnaround
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_inv, ddim_sched.timesteps[0],
                          encoder_hidden_states=prompt_embeds).sample
            inv_features = {k: v.clone() for k, v in hooker.features.items()}

            # --- Reconstruction (target sampler) ---
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **scheduler_kwargs)
            z_recon = reconstruct(pipe, z_inv, prompt_embeds, num_steps)

            # Capture recon features
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_recon, pipe.scheduler.timesteps[0],
                          encoder_hidden_states=prompt_embeds).sample
            recon_features = {k: v.clone() for k, v in hooker.features.items()}

            layer_drifts = {}
            for layer_name in targets:
                if layer_name in inv_features and layer_name in recon_features:
                    f_inv = inv_features[layer_name]
                    f_recon = recon_features[layer_name]
                    drift = torch.norm(f_inv - f_recon, p=2).item()
                    layer_drifts[layer_name] = drift

            per_image_drifts.append(layer_drifts)
        except Exception as e:
            print(f"    Error on {img_path.name}: {e}")
            continue

    all_layers = list(per_image_drifts[0].keys())
    mean_drift = {}
    for layer in all_layers:
        vals = [d[layer] for d in per_image_drifts if layer in d]
        mean_drift[layer] = float(np.mean(vals)) if vals else 0.0

    ordered = sorted(mean_drift.items(), key=lambda x: targets.index(x[0]) if x[0] in targets else 999)
    drift_profile = np.array([v for _, v in ordered])

    d_min, d_max = drift_profile.min(), drift_profile.max()
    if d_max > d_min:
        drift_norm = (drift_profile - d_min) / (d_max - d_min)
    else:
        drift_norm = drift_profile.copy()

    features = extract_structural_features(drift_norm)
    features["peak_layer"] = ordered[int(np.argmax(drift_norm))][0]
    features["n_layers"] = len(drift_profile)
    features["n_images"] = len(per_image_drifts)

    return {
        "sampler": sampler_key,
        "num_steps": num_steps,
        "eta": eta,
        "layer_names": [k for k, v in ordered],
        "drift_profile_norm": drift_norm.tolist(),
        "structural_features": features,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=19)
    parser.add_argument("--skip-flux", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    images = TEST_IMAGES[:args.images]
    print(f"Images: {len(images)}, Output: {OUT_DIR}")

    all_results = []

    # =====================================================================
    # SD 1.5: Multi-sampler
    # =====================================================================
    print(f"\n{'='*60}")
    print("SD 1.5 Sampler Swap")
    print(f"{'='*60}")

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID_SD, torch_dtype=DTYPE).to(DEVICE)
    targets = discover_hook_targets(pipe.unet)
    hooker = UNetFeatureHooker(pipe.unet)
    hooker.register(targets)
    prompt_embeds = encode_empty_prompt(pipe)
    print(f"  Hook targets: {len(targets)} layers")

    for sampler_key, sched_cls, sched_kwargs in SD_SAMPLERS:
        if args.only and args.only != sampler_key:
            continue

        for n_steps in STEP_COUNTS:
            eta = 1.0 if "eta1" in sampler_key else 0.0
            print(f"\n  Sampler: {sampler_key}, Steps: {n_steps}, η={eta}")

            result = diagnose_sd_sampler(
                pipe, sampler_key, sched_cls, sched_kwargs,
                prompt_embeds, hooker, targets, images, n_steps, eta=eta
            )
            all_results.append(result)

    hooker.remove()
    del pipe
    torch.cuda.empty_cache()

    # =====================================================================
    # FLUX: Euler vs Heun (placeholder — requires flux_common.py)
    # =====================================================================
    if not args.skip_flux:
        print(f"\n{'='*60}")
        print("FLUX Sampler Swap — SKIPPED (requires flux_common.py integration)")
        print("  Protocol: Euler inversion → Euler/Heun reconstruction")
        print("  Step counts: 20, 50")
        print("  See scripts/p0b_sampler_swap_flux.py for full implementation")
        print(f"{'='*60}")

    # =====================================================================
    # Analysis: Cross-sampler comparison
    # =====================================================================
    print(f"\n{'='*60}")
    print("C3 Falsification Check")
    print(f"{'='*60}")

    # Group by (sampler, steps)
    by_sampler_steps = {}
    for r in all_results:
        key = f"{r['sampler']}_{r['num_steps']}st"
        by_sampler_steps[key] = r

    # Cross-sampler Spearman ρ (same step count)
    for n_steps in STEP_COUNTS:
        samplers_at_steps = [r for r in all_results if r["num_steps"] == n_steps]
        if len(samplers_at_steps) < 2:
            continue

        print(f"\n  {n_steps}-step cross-sampler comparison:")
        baseline = samplers_at_steps[0]  # DDIM η=0
        baseline_peak_layer = baseline["structural_features"]["peak_layer"]

        for other in samplers_at_steps[1:]:
            b = np.array(baseline["drift_profile_norm"])
            o = np.array(other["drift_profile_norm"])
            # Compute Spearman ρ on layer indices (not interpolated values)
            # Use per-layer ranking
            from scipy.stats import spearmanr
            rho, p = spearmanr(b, o)
            other_peak = other["structural_features"]["peak_layer"]
            peak_match = "✓" if other_peak == baseline_peak_layer else "✗"
            print(f"    {baseline['sampler']} vs {other['sampler']}: ρ={rho:.4f} (p={p:.4f}), "
                  f"peak: {baseline_peak_layer} vs {other_peak} {peak_match}")

    # Summary
    summary = {
        "protocol": {
            "images": len(images),
            "step_counts": STEP_COUNTS,
            "inversion_sd": "DDIM (deterministic for η=0, stochastic for η=1)",
            "norm": "min-max",
        },
        "pre_registered_criterion": {
            "peak_invariance": "Peak layer must be identical across all samplers",
            "spearman_rho": "ρ > 0.95 across samplers at same step count",
        },
        "results": all_results,
    }

    summary_path = OUT_DIR / "sampler_swap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
