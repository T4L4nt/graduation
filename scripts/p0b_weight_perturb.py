"""
P0b: Cross-checkpoint fingerprint stability — weight perturbation variant.

Tests C1: Φ(M) is stable under controlled weight-level perturbations
(no download required — perturbs SD 1.5 weights with Gaussian noise).

Protocol:
  - SD 1.5 baseline (fp16)
  - Perturbed copies: σ ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} × ||W||
  - 19 coco_val images, 50-step DDIM, empty prompt
  - Extract 4-feature structural vector → D_s(baseline, perturbed)
  - Dose-response curve: D_s vs perturbation magnitude
  - C1 criterion: max intra-arch D_s << min inter-arch D_s ≈ 0.249

Usage:
  python -u scripts/p0b_cross_checkpoint.py --mode perturb [--images N]
"""

import argparse, copy, json, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
COCO_VAL_DIR = PROJECT_ROOT / "data" / "coco_val"
TEST_IMAGES = sorted(COCO_VAL_DIR.glob("*.jpg"))

# Perturbation levels (relative to weight norm)
SIGMA_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


# ---------------------------------------------------------------------------
# UNet hook infrastructure
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
# Weight perturbation
# ---------------------------------------------------------------------------

def perturb_weights(unet, sigma):
    """Add Gaussian noise scaled by sigma * ||W|| to each parameter of unet.
    Operates in-place on a clone. Returns the perturbed unet.
    """
    unet_pert = copy.deepcopy(unet)
    with torch.no_grad():
        for name, param in unet_pert.named_parameters():
            if param.requires_grad or not param.requires_grad:  # all params
                w_norm = param.norm().item()
                noise_scale = sigma * max(w_norm, 1e-8)
                param.add_(torch.randn_like(param) * noise_scale)
    return unet_pert


# ---------------------------------------------------------------------------
# DDIM inversion / reconstruction
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


def ddim_reconstruction(pipe, noise, prompt_embeds, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


def load_and_encode(pipe, path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2 * tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def gini(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x)
    n = len(x)
    if np.sum(x) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def extract_structural_features(drift_profile):
    from scipy.signal import find_peaks
    profile = np.asarray(drift_profile, dtype=np.float64)
    L = len(profile)
    peak_pos = float(np.argmax(profile)) / L
    peaks, _ = find_peaks(profile, prominence=0.1)
    peak_cnt = int(np.sum(profile[peaks] > 0.5)) if len(peaks) > 0 else 0
    k = max(1, int(np.ceil(0.2 * L)))
    top_indices = np.argsort(profile)[-k:]
    concentration = float(np.sum(profile[top_indices]) / np.sum(profile)) if np.sum(profile) > 0 else 0.0
    spread = float(gini(profile))
    return {
        "peak_position": peak_pos,
        "peak_count": peak_cnt,
        "concentration": concentration,
        "spread": spread,
    }


def structural_distance(feat_a, feat_b):
    keys = ["peak_position", "peak_count", "concentration", "spread"]
    va = np.array([feat_a[k] for k in keys])
    vb = np.array([feat_b[k] for k in keys])
    return float(np.linalg.norm(va - vb))


# ---------------------------------------------------------------------------
# Diagnostic for one model variant
# ---------------------------------------------------------------------------

def diagnose_variant(label, pipe, prompt_embeds, hooker, targets, images, num_steps=50):
    """Run full diagnostic on one model variant. pipe.unet must be already set."""
    per_image_drifts = []
    for img_path in tqdm(images, desc=f"    {label}"):
        try:
            latent = load_and_encode(pipe, str(img_path))
            z_inv = ddim_inversion(pipe, latent, prompt_embeds, num_steps)
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=prompt_embeds).sample
            inv_features = {k: v.clone() for k, v in hooker.features.items()}
            z_recon = ddim_reconstruction(pipe, z_inv, prompt_embeds, num_steps)
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_recon, pipe.scheduler.timesteps[0], encoder_hidden_states=prompt_embeds).sample
            recon_features = {k: v.clone() for k, v in hooker.features.items()}
            layer_drifts = {}
            for layer_name in targets:
                if layer_name in inv_features and layer_name in recon_features:
                    drift = torch.norm(inv_features[layer_name] - recon_features[layer_name], p=2).item()
                    layer_drifts[layer_name] = drift
            per_image_drifts.append(layer_drifts)
        except Exception as e:
            print(f"      Error on {img_path.name}: {e}")
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
        "label": label,
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mode", default="perturb", choices=["perturb", "checkpoint"])
    args = parser.parse_args()

    images = TEST_IMAGES[:args.images]
    print(f"Images: {len(images)}, Steps: {args.steps}, Mode: {args.mode}")
    print(f"Output: {OUT_DIR}")

    # Load base model
    print("\nLoading SD 1.5 baseline...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_hook_targets(pipe.unet)
    hooker = UNetFeatureHooker(pipe.unet)
    hooker.register(targets)
    prompt_embeds = encode_empty_prompt(pipe)
    print(f"  Hook targets: {len(targets)} layers")

    # Save original weights
    original_state = copy.deepcopy(pipe.unet.state_dict())

    # --- Baseline ---
    print("\n=== Baseline (σ=0) ===")
    baseline_result = diagnose_variant("baseline_sigma0", pipe, prompt_embeds, hooker, targets, images, args.steps)
    bf = baseline_result["structural_features"]
    print(f"  Peak: {bf['peak_layer']} (pos={bf['peak_position']:.3f})")
    print(f"  Peaks: {bf['peak_count']}, Concentration: {bf['concentration']:.3f}, Gini: {bf['spread']:.3f}")

    # --- Perturbed variants ---
    results = [baseline_result]
    pairwise_ds = {}

    for sigma in SIGMA_VALUES:
        print(f"\n=== Perturbed: σ={sigma} ===")

        # Restore original weights, then perturb
        pipe.unet.load_state_dict(copy.deepcopy(original_state))
        pipe.unet = perturb_weights(pipe.unet, sigma)
        hooker.remove()
        hooker = UNetFeatureHooker(pipe.unet)
        hooker.register(targets)

        label = f"sigma_{sigma:.0e}"
        result = diagnose_variant(label, pipe, prompt_embeds, hooker, targets, images, args.steps)

        pf = result["structural_features"]
        ds = structural_distance(bf, pf)
        pairwise_ds[f"baseline_vs_{label}"] = float(ds)

        print(f"  Peak: {pf['peak_layer']} (pos={pf['peak_position']:.3f})")
        print(f"  D_s(baseline, {label}) = {ds:.6f}")
        results.append(result)

    # Cleanup
    hooker.remove()
    del pipe
    torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("C1 Weight-Perturbation Dose-Response")
    print(f"{'='*60}")
    print(f"{'σ':>10s}  {'D_s':>10s}  {'% of min inter-arch':>20s}")
    print("-" * 50)

    min_inter_arch = 0.249  # SD 1.5 vs SDXL
    for sigma in SIGMA_VALUES:
        key = f"baseline_vs_sigma_{sigma:.0e}"
        ds = pairwise_ds.get(key, float('nan'))
        pct = (ds / min_inter_arch * 100)
        print(f"{sigma:10.0e}  {ds:10.6f}  {pct:19.1f}%")

    print(f"\nC1 Falsification Check:")
    max_intra = max(pairwise_ds.values())
    print(f"  max D_s(intra-architecture) = {max_intra:.6f}")
    print(f"  min D_s(inter-architecture) = {min_inter_arch:.4f}  (SD 1.5 vs SDXL, reference)")

    if max_intra < min_inter_arch:
        ratio = max_intra / min_inter_arch
        print(f"  D_s(intra)/D_s(inter) ratio = {ratio:.4f}")
        print(f"  ✓ C1 PASS: intra-architecture distance < inter-architecture distance")
    else:
        print(f"  ✗ C1 FAIL at σ >= 1e-2")

    # Save summary
    summary = {
        "protocol": {
            "images": len(images),
            "steps": args.steps,
            "architecture": "SD 1.5 UNet",
            "perturbation": "Gaussian noise scaled by σ·||W||",
            "sigmas": [float(s) for s in SIGMA_VALUES],
            "min_inter_arch_Ds": min_inter_arch,
            "reference_pair": "SD 1.5 vs SDXL",
        },
        "baseline": baseline_result["structural_features"],
        "pairwise_Ds": pairwise_ds,
        "dose_response": [
            {"sigma": float(s), "Ds": float(pairwise_ds[f"baseline_vs_sigma_{s:.0e}"])}
            for s in SIGMA_VALUES
        ],
    }

    summary_path = OUT_DIR / "weight_perturbation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
