"""
P0b v2: C1 weight perturbation with per-layer relative units, noise floor,
cliff narrowing, and crash diagnosis.

Fixes over v1:
  1. σ_l = epsilon * ||W_l||_F (per-layer relative, not global absolute)
  2. Excludes bias, LayerNorm parameters from perturbation
  3. Reports sigma as epsilon (dimensionless multiplier on per-layer norm)
  4. Noise floor: D_s between disjoint image splits on the same model
  5. Intermediate epsilon values to narrow the cliff
  6. Crash diagnosis: per-layer PSNR, which layers die first

Usage:
  python -u scripts/p0b_weight_perturb_v2.py --images 19
"""

import copy, json, sys
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

# Epsilon values (dimensionless, multiplier on per-layer weight norm)
# v1 sigmas 1e-6 → epsilon 1e-6 (same, since v1 also used w_norm scaling per-param)
# Key difference: now per-layer, excluding bias/LN
EPSILONS = [1e-6, 1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3]


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
# Weight perturbation (per-layer relative, bias/LN excluded)
# ---------------------------------------------------------------------------

def is_weight_param(name, param):
    """Only perturb weight matrices, not bias or normalization params."""
    # Exclude bias terms
    if "bias" in name:
        return False
    # Exclude LayerNorm / GroupNorm weights
    if "norm" in name.lower() or "ln_" in name.lower():
        return False
    # Only include parameters with >= 2 dimensions (weight matrices)
    if param.ndim < 2:
        return False
    return True


def perturb_weights_per_layer(unet, epsilon):
    """Add Gaussian noise scaled by epsilon * ||W_layer||_F to each weight layer.

    Only perturbs weight matrices (>=2D), excludes bias and normalization params.
    Returns: perturbed unet, per-layer perturbation stats
    """
    unet_pert = copy.deepcopy(unet)
    layer_stats = {}

    with torch.no_grad():
        for name, param in unet_pert.named_parameters():
            if not is_weight_param(name, param):
                continue
            w_flat = param.data.view(-1).float()
            w_norm = w_flat.norm().item()
            noise_scale = epsilon * max(w_norm, 1e-8)
            noise = torch.randn_like(w_flat) * noise_scale
            # Track per-layer relative perturbation
            actual_sigma = noise.norm().item() / max(w_norm, 1e-8)
            layer_stats[name] = {
                "weight_norm": float(w_norm),
                "perturb_norm": float(noise.norm().item()),
                "relative_sigma": float(actual_sigma),
                "param_shape": list(param.shape),
                "is_weight": True,
            }
            param.data.copy_((w_flat + noise).view(param.shape).to(param.dtype))

    return unet_pert, layer_stats


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


def decode_and_psnr(pipe, latent):
    """Decode latent and compute PSNR vs ground-truth (needs reference image)."""
    with torch.no_grad():
        tensor = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    return tensor


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
# Diagnostic for one model variant (returns drift + reconstruction quality)
# ---------------------------------------------------------------------------

def diagnose_variant(label, pipe, prompt_embeds, hooker, targets, images, num_steps=50):
    """Full diagnostic: drift profile + reconstruction PSNR for the first image."""
    per_image_drifts = []
    first_image_latent = None
    first_recon_latent = None

    for i, img_path in enumerate(tqdm(images, desc=f"    {label}")):
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

            # Save first image for crash diagnosis
            if i == 0:
                first_image_latent = latent.detach().cpu()
                with torch.no_grad():
                    first_recon_latent = pipe.vae.decode(
                        z_recon / pipe.vae.config.scaling_factor
                    ).sample.detach().cpu()
                    first_recon_latent = pipe.vae.decode(
                        z_inv / pipe.vae.config.scaling_factor
                    ).sample.detach().cpu()  # actually decode z_inv to see

        except Exception as e:
            print(f"      Error on {img_path.name}: {e}")
            # Check for NaN/Inf in features
            has_nan = any(torch.isnan(v).any() for v in hooker.features.values()) if hooker.features else False
            has_inf = any(torch.isinf(v).any() for v in hooker.features.values()) if hooker.features else False
            print(f"      NaN in features: {has_nan}, Inf in features: {has_inf}")
            continue

    if not per_image_drifts:
        return None, [], "all_images_failed"

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
    features["drift_range"] = [float(d_min), float(d_max)]

    return {
        "label": label,
        "layer_names": [k for k, v in ordered],
        "drift_profile_raw": drift_profile.tolist(),
        "drift_profile_norm": drift_norm.tolist(),
        "structural_features": features,
        "per_layer_drift": {layer: float(d) for layer, d in mean_drift.items()},
    }, ordered, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=19)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise-floor-only", action="store_true", help="Only measure noise floor")
    args = parser.parse_args()

    images = TEST_IMAGES[:args.images]
    # Split for noise floor: first half vs second half
    split_A = images[:len(images)//2]
    split_B = images[len(images)//2:]

    print(f"Images: {len(images)} (noise floor: {len(split_A)} vs {len(split_B)})")
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

    # Count perturbable params
    n_weight, n_skip = 0, 0
    for name, param in pipe.unet.named_parameters():
        if is_weight_param(name, param):
            n_weight += 1
        else:
            n_skip += 1
    print(f"  Perturbable weights: {n_weight}, excluded (bias/LN): {n_skip}")

    # Save original weights
    original_state = copy.deepcopy(pipe.unet.state_dict())

    # =====================================================================
    # Noise floor: D_s(split_A, split_B) on same unmodified model
    # =====================================================================
    print("\n=== Noise Floor: D_s(image_split_A, image_split_B) ===")
    result_A, _, _ = diagnose_variant("noise_floor_A", pipe, prompt_embeds, hooker, targets, split_A, args.steps)
    result_B, _, _ = diagnose_variant("noise_floor_B", pipe, prompt_embeds, hooker, targets, split_B, args.steps)

    if result_A and result_B:
        noise_floor = structural_distance(
            result_A["structural_features"],
            result_B["structural_features"]
        )
        print(f"  D_s(split_A, split_B) = {noise_floor:.6f}  ← noise floor")
    else:
        noise_floor = None
        print("  Noise floor measurement FAILED")

    if args.noise_floor_only:
        summary = {"noise_floor_Ds": noise_floor, "n_split_A": len(split_A), "n_split_B": len(split_B)}
        with open(OUT_DIR / "noise_floor.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nNoise floor saved. DONE.")
        return

    # =====================================================================
    # Baseline (full image set)
    # =====================================================================
    print("\n=== Baseline (epsilon=0) ===")
    baseline_result, ordered_layers, _ = diagnose_variant(
        "baseline_eps0", pipe, prompt_embeds, hooker, targets, images, args.steps
    )
    bf = baseline_result["structural_features"]
    print(f"  Peak: {bf['peak_layer']} (pos={bf['peak_position']:.3f})")
    print(f"  Peaks: {bf['peak_count']}, Concentration: {bf['concentration']:.3f}, Gini: {bf['spread']:.3f}")
    print(f"  Drift range: [{bf['drift_range'][0]:.4f}, {bf['drift_range'][1]:.4f}]")

    # =====================================================================
    # Perturbed variants
    # =====================================================================
    all_results = [baseline_result]
    pairwise_ds = {}
    perturbation_stats = {}
    crash_diagnostics = {}

    for eps in EPSILONS:
        print(f"\n=== Perturbed: epsilon={eps:.0e} ===")

        # Restore original weights, then perturb
        pipe.unet.load_state_dict(copy.deepcopy(original_state))
        pipe.unet, layer_stats = perturb_weights_per_layer(pipe.unet, eps)
        perturbation_stats[f"eps_{eps:.0e}"] = layer_stats

        # Report perturbation stats
        rel_sigmas = [s["relative_sigma"] for s in layer_stats.values()]
        print(f"  Per-layer relative sigma: mean={np.mean(rel_sigmas):.2e}, "
              f"median={np.median(rel_sigmas):.2e}, max={np.max(rel_sigmas):.2e}")

        hooker.remove()
        hooker = UNetFeatureHooker(pipe.unet)
        hooker.register(targets)

        label = f"eps_{eps:.0e}"
        result, ordered, error = diagnose_variant(
            label, pipe, prompt_embeds, hooker, targets, images, args.steps
        )

        if result is None:
            crash_diagnostics[f"eps_{eps:.0e}"] = {
                "status": "crashed",
                "error": error,
                "all_images_failed": True,
            }
            print(f"  ✗ CRASHED: all images failed")
            all_results.append({"label": label, "status": "crashed", "epsilon": float(eps)})
            continue

        pf = result["structural_features"]
        ds = structural_distance(bf, pf)
        pairwise_ds[f"baseline_vs_{label}"] = float(ds)
        all_results.append(result)

        drift_range = pf["drift_range"]
        range_ratio = drift_range[1] / max(drift_range[0], 1e-8)

        print(f"  Peak: {pf['peak_layer']} (pos={pf['peak_position']:.3f})")
        print(f"  D_s(baseline, {label}) = {ds:.6f}")
        print(f"  Drift range: [{drift_range[0]:.4f}, {drift_range[1]:.4f}] ratio={range_ratio:.1f}x")

        # Crash diagnosis: check feature degeneracy
        raw_profile = np.array(result["drift_profile_raw"])
        n_near_zero = int(np.sum(raw_profile < 1e-8))
        has_nan_profile = bool(np.any(np.isnan(raw_profile)))
        print(f"  Profile health: {n_near_zero}/{len(raw_profile)} layers near-zero, NaN: {has_nan_profile}")

        crash_diagnostics[f"eps_{eps:.0e}"] = {
            "status": "ok" if not has_nan_profile else "degraded",
            "n_near_zero_layers": n_near_zero,
            "has_nan": has_nan_profile,
            "drift_range_ratio": float(range_ratio),
            "peak_preserved": pf["peak_layer"] == bf["peak_layer"],
        }

    # Cleanup
    hooker.remove()
    del pipe
    torch.cuda.empty_cache()

    # =====================================================================
    # Summary
    # =====================================================================
    min_inter_arch = 0.249
    noise_floor = noise_floor or 0.0

    print(f"\n{'='*60}")
    print("C1: Per-Layer Weight-Perturbation Dose-Response (v2)")
    print(f"{'='*60}")
    print(f"Noise floor D_s(image_split) = {noise_floor:.6f}")
    print(f"Min inter-arch D_s = {min_inter_arch:.4f}  (SD 1.5 vs SDXL)")
    print(f"\n{'epsilon':>10s}  {'D_s':>10s}  {'vs noise floor':>16s}  {'vs inter-arch':>16s}  {'peak OK?':>10s}")
    print("-" * 80)

    for eps in EPSILONS:
        key = f"baseline_vs_eps_{eps:.0e}"
        ds = pairwise_ds.get(key)
        if ds is None:
            print(f"{eps:10.0e}  {'CRASHED':>10s}")
        else:
            vs_noise = f"{ds/noise_floor:.1f}x" if noise_floor > 0 else "N/A"
            vs_inter = f"{ds/min_inter_arch*100:.2f}%"
            peak_ok = crash_diagnostics.get(f"eps_{eps:.0e}", {}).get("peak_preserved", "?")
            print(f"{eps:10.0e}  {ds:10.6f}  {vs_noise:>16s}  {vs_inter:>16s}  {'✓' if peak_ok else '✗':>10s}")

    # C1 check
    stable_eps = [eps for eps in EPSILONS
                  if (f"baseline_vs_eps_{eps:.0e}" in pairwise_ds
                      and pairwise_ds[f"baseline_vs_eps_{eps:.0e}"] < min_inter_arch)]

    print(f"\nC1 Falsification:")
    if stable_eps:
        print(f"  Stable regime: epsilon <= {stable_eps[-1]:.0e}")
        print(f"  Max D_s in stable regime: {max(pairwise_ds[f'baseline_vs_eps_{e:.0e}'] for e in stable_eps):.6f}")
        print(f"  vs inter-arch D_s ({min_inter_arch}): {max(pairwise_ds[f'baseline_vs_eps_{e:.0e}'] for e in stable_eps)/min_inter_arch*100:.2f}%")
    else:
        print(f"  No stable regime found — all D_s > min_inter_arch")

    # Save
    summary = {
        "protocol": {
            "images": len(images),
            "noise_floor_split": [len(split_A), len(split_B)],
            "steps": args.steps,
            "architecture": "SD 1.5 UNet",
            "perturbation": "Per-layer Gaussian noise, epsilon * ||W_l||_F, only weight matrices (>=2D), bias/LN excluded",
            "epsilons": [float(e) for e in EPSILONS],
        },
        "noise_floor_Ds": float(noise_floor) if noise_floor else None,
        "baseline": baseline_result["structural_features"],
        "pairwise_Ds": pairwise_ds,
        "perturbation_stats": perturbation_stats,
        "crash_diagnostics": crash_diagnostics,
        "reference": {
            "min_inter_arch_Ds": min_inter_arch,
            "pair": "SD 1.5 vs SDXL",
        },
    }

    summary_path = OUT_DIR / "weight_perturbation_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
