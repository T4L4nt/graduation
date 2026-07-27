"""
P0b: Cross-checkpoint fingerprint stability.

Tests C1: Φ(M) is invariant to weight-level perturbations (different
checkpoints of the same architecture) but responds to architectural differences.

Protocol:
  - SD 1.4 vs SD 1.5 vs DreamShaper (community fine-tune, same UNet architecture)
  - SDXL base vs SDXL fine-tune (if available)
  - 19 coco_val images, 50-step DDIM, empty prompt
  - Per-layer drift profile → 4-feature structural vector → D_s
  - Pre-registered criterion: D_s(intra-checkpoint) << min D_s(inter-architecture)

Usage:
  LD_PRELOAD="..." python scripts/p0b_cross_checkpoint.py [--images N] [--steps N]
"""

import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

# Add scripts/ to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model registry ---
# (model_key, model_id, pipeline_cls)
SD_CHECKPOINTS = [
    ("sd14", "runwayml/stable-diffusion-v1-4", StableDiffusionPipeline),
    ("sd15", "runwayml/stable-diffusion-v1-5", StableDiffusionPipeline),
    # Community fine-tune: same UNet architecture, different weights
    ("dreamshaper8", "Lykon/dreamshaper-8", StableDiffusionPipeline),
]

SDXL_CHECKPOINTS = [
    ("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", StableDiffusionXLPipeline),
]

# --- Image set ---
COCO_VAL_DIR = PROJECT_ROOT / "data" / "coco_val"
TEST_IMAGES = sorted(COCO_VAL_DIR.glob("*.jpg"))


# ---------------------------------------------------------------------------
# UNet Hook discovery (reused from phase1_diagnostics.py)
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
# DDIM inversion / reconstruction (SD 1.x, empty prompt)
# ---------------------------------------------------------------------------

def encode_empty_prompt(pipe):
    """Encode empty prompt for unconditional inversion."""
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
# Structural feature extraction
# ---------------------------------------------------------------------------

def gini(x):
    """Gini coefficient of a 1D array."""
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def extract_structural_features(drift_profile):
    """Extract 4-dim structural feature vector from drift profile (raw layers, no interpolation).

    Args:
        drift_profile: np.array of per-layer drift values (L layers, already min-max normalized)
    Returns:
        dict with peak_position, peak_count, concentration, spread
    """
    from scipy.signal import find_peaks

    profile = np.asarray(drift_profile, dtype=np.float64)
    L = len(profile)

    # peak_position: relative position of the max-drift layer
    peak_pos = float(np.argmax(profile)) / L

    # peak_count: number of peaks with normalized drift > 0.5
    peaks, props = find_peaks(profile, prominence=0.1)
    peak_cnt = int(np.sum(profile[peaks] > 0.5)) if len(peaks) > 0 else 0

    # concentration: fraction of drift in top 20% of layers
    k = max(1, int(np.ceil(0.2 * L)))
    top_indices = np.argsort(profile)[-k:]
    concentration = float(np.sum(profile[top_indices]) / np.sum(profile))

    # spread: Gini coefficient
    spread = float(gini(profile))

    return {
        "peak_position": peak_pos,
        "peak_count": peak_cnt,
        "concentration": concentration,
        "spread": spread,
    }


def structural_distance(feat_a, feat_b):
    """Euclidean D_s between two 4-dim structural feature vectors."""
    keys = ["peak_position", "peak_count", "concentration", "spread"]
    va = np.array([feat_a[k] for k in keys])
    vb = np.array([feat_b[k] for k in keys])
    return float(np.linalg.norm(va - vb))


# ---------------------------------------------------------------------------
# Main diagnostic for one model
# ---------------------------------------------------------------------------

def diagnose_model(model_key, model_id, pipeline_cls, images, num_steps=50):
    """Run full inversion-reconstruction diagnostic on one model checkpoint.

    Returns:
        dict with per-image drift profiles, mean profile, structural features
    """
    print(f"\n{'='*60}")
    print(f"Diagnosing: {model_key} ({model_id})")
    print(f"{'='*60}")

    # Load pipeline
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Discover hook targets
    unet = pipe.unet
    targets = discover_hook_targets(unet)
    hooker = UNetFeatureHooker(unet)
    hooker.register(targets)
    print(f"  Hook targets: {len(targets)} layers")

    # Encode empty prompt once
    if pipeline_cls == StableDiffusionXLPipeline:
        # SDXL needs two text encoders
        prompt_embeds, pooled = pipe.encode_prompt("", DEVICE, 1, False)
        prompt_embeds = torch.cat([prompt_embeds, pooled], dim=-1)  # simplified; uses full embeds
    else:
        prompt_embeds = encode_empty_prompt(pipe)

    per_image_drifts = []

    for img_path in tqdm(images, desc=f"  {model_key}"):
        try:
            # Load + encode
            size = 1024 if pipeline_cls == StableDiffusionXLPipeline else 512
            latent = load_and_encode_sd(pipe, str(img_path), size=size)

            # Inversion
            z_inv = ddim_inversion(pipe, latent, prompt_embeds, num_steps)

            # Capture inversion features at turnaround
            with torch.no_grad():
                pipe.unet(z_inv, pipe.scheduler.timesteps[0],
                          encoder_hidden_states=prompt_embeds).sample
            inv_features = {k: v.clone() for k, v in hooker.features.items()}

            # Reconstruction
            z_recon = ddim_reconstruction(pipe, z_inv, prompt_embeds, num_steps)

            # Capture reconstruction features at same timestep
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_recon, pipe.scheduler.timesteps[0],
                          encoder_hidden_states=prompt_embeds).sample
            recon_features = {k: v.clone() for k, v in hooker.features.items()}

            # Per-layer drift at turnaround
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

    # Aggregate across images
    all_layers = list(per_image_drifts[0].keys())
    mean_drift = {}
    for layer in all_layers:
        vals = [d[layer] for d in per_image_drifts if layer in d]
        mean_drift[layer] = float(np.mean(vals)) if vals else 0.0

    # Ordered drift profile (by UNet structural order, not drift magnitude)
    ordered = sorted(mean_drift.items(), key=lambda x: targets.index(x[0]) if x[0] in targets else 999)
    drift_profile = np.array([v for _, v in ordered])

    # Min-max normalize
    d_min, d_max = drift_profile.min(), drift_profile.max()
    if d_max > d_min:
        drift_norm = (drift_profile - d_min) / (d_max - d_min)
    else:
        drift_norm = drift_profile.copy()

    # Extract structural features
    features = extract_structural_features(drift_norm)
    features["peak_layer"] = ordered[int(np.argmax(drift_norm))][0]
    features["n_layers"] = len(drift_profile)
    features["n_images"] = len(per_image_drifts)

    # Cleanup
    hooker.remove()
    del pipe
    torch.cuda.empty_cache()

    result = {
        "model_key": model_key,
        "model_id": model_id,
        "layer_names": [k for k, v in ordered],
        "drift_profile_raw": drift_profile.tolist(),
        "drift_profile_norm": drift_norm.tolist(),
        "structural_features": features,
        "per_image_drifts": {
            layer: [d.get(layer, None) for d in per_image_drifts]
            for layer in all_layers
        },
    }

    # Save per-model result
    out_path = OUT_DIR / f"{model_key}_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → Saved {out_path}")
    print(f"  Peak layer: {features['peak_layer']} (pos={features['peak_position']:.3f})")
    print(f"  Peaks: {features['peak_count']}, Concentration: {features['concentration']:.3f}, Gini: {features['spread']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=19, help="Number of coco_val images (1-19)")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--skip-sdxl", action="store_true", help="Skip SDXL (heavier)")
    parser.add_argument("--only", type=str, default=None, help="Only run specific model key")
    args = parser.parse_args()

    images = TEST_IMAGES[:args.images]
    print(f"Images: {len(images)}")
    print(f"Steps: {args.steps}")
    print(f"Output: {OUT_DIR}")

    results = {}

    # SD checkpoints
    for model_key, model_id, pipeline_cls in SD_CHECKPOINTS:
        if args.only and args.only != model_key:
            continue
        try:
            results[model_key] = diagnose_model(model_key, model_id, pipeline_cls,
                                                 images, args.steps)
        except Exception as e:
            print(f"  FAILED: {model_key} - {e}")
            import traceback
            traceback.print_exc()

    # SDXL checkpoints
    if not args.skip_sdxl:
        for model_key, model_id, pipeline_cls in SDXL_CHECKPOINTS:
            if args.only and args.only != model_key:
                continue
            try:
                results[model_key] = diagnose_model(model_key, model_id, pipeline_cls,
                                                     images, args.steps)
            except Exception as e:
                print(f"  FAILED: {model_key} - {e}")
                import traceback
                traceback.print_exc()

    # --- Cross-checkpoint comparison ---
    print(f"\n{'='*60}")
    print("Cross-Checkpoint Structural Distance (D_s)")
    print(f"{'='*60}")

    model_keys = list(results.keys())
    pairwise = {}

    for i, ka in enumerate(model_keys):
        for kb in model_keys[i+1:]:
            fa = results[ka]["structural_features"]
            fb = results[kb]["structural_features"]
            ds = structural_distance(fa, fb)
            pairwise[f"{ka}_vs_{kb}"] = float(ds)

            # Highlight intra-architecture pairs
            same_arch = ("sd" in ka and "sd" in kb) or \
                        ("sdxl" in ka and "sdxl" in kb)
            tag = "<<< SAME ARCHITECTURE" if same_arch else ""
            print(f"  {ka} vs {kb}: D_s = {ds:.4f}  {tag}")

    # Comparison with cross-architecture baseline
    # (from Property 2: min cross-arch D_s ≈ 0.249 SD1.5-SDXL)
    print(f"\n{'='*60}")
    print("C1 Falsification Check")
    print(f"{'='*60}")
    print("Pre-registered criterion: D_s(intra-checkpoint) << min D_s(inter-architecture)")
    print("Cross-architecture reference (SD 1.5 vs SDXL): D_s ≈ 0.249")

    for pair, ds in pairwise.items():
        is_intra = (pair.startswith("sd14_vs_sd15") or pair.startswith("sd15_vs_sd14") or
                    pair.startswith("sd14_vs_dream") or pair.startswith("sd15_vs_dream") or
                    pair.startswith("dreamshaper"))
        if is_intra:
            status = "PASS" if ds < 0.249 / 2 else "FAIL"
            print(f"  {pair}: D_s = {ds:.4f}  [{status}]  (threshold = {0.249/2:.4f})")

    # Save summary
    summary = {
        "protocol": {
            "images": len(images),
            "steps": args.steps,
            "inversion": "DDIM empty-prompt",
            "norm": "min-max",
            "distance": "D_s (Euclidean, 4 features)",
        },
        "pairwise_Ds": pairwise,
        "pre_registered_threshold": 0.249 / 2,
        "c1_falsification": "D_s(intra) must be substantially smaller than min D_s(inter) = 0.249",
    }

    summary_path = OUT_DIR / "cross_checkpoint_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return results


if __name__ == "__main__":
    main()
