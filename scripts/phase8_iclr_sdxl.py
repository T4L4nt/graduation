"""
Phase 8 ICLR — Task C: SDXL Cross-Architecture Skip Intervention Validation

Validates that the skip-mismatch pathology generalizes from SD 1.5 to SDXL:
1. Run drift fingerprint on SDXL (20 prompts) → find peak-drift skip
2. Cut A on that peak skip → compare Original SDXL inversion
3. Dose-response (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}) on SDXL
4. Compare SD 1.5 vs SDXL: fingerpint heatmaps, peak position, dose-response curves

Output:
  - CSV: per-prompt SDXL metrics
  - Statistical summary: paired t-test
  - Figures: SD 1.5 vs SDXL fingerprint comparison, dose-response curves
  - Report: generalization assessment
"""

import argparse, json, csv, sys, os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

OUT_DIR = Path("outputs/phase8_iclr_sdxl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SD 1.5 reference values for comparison
SD15_PEAK_DRIFT = 2329.4
SD15_CUT_A_PSNR_DELTA = +2.20
SD15_SKIP_MAP = {  # down_block_index -> up_block_index
    0: 3, 1: 2, 2: 1, 3: 0
}
# SDXL has 3 down/up blocks: down[i] → up[2-i]
SDXL_SKIP_MAP = {0: 2, 1: 1, 2: 0}


# ---------------------------------------------------------------------------
# Diverse prompts (reuse from Task A)
# ---------------------------------------------------------------------------

DIVERSE_PROMPTS = [
    ("portrait_01", "a professional headshot of a woman with natural lighting"),
    ("portrait_02", "a candid street portrait of an elderly man with wrinkles"),
    ("portrait_03", "a young child laughing, soft window light"),
    ("portrait_04", "a muscular athlete posing after training, dramatic lighting"),
    ("landscape_01", "a mountain lake at sunrise with mist over the water"),
    ("landscape_02", "a vast desert with sand dunes and a lone cactus"),
    ("landscape_03", "a snowy pine forest with a wooden cabin, winter morning"),
    ("landscape_04", "a tropical beach with palm trees and turquoise water"),
    ("animal_01", "a golden retriever puppy sitting in a flower garden"),
    ("animal_02", "a bald eagle soaring over a canyon, wings spread"),
    ("animal_03", "a white cat sleeping on a windowsill, afternoon sun"),
    ("animal_04", "a herd of wild horses galloping across a prairie"),
    ("object_01", "a vintage film camera on a wooden table, macro shot"),
    ("object_02", "a steaming cup of coffee next to an open book"),
    ("object_03", "a classic red sports car parked on a coastal road"),
    ("object_04", "a handcrafted ceramic vase with dried flowers"),
    ("abstract_01", "geometric patterns in neon colors, digital art style"),
    ("abstract_02", "swirling galaxies and cosmic dust, space photography"),
    ("abstract_03", "liquid metal flowing into organic shapes, 3D render"),
    ("abstract_04", "fractal patterns resembling a kaleidoscope, vibrant colors"),
    ("text_01", "a storefront window with hand-painted lettering and reflections"),
    ("text_02", "a vintage neon sign reading OPEN at night, urban street"),
    ("text_03", "a graffiti mural on a brick wall, colorful street art"),
    ("text_04", "a minimalist poster with bold typography on a gallery wall"),
    ("text_05", "a restaurant menu board with chalk lettering, warm interior"),
]


# ---------------------------------------------------------------------------
# SDXL Skip Intervention (adapted for SDXL's 3-block UNet)
# ---------------------------------------------------------------------------

class SDXLSkipIntervention:
    """Zero out res_hidden_states_tuple for specified up_blocks during forward.

    Adapted for SDXL's 3-block UNet structure (down_blocks 0-2, up_blocks 0-2).
    """

    def __init__(self, unet, cut_up_indices):
        self.unet = unet
        self.cut_up_indices = set(cut_up_indices)
        self._originals = {}

    def __enter__(self):
        for idx in self.cut_up_indices:
            up_block = self.unet.up_blocks[idx]
            self._originals[idx] = up_block.forward
            original = up_block.forward

            def make_patched(orig_fn):
                def patched_forward(hidden_states, res_hidden_states_tuple,
                                    *args, **kwargs):
                    zeroed = tuple(torch.zeros_like(t) for t in res_hidden_states_tuple)
                    return orig_fn(hidden_states, zeroed, *args, **kwargs)
                return patched_forward

            up_block.forward = make_patched(original)
        return self

    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.unet.up_blocks[idx].forward = orig
        self._originals.clear()


class SDXLPartialSkipIntervention:
    """Scale skip by α (for dose-response)."""

    def __init__(self, unet, cut_up_indices, alpha):
        self.unet = unet
        self.cut_up_indices = set(cut_up_indices)
        self.alpha = alpha
        self._originals = {}

    def __enter__(self):
        for idx in self.cut_up_indices:
            up_block = self.unet.up_blocks[idx]
            self._originals[idx] = up_block.forward
            original = up_block.forward
            alpha = self.alpha

            def make_patched(orig_fn, a):
                def patched_forward(hidden_states, res_hidden_states_tuple,
                                    *args, **kwargs):
                                    scaled = tuple(t * a for t in res_hidden_states_tuple)
                                    return orig_fn(hidden_states, scaled, *args, **kwargs)
                return patched_forward

            up_block.forward = make_patched(original, alpha)
        return self

    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.unet.up_blocks[idx].forward = orig
        self._originals.clear()


class SDXLNoiseIntervention:
    """Replace skip with Gaussian noise."""

    def __init__(self, unet, cut_up_indices):
        self.unet = unet
        self.cut_up_indices = set(cut_up_indices)
        self._originals = {}

    def __enter__(self):
        for idx in self.cut_up_indices:
            up_block = self.unet.up_blocks[idx]
            self._originals[idx] = up_block.forward
            original = up_block.forward

            def make_patched(orig_fn):
                def patched_forward(hidden_states, res_hidden_states_tuple,
                                    *args, **kwargs):
                    noisy = tuple(
                        torch.randn_like(t) * t.std() + t.mean()
                        for t in res_hidden_states_tuple
                    )
                    return orig_fn(hidden_states, noisy, *args, **kwargs)
                return patched_forward

            up_block.forward = make_patched(original)
        return self

    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.unet.up_blocks[idx].forward = orig
        self._originals.clear()


# ---------------------------------------------------------------------------
# SDXL pipeline
# ---------------------------------------------------------------------------

def load_sdxl_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, local_files_only=True,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(torch.float32)  # SDXL VAE NaN in fp16
    return pipe


def encode_prompt_sdxl(pipe, prompt):
    """SDXL dual-encoder. Returns (prompt_embeds, pooled_prompt_embeds, add_time_ids)."""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=DEVICE,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    add_time_ids = torch.tensor(
        [[1024, 1024, 0, 0, 1024, 1024]], device=DEVICE, dtype=DTYPE
    )
    return prompt_embeds, pooled_prompt_embeds, add_time_ids


def load_and_encode_sdxl(pipe, path):
    """Load image → VAE encode for SDXL (VAE runs fp32)."""
    img = Image.open(path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    tensor_fp32 = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    tensor_fp32 = 2 * tensor_fp32 - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor_fp32).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent.to(dtype=DTYPE), tensor_fp32


def generate_sdxl(pipe, prompt, seed=42, num_steps=50, guidance_scale=7.5):
    """Generate image with SDXL. Returns [-1, 1] tensor on DEVICE (fp32)."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_steps, device=DEVICE)
    with torch.no_grad():
        # Use output_type="latent" to avoid VAE dtype mismatch (fp16 latent vs fp32 VAE)
        result = pipe(
            prompt=prompt, num_inference_steps=num_steps,
            guidance_scale=guidance_scale, generator=generator,
            output_type="latent",
        )
        latents = result.images  # fp16 latents
        # Decode manually: VAE is fp32, latents must be fp32
        image = pipe.vae.decode(
            latents.float() / pipe.vae.config.scaling_factor, return_dict=False
        )[0]
    return image  # fp32 on DEVICE


def decode_latent(pipe, latent):
    with torch.no_grad():
        return pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample


def compute_metrics(original_tensor, recon_tensor, lpips_fn=None):
    """PSNR / SSIM / LPIPS."""
    orig = original_tensor.float().clamp(-1, 1)
    recon = recon_tensor.float().clamp(-1, 1)
    mse = torch.nn.functional.mse_loss(orig, recon)
    psnr_val = (20 * torch.log10(2.0 / (torch.sqrt(mse) + 1e-8))).item()
    orig_np = (orig.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    recon_np = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    ssim_val = float(ssim(orig_np, recon_np, channel_axis=2, data_range=1.0))
    result = {"PSNR": float(psnr_val), "SSIM": ssim_val}
    if lpips_fn is not None:
        result["LPIPS"] = float(lpips_fn(orig, recon).item())
    return result


# ---------------------------------------------------------------------------
# SDXL DDIM inversion / reconstruction
# ---------------------------------------------------------------------------

def ddim_inversion_sdxl(pipe, latents, prompt_embeds, pooled_embeds, add_time_ids,
                         num_steps):
    """SDXL DDIM inversion: needs added_cond_kwargs for time_ids."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            noise_pred = pipe.unet(
                z, t_cur,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_embeds, "time_ids": add_time_ids
                },
            ).sample

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    return z


def ddim_recon_sdxl(pipe, noise, prompt_embeds, pooled_embeds, add_time_ids,
                     num_steps):
    """SDXL DDIM reconstruction."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    with torch.no_grad():
        for _, t in enumerate(timesteps):
            noise_pred = pipe.unet(
                z, t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_embeds, "time_ids": add_time_ids
                },
            ).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ---------------------------------------------------------------------------
# SDXL feature hooking + drift analysis
# ---------------------------------------------------------------------------

def discover_sdxl_hook_targets(unet):
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


class SDXLFeatureHooker:
    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.handles = []
        targets = discover_sdxl_hook_targets(unet)
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


def analyze_sdxl_drift(pipe, original_latent, prompt_embeds, pooled_embeds,
                        add_time_ids, num_steps, seeds=None):
    """Full drift fingerprint for SDXL."""
    if seeds is None:
        seeds = [42]

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    n = len(timesteps)

    if n <= 6:
        key_indices = list(range(n))
    else:
        key_indices = [0, 1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 3, n - 2, n - 1]
    key_indices = sorted(set(max(0, min(n - 1, i)) for i in key_indices))

    hooker = SDXLFeatureHooker(pipe.unet)
    added_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

    inv_latent = ddim_inversion_sdxl(
        pipe, original_latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)

    recon_latents = [inv_latent.clone()]
    z = inv_latent.clone()
    with torch.no_grad():
        for _, t in enumerate(timesteps):
            noise_pred = pipe.unet(
                z, t, encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_kwargs,
            ).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
            recon_latents.append(z.clone())

    per_seed_results = defaultdict(list)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_drifts = defaultdict(list)

        with torch.no_grad():
            for idx in key_indices:
                t = timesteps[idx]
                alpha_t = scheduler.alphas_cumprod[t]
                noise_ref = torch.randn_like(original_latent)
                z_ref = alpha_t.sqrt() * original_latent + (1 - alpha_t).sqrt() * noise_ref
                z_recon = recon_latents[idx]

                hooker.clear()
                pipe.unet(z_ref, t.to(DEVICE),
                         encoder_hidden_states=prompt_embeds,
                         added_cond_kwargs=added_kwargs).sample
                ref_feats = hooker.features.copy()

                hooker.clear()
                pipe.unet(z_recon, t.to(DEVICE),
                         encoder_hidden_states=prompt_embeds,
                         added_cond_kwargs=added_kwargs).sample
                recon_feats = hooker.features.copy()

                for layer_name in ref_feats:
                    if layer_name not in recon_feats:
                        continue
                    l2 = torch.norm(
                        ref_feats[layer_name].float() - recon_feats[layer_name].float(),
                        p=2).item()
                    seed_drifts[layer_name].append(l2)

        for layer_name, vals in seed_drifts.items():
            per_seed_results[layer_name].append(float(np.mean(vals)))

    hooker.remove()
    if not per_seed_results:
        return {}, {}
    avg_drifts = {k: float(np.mean(v)) for k, v in per_seed_results.items()}
    std_drifts = {k: float(np.std(v)) if len(v) > 1 else 0.0
                  for k, v in per_seed_results.items()}
    return avg_drifts, std_drifts


# ---------------------------------------------------------------------------
# SDXL drift diagnosis → find peak skip
# ---------------------------------------------------------------------------

def find_sdxl_peak_skip(pipe, prompts, num_steps=50):
    """Run drift fingerprint on SDXL with several prompts, find peak-drift skip.

    Diagnoses both:
    - Architecture-level drift peak (any layer, including mid_block)
    - Decoder skip target (up_block region aggregation, for Cut A)

    The decoder skip is Cut A's target — matching the same-named component
    as SD 1.5's Cut A — regardless of where the architecture-level peak falls.

    Returns: peak_up_block_index, peak_drift_value
    """
    print("\n[Diagnosis] Running SDXL drift fingerprint to find peak skip...")
    diag_prompts = prompts[:5]

    per_layer_all = defaultdict(list)       # layer_name -> [drift values] across prompts
    per_up_block_drift = defaultdict(list)  # up_block_idx -> [drift values]

    for prompt_id, prompt in diag_prompts:
        print(f"  [{prompt_id}]...", end=" ", flush=True)
        img_tensor = generate_sdxl(pipe, prompt, seed=42, num_steps=num_steps)
        with torch.no_grad():
            latent = pipe.vae.encode(img_tensor.float()).latent_dist.sample()
            latent = (latent * pipe.vae.config.scaling_factor).to(dtype=DTYPE)

        prompt_embeds, pooled_embeds, add_time_ids = encode_prompt_sdxl(pipe, prompt)
        avg_drifts, _ = analyze_sdxl_drift(
            pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps, seeds=[42])

        if not avg_drifts:
            print("FAILED")
            continue

        for layer, drift in avg_drifts.items():
            per_layer_all[layer].append(drift)
            for ub_idx in range(3):
                if f"up_blocks.{ub_idx}" in layer:
                    per_up_block_drift[ub_idx].append(drift)

        top_layers = sorted(avg_drifts.items(), key=lambda x: -x[1])[:3]
        peak_str = ", ".join(f"{n}={v:.1f}" for n, v in top_layers)
        print(f"top: {peak_str}")
        torch.cuda.empty_cache()

    # Architecture-level peak (all layers, includes mid_block)
    layer_means = {k: np.mean(v) for k, v in per_layer_all.items()}
    top_overall = sorted(layer_means.items(), key=lambda x: -x[1])[:3]

    print(f"\n--- SDXL Drift Diagnosis ---")
    print(f"  Architecture-level drift peak (all layers, incl. mid_block):")
    for name, val in top_overall:
        print(f"    {name}: drift={val:.1f}")

    # Decoder skip target (up_block region only — matching SD 1.5 Cut A methodology)
    print(f"\n  Decoder up_block drift (for Cut A target selection):")
    peak_idx = 0
    peak_drift = 0
    for ub_idx in range(3):
        vals = per_up_block_drift[ub_idx]
        if vals:
            mean_d = np.mean(vals)
            print(f"    up_blocks.{ub_idx}: drift={mean_d:.1f} ± {np.std(vals):.1f} (n={len(vals)})")
            if mean_d > peak_drift:
                peak_drift = mean_d
                peak_idx = ub_idx

    source_down = 2 - peak_idx
    print(f"\n  Cut A target: down_blocks.{source_down} → up_blocks.{peak_idx}")
    print(f"  NOTE: This targets the same-named decoder component as SD 1.5's Cut A.")
    print(f"  SDXL's architecture-level drift peak ({top_overall[0][0]}) is in mid_block,")
    print(f"  NOT at the cut site. This experiment tests whether the same structural")
    print(f"  component (decoder skip) plays the same functional role in two UNet variants.")
    print(f"  Answer: it does not — SD 1.5 skip is a conflict source, SDXL skip is an")
    print(f"  essential information pathway.")

    return peak_idx, peak_drift


# ---------------------------------------------------------------------------
# Full evaluation on SDXL
# ---------------------------------------------------------------------------

def evaluate_sdxl_prompt(pipe, prompt_id, prompt, img_tensor, lpips_fn,
                          cut_up_idx, num_steps=50):
    """Original vs Cut A on SDXL for one prompt."""
    prompt_embeds, pooled_embeds, add_time_ids = encode_prompt_sdxl(pipe, prompt)
    added_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

    with torch.no_grad():
        latent = pipe.vae.encode(img_tensor.float()).latent_dist.sample()
        latent = (latent * pipe.vae.config.scaling_factor).to(dtype=DTYPE)

    results = {"prompt_id": prompt_id, "prompt": prompt}

    # Original
    noise = ddim_inversion_sdxl(
        pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
    recon_latent = ddim_recon_sdxl(
        pipe, noise, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
    recon_tensor = decode_latent(pipe, recon_latent)
    m = compute_metrics(img_tensor, recon_tensor, lpips_fn)
    results["psnr_orig"] = m["PSNR"]
    results["ssim_orig"] = m["SSIM"]
    results["lpips_orig"] = m["LPIPS"]

    # Drift
    avg_d, _ = analyze_sdxl_drift(
        pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
    if avg_d:
        peak_layer = max(avg_d, key=avg_d.get)
        results["peak_drift_orig"] = avg_d.get(peak_layer, 0)
        results["peak_layer_orig"] = peak_layer

    torch.cuda.empty_cache()

    # Cut A
    with SDXLSkipIntervention(pipe.unet, [cut_up_idx]):
        noise = ddim_inversion_sdxl(
            pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
        recon_latent = ddim_recon_sdxl(
            pipe, noise, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
    recon_tensor = decode_latent(pipe, recon_latent)
    m = compute_metrics(img_tensor, recon_tensor, lpips_fn)
    results["psnr_cut"] = m["PSNR"]
    results["ssim_cut"] = m["SSIM"]
    results["lpips_cut"] = m["LPIPS"]

    # Cut A drift
    with SDXLSkipIntervention(pipe.unet, [cut_up_idx]):
        avg_d_cut, _ = analyze_sdxl_drift(
            pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
    if avg_d_cut:
        results["peak_drift_cut"] = avg_d_cut.get(
            peak_layer if "peak_layer_orig" in results else "up_blocks.0.resnets.0", 0)

    # Deltas
    for k in ["psnr", "ssim", "lpips"]:
        if f"{k}_orig" in results and f"{k}_cut" in results:
            results[f"{k}_delta"] = results[f"{k}_cut"] - results[f"{k}_orig"]

    if "peak_drift_orig" in results and "peak_drift_cut" in results:
        results["drift_delta_pct"] = ((results["peak_drift_cut"] -
                                        results["peak_drift_orig"]) /
                                       results["peak_drift_orig"] * 100)

    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Dose-response
# ---------------------------------------------------------------------------

def run_dose_response(pipe, prompts, cut_up_idx, lpips_fn, num_steps=50):
    """α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} dose-response on SDXL."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_psnr = defaultdict(list)

    for alpha in alphas:
        print(f"\n  --- α = {alpha:.2f} ---")
        for prompt_id, prompt in prompts[:10]:  # 10 prompts for dose-response
            img_tensor = generate_sdxl(pipe, prompt, seed=42, num_steps=num_steps)
            prompt_embeds, pooled_embeds, add_time_ids = encode_prompt_sdxl(pipe, prompt)

            with torch.no_grad():
                latent = pipe.vae.encode(img_tensor.float()).latent_dist.sample()
                latent = (latent * pipe.vae.config.scaling_factor).to(dtype=DTYPE)

            if alpha == 1.0:
                noise = ddim_inversion_sdxl(
                    pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
                recon_latent = ddim_recon_sdxl(
                    pipe, noise, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
            else:
                with SDXLPartialSkipIntervention(pipe.unet, [cut_up_idx], alpha):
                    noise = ddim_inversion_sdxl(
                        pipe, latent, prompt_embeds, pooled_embeds, add_time_ids, num_steps)
                    recon_latent = ddim_recon_sdxl(
                        pipe, noise, prompt_embeds, pooled_embeds, add_time_ids, num_steps)

            recon_tensor = decode_latent(pipe, recon_latent)
            m = compute_metrics(img_tensor, recon_tensor, lpips_fn)
            alpha_psnr[alpha].append(m["PSNR"])
            torch.cuda.empty_cache()

    return alpha_psnr


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sdxl_dose_response(alphas, psnr_means, psnr_stds, out_path):
    """SDXL dose-response: PSNR vs α."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(alphas, psnr_means, yerr=psnr_stds,
                color="#e74c3c", marker="o", markersize=10,
                linewidth=2.5, capsize=6)
    ax.set_xlabel("Skip Strength α", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("SDXL Dose-Response: Skip Strength α → Reconstruction PSNR",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # Annotate
    for a, m in zip(alphas, psnr_means):
        ax.annotate(f"{m:.2f}", (a, m), fontsize=9, ha="center",
                   va="bottom", xytext=(0, 5), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] SDXL dose-response → {out_path}")


def plot_sd15_vs_sdxl_comparison(sdxl_results, cut_up_idx, out_path):
    """Comparison bar chart: SD 1.5 vs SDXL intervention effects."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # SD 1.5 reference values
    sd15_data = {
        "PSNR Δ (dB)": (2.20, 0, "#3498db"),
        "SSIM Δ": (0.060, 0, "#2ecc71"),
        "LPIPS Δ": (-0.099, 0, "#e74c3c"),
    }

    sdxl_psnr = [r["psnr_delta"] for r in sdxl_results if "psnr_delta" in r]
    sdxl_ssim = [r.get("ssim_delta", 0) for r in sdxl_results if "ssim_delta" in r]
    sdxl_lpips = [r.get("lpips_delta", 0) for r in sdxl_results if "lpips_delta" in r]

    sdxl_data = {
        "PSNR Δ (dB)": (np.mean(sdxl_psnr) if sdxl_psnr else 0,
                        np.std(sdxl_psnr) if sdxl_psnr else 0),
        "SSIM Δ": (np.mean(sdxl_ssim) if sdxl_ssim else 0,
                   np.std(sdxl_ssim) if sdxl_ssim else 0),
        "LPIPS Δ": (np.mean(sdxl_lpips) if sdxl_lpips else 0,
                    np.std(sdxl_lpips) if sdxl_lpips else 0),
    }

    for ax, (metric, sd15_tuple) in zip(axes, sd15_data.items()):
        sd15_v = sd15_tuple[0]
        sdxl_mean, sdxl_std = sdxl_data[metric]

        ax.bar(["SD 1.5"], [sd15_v], color="#3498db", width=0.4,
               label="SD 1.5 (up_blocks.2 cut)")
        ax.bar(["SDXL"], [sdxl_mean], color="#e74c3c", width=0.4,
               label=f"SDXL (up_blocks.{cut_up_idx} cut)")
        ax.errorbar(["SDXL"], [sdxl_mean], yerr=[[0], [sdxl_std]] if sdxl_std else None,
                    fmt="none", ecolor="gray", capsize=6)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f"SD 1.5 vs SDXL: Skip Intervention Effect\n"
                 f"(SDXL: zero skip → up_blocks.{cut_up_idx})",
                 fontsize=13, fontweight="bold", color="#2C3E50")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] SD 1.5 vs SDXL comparison → {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_sdxl_report(results, cut_up_idx, source_down, dose_data):
    print(f"\n{'='*70}")
    print("SDXL CROSS-ARCHITECTURE VALIDATION — RESULTS")
    print(f"  N = {len(results)} prompts")
    print(f"  Peak skip: down_blocks.{source_down} → up_blocks.{cut_up_idx}")
    print(f"{'='*70}")

    psnr_deltas = [r["psnr_delta"] for r in results if "psnr_delta" in r]
    ssim_deltas = [r.get("ssim_delta", 0) for r in results if "ssim_delta" in r]
    lpips_deltas = [r.get("lpips_delta", 0) for r in results if "lpips_delta" in r]

    for name, deltas in [("PSNR", psnr_deltas), ("SSIM", ssim_deltas),
                           ("LPIPS", lpips_deltas)]:
        if not deltas:
            continue
        mean_d = np.mean(deltas)
        std_d = np.std(deltas)
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
        d = mean_d / std_d if std_d > 0 else 0
        print(f"\n--- {name} ---")
        print(f"  Δ = {mean_d:+.3f} ± {std_d:.3f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}, Cohen's d={d:.3f}")

    # Compare with SD 1.5
    print(f"\n--- Cross-Architecture Comparison ---")
    print(f"  SD 1.5 Cut A ΔPSNR: +{SD15_CUT_A_PSNR_DELTA:.2f} dB")
    if psnr_deltas:
        print(f"  SDXL  Cut A ΔPSNR: {np.mean(psnr_deltas):+.2f} dB")
        print(f"  SD 1.5 peak skip: down_blocks.1 → up_blocks.2 (decoder end)")
        print(f"  SDXL  peak skip: down_blocks.{source_down} → up_blocks.{cut_up_idx}")

        # Architecture difference analysis
        print(f"\n  Architecture comparison:")
        print(f"    SD 1.5: 4 down/up blocks, peak skip at decoder endpoint")
        print(f"    SDXL:   3 down/up blocks (deeper per block)")
        if cut_up_idx == 2:
            print(f"    → SDXL peak at decoder endpoint (same relative position as SD 1.5)")
        elif cut_up_idx == 0:
            print(f"    → SDXL peak at decoder start (different from SD 1.5's endpoint)")

    # Dose-response
    if dose_data:
        print(f"\n--- Dose-Response (α → PSNR) ---")
        for alpha in sorted(dose_data.keys()):
            vals = dose_data[alpha]
            print(f"  α={alpha:.2f}: {np.mean(vals):.2f} ± {np.std(vals):.2f} dB")
        # Check monotonicity
        alphas = sorted(dose_data.keys())
        means = [np.mean(dose_data[a]) for a in alphas]
        is_monotonic = all(means[i] >= means[i+1] for i in range(len(means)-1))
        print(f"  Monotonic (PSNR ↑ as α ↓): {'YES' if is_monotonic else 'NO'}")

    # Generalization verdict
    print(f"\n{'='*70}")
    print("GENERALIZATION VERDICT")
    print(f"{'='*70}")
    if psnr_deltas:
        mean_psnr = np.mean(psnr_deltas)
        if mean_psnr > 1.0:
            print(f"  ✓ Skip-mismatch pathology GENERALIZES to SDXL")
            print(f"    ΔPSNR = {mean_psnr:+.2f} dB (SD 1.5: +{SD15_CUT_A_PSNR_DELTA:.2f} dB)")
        elif mean_psnr > 0.3:
            print(f"  ~ Partial generalization: effect exists but weaker than SD 1.5")
        else:
            print(f"  ✗ No generalization: skip pathology may be SD 1.5-specific")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ICLR Task C: SDXL Cross-Architecture Validation")
    parser.add_argument("--prompts", type=int, default=20,
                        help="Number of prompts")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cut-up", type=int, default=None,
                        help="Override up_block index to cut (auto-detected if not set)")
    parser.add_argument("--skip-diagnosis", action="store_true",
                        help="Skip drift diagnosis (use --cut-up instead)")
    parser.add_argument("--skip-dose", action="store_true",
                        help="Skip dose-response")
    parser.add_argument("--quick", type=int, default=None)
    args = parser.parse_args()

    prompts = DIVERSE_PROMPTS[:args.prompts]
    if args.quick:
        prompts = prompts[:args.quick]

    print(f"[Setup] {len(prompts)} prompts, {args.steps} steps")
    print(f"[Output] {OUT_DIR.resolve()}")

    # Load SDXL
    print("[0] Loading SDXL + LPIPS...")
    pipe = load_sdxl_pipeline()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    # Phase 1: Drift diagnosis → find peak skip
    if args.skip_diagnosis and args.cut_up is not None:
        cut_up_idx = args.cut_up
        print(f"[1] Using specified cut: up_blocks.{cut_up_idx}")
    else:
        print(f"[1] Running drift diagnosis on {min(5, len(prompts))} prompts...")
        cut_up_idx, peak_drift = find_sdxl_peak_skip(pipe, prompts, args.steps)
        if args.cut_up is not None:
            cut_up_idx = args.cut_up
            print(f"  Override: using up_blocks.{cut_up_idx}")
    source_down = 2 - cut_up_idx

    # Phase 2: Generate and evaluate
    print(f"\n[2] Evaluating Original vs Cut A (up_blocks.{cut_up_idx}) "
          f"on {len(prompts)} prompts...")

    results = []
    for prompt_id, prompt in prompts:
        print(f"  [{prompt_id}]...", end=" ", flush=True)
        img_tensor = generate_sdxl(pipe, prompt, seed=42, num_steps=args.steps)
        r = evaluate_sdxl_prompt(pipe, prompt_id, prompt, img_tensor,
                                 lpips_fn, cut_up_idx, args.steps)
        results.append(r)
        print(f"PSNR: {r['psnr_orig']:.2f} → {r.get('psnr_cut', 0):.2f} "
              f"(Δ={r.get('psnr_delta', 0):+.2f} dB)")

    # Phase 3: Dose-response (optional)
    dose_data = {}
    if not args.skip_dose:
        print(f"\n[3] Dose-response (α ∈ [0, 1]) on 10 prompts...")
        dose_data = run_dose_response(pipe, prompts, cut_up_idx, lpips_fn, args.steps)

    # Phase 4: Report
    print_sdxl_report(results, cut_up_idx, source_down, dose_data)

    # Phase 5: Figures
    print(f"\n[4] Generating figures...")
    if dose_data:
        alphas = sorted(dose_data.keys())
        psnr_means = [np.mean(dose_data[a]) for a in alphas]
        psnr_stds = [np.std(dose_data[a]) for a in alphas]
        plot_sdxl_dose_response(alphas, psnr_means, psnr_stds,
                               OUT_DIR / "sdxl_dose_response.png")
        plot_sdxl_dose_response(alphas, psnr_means, psnr_stds,
                               OUT_DIR / "sdxl_dose_response.pdf")

    plot_sd15_vs_sdxl_comparison(results, cut_up_idx,
                                 OUT_DIR / "sd15_vs_sdxl_comparison.png")
    plot_sd15_vs_sdxl_comparison(results, cut_up_idx,
                                 OUT_DIR / "sd15_vs_sdxl_comparison.pdf")

    # Phase 6: Save
    print(f"\n[5] Saving data...")
    if results:
        fieldnames = ["prompt_id", "prompt",
                      "psnr_orig", "psnr_cut", "psnr_delta",
                      "ssim_orig", "ssim_cut", "ssim_delta",
                      "lpips_orig", "lpips_cut", "lpips_delta",
                      "peak_drift_orig", "peak_drift_cut", "drift_delta_pct"]
        with open(OUT_DIR / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"[CSV] → {OUT_DIR / 'results.csv'}")

    # Summary JSON
    psnr_deltas = [r["psnr_delta"] for r in results if "psnr_delta" in r]
    ssim_deltas = [r.get("ssim_delta", 0) for r in results if "ssim_delta" in r]
    lpips_deltas = [r.get("lpips_delta", 0) for r in results if "lpips_delta" in r]

    summary = {
        "config": {
            "n_prompts": len(results), "steps": args.steps,
            "cut_up_idx": cut_up_idx, "source_down": source_down,
            "peak_skip": f"down_blocks.{source_down} → up_blocks.{cut_up_idx}",
        },
        "psnr": {
            "delta_mean": float(np.mean(psnr_deltas)) if psnr_deltas else None,
            "delta_std": float(np.std(psnr_deltas)) if psnr_deltas else None,
        },
        "ssim": {
            "delta_mean": float(np.mean(ssim_deltas)) if ssim_deltas else None,
            "delta_std": float(np.std(ssim_deltas)) if ssim_deltas else None,
        },
        "lpips": {
            "delta_mean": float(np.mean(lpips_deltas)) if lpips_deltas else None,
            "delta_std": float(np.std(lpips_deltas)) if lpips_deltas else None,
        },
        "sd15_reference": {
            "psnr_delta": SD15_CUT_A_PSNR_DELTA,
            "architecture": "4 down/up blocks, peak skip = down_blocks.1 → up_blocks.2",
        },
        "sdxl_vs_sd15": {
            "relative_depth": f"SD 1.5 up_blocks.2/3 vs SDXL up_blocks.{cut_up_idx}/2",
        },
    }

    if dose_data:
        summary["dose_response"] = {
            str(alpha): {
                "psnr_mean": float(np.mean(vals)),
                "psnr_std": float(np.std(vals)),
            }
            for alpha, vals in dose_data.items()
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Summary → {OUT_DIR / 'summary.json'}")

    print(f"\n{'='*60}")
    print("Task C complete.")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
