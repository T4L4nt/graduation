"""
SDXL Phase 3: CLIP 正交投影 + 风格注入 + 钉扎约束

在 SDXL 上验证风格解耦框架：
1. CLIP 正交投影（架构无关，直接复用 phase3_prep）
2. 风格注入（SDXL 双文本编码器适配，2048-dim embedding）
3. 正交钉扎约束（像素空间，架构无关）
"""
import argparse, json, sys, time
from pathlib import Path

import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
import matplotlib; matplotlib.use("Agg")

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase3_prep import (
    CLIPFeatureExtractor, STYLE_CANDIDATES, find_closest_style,
)
from phase2_common import (
    FeatureCollector, FeatureCorrector, LambdaScheduler, StyleFeatureInjector,
    get_top_drift_layers, compute_metrics,
    DEVICE, DTYPE,
)
import lpips as _lpips_lib
from torchvision.utils import save_image as _save_image

def _get_lpips():
    return _lpips_lib.LPIPS(net="alex").to("cuda")

def _save(tensor, path):
    """Save [-1,1] tensor as PNG."""
    img = (tensor.squeeze(0) + 1) / 2
    _save_image(img.clamp(0, 1), path)

# SDXL-specific layer groups (from SDXL Phase 1 diagnostics)
SDXL_TOP5 = [
    "mid_block.resnets.1",
    "up_blocks.0.resnets.0",
    "up_blocks.0.resnets.1",
    "mid_block.resnets.0",
    "down_blocks.2.resnets.1",
]

OUT_DIR = Path("outputs/sdxl_phase3")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_SIZE = 1024
NUM_STEPS = 50


# ═══════════════════════════════════════════════════════════════
# SDXL Pipeline helpers
# ═══════════════════════════════════════════════════════════════

def load_sdxl_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(torch.float32)
    return pipe


def load_and_encode_sdxl(pipe, path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to("cuda", dtype=torch.float32)
    tensor = 2 * tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent.to(dtype=torch.float16), tensor, img


def decode_latent_sdxl(pipe, latent):
    with torch.no_grad():
        tensor = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
    return tensor


def encode_prompt_sdxl(pipe, prompt="", device="cuda"):
    (prompt_embeds, neg_embeds, pooled_embeds, neg_pooled) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]],
                            device=device, dtype=torch.float16)
    added_cond = {"text_embeds": pooled_embeds, "time_ids": time_ids}
    return prompt_embeds, pooled_embeds, added_cond


# ═══════════════════════════════════════════════════════════════
# SDXL DDIM inversion / reconstruction
# ═══════════════════════════════════════════════════════════════

def ddim_inversion_with_features_sdxl(pipe, latents, prompt_embeds, added_cond,
                                       num_steps, hook_layers):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps

    collector = FeatureCollector(pipe.unet, hook_layers)
    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    saved_features = {}

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            collector.clear()
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            saved_features[int(t_cur)] = {
                k: v.detach().cpu().clone() for k, v in collector.features.items()
            }
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            z = coeff1 * z + (sigma_next - coeff1 * sigma_cur) * noise_pred

    collector.remove()
    return z, saved_features


def ddim_inversion_sdxl(pipe, latents, prompt_embeds, added_cond, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps
    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]; t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            ac = scheduler.alphas_cumprod[t_cur]; an = scheduler.alphas_cumprod[t_next]
            c1 = (an / ac).sqrt()
            sc = (1 - ac).sqrt(); sn = (1 - an).sqrt()
            z = c1 * z + (sn - c1 * sc) * noise_pred
    return z


def ddim_reconstruction_sdxl(pipe, noise, prompt_embeds, added_cond, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ═══════════════════════════════════════════════════════════════
# SDXL Style Injection for cross-attention
# ═══════════════════════════════════════════════════════════════

def build_sdxl_style_prompt_embeds(pipe, base_prompt_embeds, v_style, strength=0.5):
    """Inject v_style into SDXL prompt embeddings.

    SDXL text embeddings: [1, 77, 2048] (concatenated CLIP-L + OpenCLIP-G/14).
    v_style: [1, 768] from CLIPFeatureExtractor.
    Dimension mismatch: 768 → 2048 via cyclic padding.
    """
    v = v_style.to(dtype=base_prompt_embeds.dtype, device=base_prompt_embeds.device)
    # Cyclic pad 768 → 2048
    D = base_prompt_embeds.shape[-1]
    repeats = (D + v.shape[-1] - 1) // v.shape[-1]
    v_pad = v.repeat(1, repeats)[:, :D]  # [1, 2048]
    v_broadcast = v_pad.unsqueeze(1).repeat(1, base_prompt_embeds.shape[1], 1)
    return (1.0 - strength) * base_prompt_embeds + strength * v_broadcast


# ═══════════════════════════════════════════════════════════════
# Full Pipeline: correction + style + pinning (SDXL)
# ═══════════════════════════════════════════════════════════════

def run_correction_with_style_and_pinning_sdxl(
    pipe, original_latent, original_tensor, prompt_embeds, added_cond,
    num_steps, corr_lam, corr_layers,
    styled_prompt_embeds,
    extractor, v_content,
    style_injector=None,
    pinning_freq=10, pinning_threshold=0.02, pinning_strength=0.5,
):
    """SDXL version: correction + style injection + orthogonal pinning."""
    t0 = time.perf_counter()

    # Reference content projection
    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(original_tensor), v_content)
    print(f"  [Pin] Reference content projection: {ref_proj:.4f}")

    # DDIM inversion with feature collection
    noise, saved = ddim_inversion_with_features_sdxl(
        pipe, original_latent, prompt_embeds, added_cond, num_steps, corr_layers,
    )

    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)

    base_strength = style_injector.strength if style_injector else 0.5

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps
    z = noise.clone()
    pinning_log = []
    effective_strength = base_strength

    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            if t_int in saved:
                corrector.set_reference(saved[t_int], step_idx)
            else:
                corrector.set_reference({}, step_idx)

            # Pinning check
            if step_idx > 0 and pinning_freq > 0 and step_idx % pinning_freq == 0:
                current_img = decode_latent_sdxl(pipe, z.clone())
                v_current = extractor.encode_image_from_tensor(current_img)
                cur_proj = extractor.compute_content_projection(v_current, v_content)
                deviation = abs(cur_proj - ref_proj)
                pinning_log.append({
                    "step": step_idx, "t": t_int,
                    "cur_proj": float(cur_proj), "ref_proj": float(ref_proj),
                    "deviation": float(deviation), "triggered": deviation > pinning_threshold,
                    "effective_strength": effective_strength,
                })

                if deviation > pinning_threshold:
                    scale = max(0.0, 1.0 - pinning_strength * (deviation / max(ref_proj, 0.01)))
                    effective_strength = base_strength * scale
                    if style_injector is not None:
                        style_injector.set_strength(effective_strength)
                    print(f"  [Pin] step={step_idx:3d} dev={deviation:.4f}  "
                          f"→ style={effective_strength:.3f}")

            # UNet forward with style-modified prompt embeds
            noise_pred = pipe.unet(z, t, encoder_hidden_states=styled_prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

    corrector.remove()
    if style_injector is not None:
        style_injector.remove()

    recon = decode_latent_sdxl(pipe, z)
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(original_tensor, recon, _get_lpips())

    if pinning_log:
        max_dev = max(d["deviation"] for d in pinning_log)
        triggered = sum(1 for d in pinning_log if d["triggered"])
        print(f"  [Pin] checks={len(pinning_log)} max_dev={max_dev:.4f} "
              f"triggered={triggered}/{len(pinning_log)}")

    return metrics, recon, elapsed, pinning_log


# ═══════════════════════════════════════════════════════════════
# Mode: full
# ═══════════════════════════════════════════════════════════════

def mode_full(pipe, img_path, extractor, v_content):
    print(f"\n{'='*60}")
    print("SDXL Full Pipeline: Correction + Style + Pinning")
    print(f"{'='*60}")

    cond = encode_prompt_sdxl(pipe, "")
    prompt_embeds, pooled_embeds, added_cond = cond
    latents, tensor, _ = load_and_encode_sdxl(pipe, img_path)
    img_name = Path(img_path).stem

    # Get style direction
    v_img = extractor.encode_image(img_path)
    _, v_style, _ = extractor.compute_orthogonal_decomposition(v_img, v_content)
    style_text, style_key, sim = find_closest_style(v_style, extractor)
    print(f"Image: {img_name}")
    print(f"Style: {style_key} ({style_text}) [sim={sim:.3f}]")

    # 1. Baseline
    print(f"\n[1/4] Baseline...")
    t0 = time.perf_counter()
    noise_bl = ddim_inversion_sdxl(pipe, latents, prompt_embeds, added_cond, NUM_STEPS)
    recon_bl = ddim_reconstruction_sdxl(pipe, noise_bl, prompt_embeds, added_cond, NUM_STEPS)
    recon_bl_img = decode_latent_sdxl(pipe, recon_bl)
    m_bl = compute_metrics(tensor, recon_bl_img, _get_lpips())
    print(f"  PSNR={m_bl['PSNR']:.2f}  LPIPS={m_bl['LPIPS']:.4f}  ({time.perf_counter()-t0:.1f}s)")
    _save(recon_bl_img, OUT_DIR / f"{img_name}_baseline.png")

    # 2. Correction only (Phase 2)
    print(f"\n[2/4] Correction only (top-5, λ=0.7)...")
    t0 = time.perf_counter()
    noise_c, saved_c = ddim_inversion_with_features_sdxl(
        pipe, latents, prompt_embeds, added_cond, NUM_STEPS, SDXL_TOP5,
    )
    sched_c = LambdaScheduler(0.7, NUM_STEPS, "constant")
    corrector = FeatureCorrector(pipe.unet, SDXL_TOP5, sched_c)
    recon_c = ddim_reconstruction_with_correction_sdxl(
        pipe, noise_c, prompt_embeds, added_cond, NUM_STEPS, saved_c, corrector,
    )
    corrector.remove()
    recon_c_img = decode_latent_sdxl(pipe, recon_c)
    m_c = compute_metrics(tensor, recon_c_img, _get_lpips())
    print(f"  PSNR={m_c['PSNR']:.2f}  LPIPS={m_c['LPIPS']:.4f}  ({time.perf_counter()-t0:.1f}s)")
    _save(recon_c_img, OUT_DIR / f"{img_name}_correction.png")

    # 3. Correction + Style (no pinning)
    print(f"\n[3/4] Correction + Style (no pinning)...")
    t0 = time.perf_counter()
    styled_emb = build_sdxl_style_prompt_embeds(pipe, prompt_embeds, v_style, strength=0.5)
    injector = StyleFeatureInjector(pipe.unet, SDXL_TOP5, v_style, strength=0.3)

    noise_s, saved_s = ddim_inversion_with_features_sdxl(
        pipe, latents, prompt_embeds, added_cond, NUM_STEPS, SDXL_TOP5,
    )
    sched_s = LambdaScheduler(0.7, NUM_STEPS, "constant")
    corrector_s = FeatureCorrector(pipe.unet, SDXL_TOP5, sched_s)

    scheduler = pipe.scheduler; scheduler.set_timesteps(NUM_STEPS, device="cuda")
    z_s = noise_s.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(scheduler.timesteps):
            t_int = int(t)
            if t_int in saved_s:
                corrector_s.set_reference(saved_s[t_int], step_idx)
            noise_pred = pipe.unet(z_s, t, encoder_hidden_states=styled_emb,
                                   added_cond_kwargs=added_cond).sample
            z_s = scheduler.step(noise_pred, t, z_s).prev_sample

    corrector_s.remove(); injector.remove()
    recon_s_img = decode_latent_sdxl(pipe, z_s)
    m_s = compute_metrics(tensor, recon_s_img, _get_lpips())
    print(f"  PSNR={m_s['PSNR']:.2f}  LPIPS={m_s['LPIPS']:.4f}  ({time.perf_counter()-t0:.1f}s)")
    _save(recon_s_img, OUT_DIR / f"{img_name}_style_no_pin.png")

    # 4. Correction + Style + Pinning
    print(f"\n[4/4] Correction + Style + Pinning...")
    injector_p = StyleFeatureInjector(pipe.unet, SDXL_TOP5, v_style, strength=0.3)

    m_p, recon_p, _, pin_log = run_correction_with_style_and_pinning_sdxl(
        pipe, latents, tensor, prompt_embeds, added_cond,
        NUM_STEPS, 0.7, SDXL_TOP5,
        styled_emb, extractor, v_content,
        style_injector=injector_p,
        pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5,
    )
    print(f"  PSNR={m_p['PSNR']:.2f}  LPIPS={m_p['LPIPS']:.4f}")
    _save(recon_p, OUT_DIR / f"{img_name}_style_pin.png")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Method':<30}{'PSNR':>8}{'LPIPS':>8}")
    print(f"{'-'*46}")
    for name, m in [("Baseline", m_bl), ("Correction", m_c),
                     ("Style (no pin)", m_s), ("Style + Pin", m_p)]:
        print(f"{name:<30}{m['PSNR']:>8.2f}{m['LPIPS']:>8.4f}")

    results = [
        {"method": "baseline", **{k: float(v) for k, v in m_bl.items()}},
        {"method": "correction", **{k: float(v) for k, v in m_c.items()}},
        {"method": "style_no_pin", **{k: float(v) for k, v in m_s.items()}},
        {"method": "style_pin", **{k: float(v) for k, v in m_p.items()}},
    ]
    with open(OUT_DIR / f"{img_name}_results.json", "w") as f:
        json.dump({"image": img_name, "results": results, "pinning_log": pin_log},
                  f, indent=2, default=str)

    return results


# ═══════════════════════════════════════════════════════════════
# Helper: reconstruction with correction (SDXL)
# ═══════════════════════════════════════════════════════════════

def ddim_reconstruction_with_correction_sdxl(pipe, noise, prompt_embeds, added_cond,
                                              num_steps, saved_features, corrector):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            if t_int in saved_features:
                corrector.set_reference(saved_features[t_int], step_idx)
            else:
                corrector.set_reference({}, step_idx)
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--n-images", type=int, default=None,
                        help="Number of coco_val images to process (overrides --image)")
    parser.add_argument("--mode", default="full", choices=["clip", "full"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SDXL Phase 3: Style Disentanglement & Pinning")
    print("=" * 60)

    # CLIP orthogonal projection (architecture agnostic)
    print("\n[Init] CLIP ViT-L/14 orthogonal projection...")
    extractor = CLIPFeatureExtractor()
    v_content = extractor.encode_text("a photo of natural scene")

    if args.mode == "clip":
        return

    print("\n[Init] Loading SDXL pipeline...")
    pipe = load_sdxl_pipeline()

    # Determine images
    if args.n_images is not None:
        coco_images = sorted(Path("data/coco_val").glob("*.jpg"))
        images = coco_images[:args.n_images]
    elif args.image:
        images = [args.image]
    else:
        images = ["data/coco_val/coco_000000000802.jpg"]

    print(f"\nProcessing {len(images)} image(s):")
    for p in images:
        print(f"  {Path(p).name}")

    all_results = []
    for img_path in images:
        results = mode_full(pipe, str(img_path), extractor, v_content)
        all_results.append({"image": Path(img_path).stem, "results": results})

    # Save aggregate summary
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary table
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    for entry in all_results:
        print(f"\n{entry['image']}:")
        print(f"{'Method':<30}{'PSNR':>8}{'LPIPS':>8}")
        print(f"{'-'*46}")
        for r in entry["results"]:
            m = r
            print(f"{r['method']:<30}{m['PSNR']:>8.2f}{m['LPIPS']:>8.4f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
