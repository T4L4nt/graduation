"""
SDXL Phase 2: Zero-training residual correction for DDIM inversion-reconstruction.

Adapts the Phase 2 pipeline (FeatureCorrector, lambda scan, ablation) to SDXL.
Reuses architecture-agnostic FeatureCorrector/FeatureCollector from phase2_common.
"""
import argparse, json, csv, sys, time
from pathlib import Path
from collections import defaultdict

import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    get_top_drift_layers, compute_metrics, histogram_match,
    make_grid_image, save_recon_img, save_results_csv,
    DEVICE, DTYPE,
)

# SDXL overrides
DEVICE_SDX = "cuda"
DTYPE_SDX = torch.float16
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUT_DIR = Path("outputs/sdxl_phase2")

# SDXL layer configs — dynamically discovered, with presets for ablation
# The SDXL UNet has 3 down/up blocks (not 4), different from SD 1.5

# Top-5 from SDXL Phase 1 diagnostics (mid_block dominant)
SDXL_TOP5 = [
    "mid_block.resnets.1",
    "up_blocks.0.resnets.0",
    "up_blocks.0.resnets.1",
    "mid_block.resnets.0",
    "down_blocks.2.resnets.1",
]

# Ablation: encoder-only ResNet blocks (top-5 by SDXL drift)
SDXL_ENCODER5 = [
    "down_blocks.2.resnets.1",
    "down_blocks.2.resnets.0",
    "down_blocks.1.resnets.1",
    "down_blocks.1.resnets.0",
    "down_blocks.0.resnets.0",
]

# Ablation: top-5 attention layers by drift
SDXL_ATTENTION5 = [
    "down_blocks.2.attentions.0.transformer_blocks.0",
    "mid_block.attentions.0.transformer_blocks.0",
    "up_blocks.0.attentions.1.transformer_blocks.0",
    "down_blocks.2.attentions.1.transformer_blocks.0",
    "up_blocks.0.attentions.0.transformer_blocks.0",
]

# Ablation: random 5 up_blocks resnets
SDXL_RANDOM5 = [
    "up_blocks.0.resnets.2",
    "up_blocks.1.resnets.1",
    "up_blocks.2.resnets.0",
    "up_blocks.1.resnets.0",
    "up_blocks.2.resnets.2",
]

SDXL_LAYER_GROUPS = {
    "top5": SDXL_TOP5,
    "encoder5": SDXL_ENCODER5,
    "attention5": SDXL_ATTENTION5,
    "random5": SDXL_RANDOM5,
    "mid_block": ["mid_block.resnets.1", "mid_block.resnets.0"],
}

COCO_IMAGES = sorted(Path("data/coco_val").glob("*.jpg"))
LAMBDA_CANDIDATES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


# ═══════════════════════════════════════════════════════════════
# SDXL pipeline helpers
# ═══════════════════════════════════════════════════════════════

def load_sdxl_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE_SDX, local_files_only=True,
    ).to(DEVICE_SDX)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.to(torch.float32)  # NaN in fp16
    return pipe


def load_and_encode_sdxl(pipe, path: str, size=1024):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    tensor_fp32 = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE_SDX, dtype=torch.float32)
    tensor_fp32 = 2 * tensor_fp32 - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor_fp32).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent.to(dtype=DTYPE_SDX), tensor_fp32, img


def decode_latent_sdxl(pipe, latent):
    with torch.no_grad():
        tensor = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
    return tensor


def encode_prompt_sdxl(pipe, prompt="", device=DEVICE_SDX):
    (prompt_embeds, neg_embeds, pooled_embeds, neg_pooled) = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]],
                            device=device, dtype=DTYPE_SDX)
    added_cond = {"text_embeds": pooled_embeds, "time_ids": time_ids}
    return prompt_embeds, pooled_embeds, added_cond


# ═══════════════════════════════════════════════════════════════
# SDXL DDIM inversion / reconstruction
# ═══════════════════════════════════════════════════════════════

def ddim_inversion_with_features_sdxl(pipe, latents, prompt_embeds, added_cond,
                                      num_steps, hook_layers):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE_SDX)
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
            saved_features[int(t_cur)] = collector.features.copy()

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    collector.remove()
    return z, saved_features


def ddim_reconstruction_with_correction_sdxl(pipe, noise, prompt_embeds, added_cond,
                                             num_steps, saved_features, corrector):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE_SDX)
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


def ddim_inversion_sdxl(pipe, latents, prompt_embeds, added_cond, num_steps):
    """Baseline inversion (no feature collection)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE_SDX)
    timesteps = scheduler.timesteps
    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
    return z


def ddim_reconstruction_sdxl(pipe, noise, prompt_embeds, added_cond, num_steps):
    """Baseline reconstruction (no correction)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE_SDX)
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ═══════════════════════════════════════════════════════════════
# Experiment: lambda scan
# ═══════════════════════════════════════════════════════════════

def run_lambda_scan(pipe, images, num_steps=50, group_name="top5", split="test",
                    save_images=True, lpips_fn=None):
    """Scan lambda values, optionally on the held-out test split."""
    layers = SDXL_LAYER_GROUPS.get(group_name, SDXL_TOP5)
    print(f"  Lambda scan: group={group_name}, layers={layers}")
    print(f"  Lambdas: {LAMBDA_CANDIDATES}")

    results = []
    for img_path in images:
        img_name = Path(img_path).stem
        print(f"\n  [{img_name}]")

        latents, tensor_fp32, pil_img = load_and_encode_sdxl(pipe, str(img_path))
        prompt_embeds, _, added_cond = encode_prompt_sdxl(pipe, "")

        # Baseline (lambda = 0)
        noise = ddim_inversion_sdxl(pipe, latents, prompt_embeds, added_cond, num_steps)
        recon_base = ddim_reconstruction_sdxl(pipe, noise, prompt_embeds, added_cond, num_steps)
        recon_tensor_base = decode_latent_sdxl(pipe, recon_base)

        if lpips_fn is not None:
            metrics_base = compute_metrics(tensor_fp32, recon_tensor_base, lpips_fn)
        else:
            metrics_base = {"PSNR": 0, "SSIM": 0, "LPIPS": 0}
        print(f"    baseline: PSNR={metrics_base['PSNR']:.2f}, LPIPS={metrics_base['LPIPS']:.4f}")

        # Lambda scan
        for lam in LAMBDA_CANDIDATES:
            noise, saved_features = ddim_inversion_with_features_sdxl(
                pipe, latents, prompt_embeds, added_cond, num_steps, layers,
            )
            corrector = FeatureCorrector(pipe.unet, layers, lam)
            recon_lat = ddim_reconstruction_with_correction_sdxl(
                pipe, noise, prompt_embeds, added_cond,
                num_steps, saved_features, corrector,
            )
            corrector.remove()

            recon_tensor = decode_latent_sdxl(pipe, recon_lat)
            if lpips_fn is not None:
                m = compute_metrics(tensor_fp32, recon_tensor, lpips_fn)
            else:
                m = {"PSNR": 0, "SSIM": 0, "LPIPS": 0}

            delta_psnr = m["PSNR"] - metrics_base["PSNR"]
            print(f"    λ={lam:.1f}: PSNR={m['PSNR']:.2f} (Δ{delta_psnr:+.2f}), "
                  f"LPIPS={m['LPIPS']:.4f}")

            results.append({
                "image": img_name, "lambda": lam, "group": group_name,
                "steps": num_steps,
                "PSNR": m["PSNR"], "SSIM": m["SSIM"], "LPIPS": m["LPIPS"],
                "Δ_PSNR": delta_psnr,
                "baseline_PSNR": metrics_base["PSNR"],
                "baseline_LPIPS": metrics_base["LPIPS"],
            })

            if save_images and lam in [0.3, 0.5, 0.7]:
                save_recon_img(recon_tensor, OUT_DIR, img_name, num_steps,
                               f"corr-{group_name}-l{lam}", subdir=group_name)

        if save_images:
            save_recon_img(recon_tensor_base, OUT_DIR, img_name, num_steps,
                           "baseline", subdir=group_name)

    return results


# ═══════════════════════════════════════════════════════════════
# Experiment: ablation
# ═══════════════════════════════════════════════════════════════

def run_ablation(pipe, images, num_steps=50, lam=0.7, lpips_fn=None):
    """Compare different layer groups at fixed lambda."""
    print(f"  Ablation: lambda={lam}, steps={num_steps}")

    all_results = []
    for img_path in images:
        img_name = Path(img_path).stem
        print(f"\n  [{img_name}]")
        latents, tensor_fp32, _ = load_and_encode_sdxl(pipe, str(img_path))
        prompt_embeds, _, added_cond = encode_prompt_sdxl(pipe, "")

        # Baseline
        noise = ddim_inversion_sdxl(pipe, latents, prompt_embeds, added_cond, num_steps)
        recon_base = ddim_reconstruction_sdxl(pipe, noise, prompt_embeds, added_cond, num_steps)
        recon_tensor_base = decode_latent_sdxl(pipe, recon_base)
        if lpips_fn is not None:
            base_m = compute_metrics(tensor_fp32, recon_tensor_base, lpips_fn)
        else:
            base_m = {"PSNR": 0, "SSIM": 0, "LPIPS": 0}

        for group_name, layers in SDXL_LAYER_GROUPS.items():
            noise, saved_features = ddim_inversion_with_features_sdxl(
                pipe, latents, prompt_embeds, added_cond, num_steps, layers,
            )
            corrector = FeatureCorrector(pipe.unet, layers, lam)
            recon_lat = ddim_reconstruction_with_correction_sdxl(
                pipe, noise, prompt_embeds, added_cond,
                num_steps, saved_features, corrector,
            )
            corrector.remove()
            recon_tensor = decode_latent_sdxl(pipe, recon_lat)
            if lpips_fn is not None:
                m = compute_metrics(tensor_fp32, recon_tensor, lpips_fn)
            else:
                m = {"PSNR": 0, "SSIM": 0, "LPIPS": 0}

            delta_psnr = m["PSNR"] - base_m["PSNR"]
            print(f"    {group_name:15s}: PSNR={m['PSNR']:.2f} (Δ{delta_psnr:+.2f}), "
                  f"LPIPS={m['LPIPS']:.4f}")
            all_results.append({
                "image": img_name, "group": group_name, "lambda": lam,
                "PSNR": m["PSNR"], "SSIM": m["SSIM"], "LPIPS": m["LPIPS"],
                "Δ_PSNR": delta_psnr,
                "baseline_PSNR": base_m["PSNR"],
            })

    return all_results


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_lambda_curve(results, out_path):
    """PSNR / LPIPS vs lambda curves."""
    df = defaultdict(lambda: defaultdict(list))
    for r in results:
        df[r["lambda"]]["psnr"].append(r["PSNR"])
        df[r["lambda"]]["lpips"].append(r["LPIPS"])
        df[r["lambda"]]["delta"].append(r["Δ_PSNR"])

    lambdas = sorted(df.keys())
    mean_psnr = [np.mean(df[l]["psnr"]) for l in lambdas]
    mean_lpips = [np.mean(df[l]["lpips"]) for l in lambdas]
    mean_delta = [np.mean(df[l]["delta"]) for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(lambdas, mean_psnr, "o-", color="steelblue", ms=6, label="PSNR")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("SDXL: PSNR vs λ")
    ax1.grid(True, alpha=0.3)

    ax2.plot(lambdas, mean_lpips, "s-", color="coral", ms=6, label="LPIPS")
    ax2.set_xlabel("λ")
    ax2.set_ylabel("LPIPS")
    ax2.set_title("SDXL: LPIPS vs λ")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ablation(results, out_path):
    """Grouped bar chart of Δ PSNR per ablation group."""
    groups = list(SDXL_LAYER_GROUPS.keys())
    group_delta = {g: [] for g in groups}
    for r in results:
        group_delta[r["group"]].append(r["Δ_PSNR"])

    means = [np.mean(group_delta[g]) for g in groups]
    stds = [np.std(group_delta[g]) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    bars = ax.bar(range(len(groups)), means, yerr=stds, color=colors,
                  capsize=5, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Δ PSNR (dB)")
    ax.set_title("SDXL: Ablation — Δ PSNR by Layer Group (λ=0.7)")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full",
                        choices=["quick", "full", "ablation", "lambda"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--group", default="top5",
                        choices=list(SDXL_LAYER_GROUPS.keys()))
    parser.add_argument("--lam", type=float, default=0.7)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "recon").mkdir(exist_ok=True)

    print("=" * 60)
    print("SDXL Phase 2: Residual Correction — λ scan + Ablation")
    print("=" * 60)

    # Pipeline
    print("\n[1/3] Loading SDXL pipeline...")
    pipe = load_sdxl_pipeline()

    # LPIPS
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE_SDX)
    except ImportError:
        lpips_fn = None
        print("[WARN] lpips not available, skipping LPIPS metrics")

    # Images
    if args.mode == "quick":
        images = [COCO_IMAGES[0]]  # 1 image for quick test
    else:
        images = COCO_IMAGES[:args.n_images] if args.n_images else COCO_IMAGES

    print(f"\n[2/3] Test images: {len(images)}")
    for p in images:
        print(f"  {p.name}")

    print(f"\n[3/3] Running experiments (mode={args.mode})...")

    all_results = []

    if args.mode in ("quick", "lambda", "full"):
        print(f"\n--- Lambda Scan (group={args.group}) ---")
        res = run_lambda_scan(pipe, images, num_steps=args.steps,
                              group_name=args.group, lpips_fn=lpips_fn)
        all_results.extend(res)
        plot_lambda_curve(res, OUT_DIR / "lambda_curve.png")

    if args.mode in ("ablation", "full"):
        print(f"\n--- Ablation ---")
        abl_imgs = images[:3] if len(images) >= 3 else images  # ablation on 3 images
        res = run_ablation(pipe, abl_imgs, num_steps=args.steps,
                           lam=args.lam, lpips_fn=lpips_fn)
        all_results.extend(res)
        plot_ablation(res, OUT_DIR / "ablation_delta_psnr.png")

    # Save CSV
    if all_results:
        save_results_csv(all_results, OUT_DIR, "sdxl_phase2_results", "")

    # Summary
    if all_results:
        deltas = [r["Δ_PSNR"] for r in all_results if r.get("group", "") == args.group
                  and abs(r.get("lambda", 0) - args.lam) < 0.01]
        if deltas:
            print(f"\n{'='*60}")
            print(f"Summary: λ={args.lam}, group={args.group}")
            print(f"  Mean Δ PSNR: {np.mean(deltas):+.2f} dB")
            print(f"  Max  Δ PSNR: {np.max(deltas):+.2f} dB")
            print(f"  N images: {len(deltas)}")

    print(f"\nDone. Output: {OUT_DIR}/")
    print(f"  lambda_curve.png, ablation_delta_psnr.png")
    print(f"  sdxl_phase2_results.csv")


if __name__ == "__main__":
    main()
