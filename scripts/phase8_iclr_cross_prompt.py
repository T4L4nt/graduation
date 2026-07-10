"""
Phase 8 ICLR — Task A: Cross-Prompt Statistical Validation of Skip Intervention

≥25 diverse prompts × SD 1.5 generation → DDIM inversion → Cut A vs Original
reconstruction. Computes PSNR/SSIM/LPIPS + peak drift + significant layers per prompt.

Output:
  - CSV: per-prompt metrics with deltas
  - Statistical summary: paired t-test, Cohen's d, bootstrap CI
  - Figure: PSNR delta histogram + KDE
  - Report: how many prompts gain >1.0/>2.0 dB, outliers
"""

import argparse, json, csv, sys, os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from phase7_skip_intervention import (
    SkipIntervention, load_pipeline, load_and_encode,
    ddim_inversion, analyze_layer_drift,
    aggregate_across_images, paired_ttest_per_layer, layer_sort_key
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"

OUT_DIR = Path("outputs/phase8_iclr_cross_prompt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 25 Diverse prompts covering 6 categories
# ---------------------------------------------------------------------------
DIVERSE_PROMPTS = [
    # Portraits (4)
    ("portrait_01", "a professional headshot of a woman with natural lighting"),
    ("portrait_02", "a candid street portrait of an elderly man with wrinkles"),
    ("portrait_03", "a young child laughing, soft window light"),
    ("portrait_04", "a muscular athlete posing after training, dramatic lighting"),

    # Landscapes (4)
    ("landscape_01", "a mountain lake at sunrise with mist over the water"),
    ("landscape_02", "a vast desert with sand dunes and a lone cactus"),
    ("landscape_03", "a snowy pine forest with a wooden cabin, winter morning"),
    ("landscape_04", "a tropical beach with palm trees and turquoise water"),

    # Animals (4)
    ("animal_01", "a golden retriever puppy sitting in a flower garden"),
    ("animal_02", "a bald eagle soaring over a canyon, wings spread"),
    ("animal_03", "a white cat sleeping on a windowsill, afternoon sun"),
    ("animal_04", "a herd of wild horses galloping across a prairie"),

    # Objects (4)
    ("object_01", "a vintage film camera on a wooden table, macro shot"),
    ("object_02", "a steaming cup of coffee next to an open book"),
    ("object_03", "a classic red sports car parked on a coastal road"),
    ("object_04", "a handcrafted ceramic vase with dried flowers"),

    # Abstract (4)
    ("abstract_01", "geometric patterns in neon colors, digital art style"),
    ("abstract_02", "swirling galaxies and cosmic dust, space photography"),
    ("abstract_03", "liquid metal flowing into organic shapes, 3D render"),
    ("abstract_04", "fractal patterns resembling a kaleidoscope, vibrant colors"),

    # Text / signage (5)
    ("text_01", "a storefront window with hand-painted lettering and reflections"),
    ("text_02", "a vintage neon sign reading OPEN at night, urban street"),
    ("text_03", "a graffiti mural on a brick wall, colorful street art"),
    ("text_04", "a minimalist poster with bold typography on a gallery wall"),
    ("text_05", "a restaurant menu board with chalk lettering, warm interior"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_latent(pipe, latent):
    with torch.no_grad():
        return pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample


def compute_metrics(original_tensor, recon_tensor, lpips_fn=None):
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


def ddim_reconstruction(pipe, noise, prompt_embeds, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for _, t in enumerate(timesteps):
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ---------------------------------------------------------------------------
# Generation from prompt
# ---------------------------------------------------------------------------

def generate_image(pipe, prompt, seed=42, num_steps=50, guidance_scale=7.5):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_steps, device=DEVICE)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
        )
    return result.images  # [-1, 1] tensor


def generate_and_save(pipe, prompt_id, prompt, num_steps=50):
    """Generate image, save it, return (tensor, latent)."""
    img_tensor = generate_image(pipe, prompt, seed=42, num_steps=num_steps)
    img_path = OUT_DIR / "generated" / f"{prompt_id}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(img_path)

    with torch.no_grad():
        latent = pipe.vae.encode(img_tensor.to(DEVICE, dtype=DTYPE)).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return img_tensor, latent


# ---------------------------------------------------------------------------
# Per-prompt evaluation
# ---------------------------------------------------------------------------

def evaluate_prompt(pipe, prompt_id, prompt, img_tensor, latent,
                     lpips_fn, num_steps=50):
    """Run Cut A vs Original inversion-reconstruction for one prompt."""
    prompt_embeds = pipe.encode_prompt(prompt, DEVICE, 1, False)[0]

    results = {"prompt_id": prompt_id, "prompt": prompt}

    # --- Original (no intervention) ---
    noise = ddim_inversion(pipe, latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    recon_tensor_orig = decode_latent(pipe, recon_latent)
    metrics_orig = compute_metrics(img_tensor, recon_tensor_orig, lpips_fn)
    for k, v in metrics_orig.items():
        results[f"{k.lower()}_orig"] = v

    # Drift for Original
    avg_drifts_orig, _ = analyze_layer_drift(
        pipe, latent, prompt_embeds, num_steps, seeds=[42])
    if avg_drifts_orig:
        peak_layer = max(avg_drifts_orig, key=avg_drifts_orig.get)
        results["peak_drift_orig"] = avg_drifts_orig.get(peak_layer, 0)
        results["peak_layer_orig"] = peak_layer
    else:
        results["peak_drift_orig"] = float("nan")
        results["peak_layer_orig"] = ""

    torch.cuda.empty_cache()

    # --- Cut A (zero skip → up_blocks.2) ---
    with SkipIntervention(pipe.unet, [2]):
        noise = ddim_inversion(pipe, latent, prompt_embeds, num_steps)
        recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    recon_tensor_cut = decode_latent(pipe, recon_latent)
    metrics_cut = compute_metrics(img_tensor, recon_tensor_cut, lpips_fn)
    for k, v in metrics_cut.items():
        results[f"{k.lower()}_cut"] = v

    # Drift for Cut A
    with SkipIntervention(pipe.unet, [2]):
        avg_drifts_cut, _ = analyze_layer_drift(
            pipe, latent, prompt_embeds, num_steps, seeds=[42])
    if avg_drifts_cut:
        peak_layer_cut = max(avg_drifts_cut, key=avg_drifts_cut.get)
        results["peak_drift_cut"] = avg_drifts_cut.get(peak_layer_cut, 0)
    else:
        results["peak_drift_cut"] = float("nan")

    # Compute deltas
    for k in ["psnr", "ssim", "lpips"]:
        if f"{k}_orig" in results and f"{k}_cut" in results:
            results[f"{k}_delta"] = results[f"{k}_cut"] - results[f"{k}_orig"]

    if "peak_drift_orig" in results and "peak_drift_cut" in results:
        if not np.isnan(results["peak_drift_orig"]) and not np.isnan(results["peak_drift_cut"]):
            results["drift_delta"] = results["peak_drift_cut"] - results["peak_drift_orig"]
            results["drift_delta_pct"] = (results["drift_delta"] / results["peak_drift_orig"]) * 100

    # Count significant layers: need full drift comparison
    if avg_drifts_orig and avg_drifts_cut:
        common_layers = set(avg_drifts_orig.keys()) & set(avg_drifts_cut.keys())
        # Per-prompt we can't do t-test; flag peak layer change direction
        peak_d = results.get("drift_delta", 0)
        results["drift_decreased"] = bool(peak_d < 0)

    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Full cross-prompt drift fingerprint (optional, for significant layer counting)
# ---------------------------------------------------------------------------

def run_full_drift_comparison(pipe, image_paths, prompts, num_steps=50):
    """Run full per-layer drift comparison (Cut A vs Original) across all prompts.
    Returns per-image drift dicts for t-test."""
    drifts_orig = {}
    drifts_cut = {}

    for prompt_id, prompt in prompts:
        latent, img_tensor = generate_and_save(pipe, prompt_id, prompt, num_steps)
        prompt_embeds = pipe.encode_prompt(prompt, DEVICE, 1, False)[0]

        # Original drift
        avg_d, _ = analyze_layer_drift(pipe, latent, prompt_embeds, num_steps, seeds=[42])
        if avg_d:
            drifts_orig[prompt_id] = avg_d

        # Cut A drift
        with SkipIntervention(pipe.unet, [2]):
            avg_d_cut, _ = analyze_layer_drift(pipe, latent, prompt_embeds, num_steps, seeds=[42])
        if avg_d_cut:
            drifts_cut[prompt_id] = avg_d_cut

        torch.cuda.empty_cache()

    return drifts_orig, drifts_cut


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_psnr_delta_histogram(results, out_path):
    """Histogram + KDE of PSNR deltas across prompts."""
    deltas = [r["psnr_delta"] for r in results if "psnr_delta" in r]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(deltas, bins=12, color="#3498db", edgecolor="white", alpha=0.7,
            density=True, label="Histogram")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(deltas)
    xs = np.linspace(min(deltas) - 0.5, max(deltas) + 0.5, 200)
    ax.plot(xs, kde(xs), color="#e74c3c", linewidth=2.5, label="KDE")

    mean_d = np.mean(deltas)
    ax.axvline(x=mean_d, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Mean Δ = {mean_d:+.2f} dB")
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1)

    # Annotate counts
    n_total = len(deltas)
    n_gt1 = sum(1 for d in deltas if d > 1.0)
    n_gt2 = sum(1 for d in deltas if d > 2.0)
    n_neg = sum(1 for d in deltas if d < 0)

    ax.set_xlabel("ΔPSNR (Cut A − Original) [dB]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"PSNR Improvement Distribution (N={n_total} prompts)\n"
                 f">1.0 dB: {n_gt1}/{n_total} | >2.0 dB: {n_gt2}/{n_total} | <0: {n_neg}/{n_total}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] PSNR delta histogram → {out_path}")


def plot_metric_bar(results, out_path):
    """Bar chart of per-prompt PSNR Original vs Cut A."""
    ids = [r["prompt_id"] for r in results]
    orig = [r["psnr_orig"] for r in results]
    cut = [r["psnr_cut"] for r in results]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width/2, orig, width, label="Original", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, cut, width, label="Cut A", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Per-Prompt PSNR: Original vs Cut A (skip → up_blocks.2 zeroed)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Per-prompt bar chart → {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results, drift_orig, drift_cut):
    """Print structured summary report."""
    psnr_deltas = [r["psnr_delta"] for r in results if "psnr_delta" in r]
    ssim_deltas = [r["ssim_delta"] for r in results if "ssim_delta" in r]
    lpips_deltas = [r["lpips_delta"] for r in results if "lpips_delta" in r]
    drift_deltas_pct = [r.get("drift_delta_pct", np.nan) for r in results
                         if not np.isnan(r.get("drift_delta_pct", np.nan))]

    print(f"\n{'='*70}")
    print("CROSS-PROMPT STATISTICAL VALIDATION — RESULTS")
    print(f"  N = {len(results)} prompts")
    print(f"{'='*70}")

    # Per-metric summary
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
        print(f"  One-sample t-test: t={t_stat:.3f}, p={p_val:.2e}")
        print(f"  Cohen's d = {d:.3f}")

    # PSNR breakdown
    print(f"\n--- PSNR Improvement Breakdown ---")
    n_total = len(psnr_deltas)
    for threshold in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        n = sum(1 for d in psnr_deltas if d > threshold)
        print(f"  >{threshold:+.1f} dB:  {n}/{n_total}  ({n/n_total*100:.0f}%)")

    n_neg = sum(1 for d in psnr_deltas if d < 0)
    print(f"  <0 dB (worse):  {n_neg}/{n_total}  ({n_neg/n_total*100:.0f}%)")

    # Drift
    if drift_deltas_pct:
        print(f"\n--- Peak Drift Change ---")
        print(f"  Mean Δdrift = {np.mean(drift_deltas_pct):+.1f}% ± {np.std(drift_deltas_pct):.1f}%")

    # Significant layers (cross-prompt t-test)
    if drift_orig and drift_cut:
        common = sorted(set(drift_orig.keys()) & set(drift_cut.keys()))
        if len(common) >= 3:
            ttest = paired_ttest_per_layer(drift_orig, drift_cut, common)
            n_sig = sum(1 for v in ttest.values() if v["significant"])
            print(f"\n--- Cross-Prompt Layer Significance ---")
            print(f"  Significant layers (p<0.05): {n_sig}/{len(ttest)}")

    # Outliers
    if n_neg > 0:
        print(f"\n--- Outliers (Cut A worse than Original) ---")
        for r in results:
            if r.get("psnr_delta", 0) < 0:
                print(f"  {r['prompt_id']}: ΔPSNR={r['psnr_delta']:+.2f} dB  "
                      f"({r['prompt'][:60]}...)")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ICLR Task A: Cross-Prompt Statistical Validation")
    parser.add_argument("--prompts", type=int, default=25,
                        help="Number of prompts to use (max 25)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip image generation (use existing)")
    parser.add_argument("--skip-drift", action="store_true",
                        help="Skip full drift analysis (faster)")
    parser.add_argument("--quick", type=int, default=None,
                        help="Quick test on N prompts")
    args = parser.parse_args()

    prompts = DIVERSE_PROMPTS[:args.prompts]
    if args.quick:
        prompts = prompts[:args.quick]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Setup] {len(prompts)} prompts, {args.steps} steps")
    print(f"[Output] {OUT_DIR.resolve()}")

    # Load
    print("[0] Loading SD 1.5 + LPIPS...")
    pipe = load_pipeline()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    # Phase 1: Generate images (or load from disk)
    generated = {}
    if not args.skip_generation:
        print(f"\n[1] Generating images from {len(prompts)} prompts...")
        for prompt_id, prompt in prompts:
            print(f"  [{prompt_id}] {prompt[:60]}...", end=" ", flush=True)
            img_tensor, latent = generate_and_save(pipe, prompt_id, prompt, args.steps)
            generated[prompt_id] = (img_tensor, latent)
            print("done")
    else:
        print("[1] Loading pre-generated images...")
        for prompt_id, prompt in prompts:
            img_path = OUT_DIR / "generated" / f"{prompt_id}.png"
            if img_path.exists():
                img_tensor, latent = load_and_encode(pipe, str(img_path))
                generated[prompt_id] = (img_tensor, latent)

    # Phase 2: Per-prompt evaluation
    print(f"\n[2] Evaluating Cut A vs Original on {len(generated)} prompts...")
    results = []
    for prompt_id, prompt in prompts:
        if prompt_id not in generated:
            continue
        img_tensor, latent = generated[prompt_id]
        print(f"  [{prompt_id}]...", end=" ", flush=True)
        r = evaluate_prompt(pipe, prompt_id, prompt, img_tensor, latent,
                           lpips_fn, args.steps)
        results.append(r)
        print(f"PSNR: {r['psnr_orig']:.2f} → {r['psnr_cut']:.2f} "
              f"(Δ={r.get('psnr_delta', 0):+.2f} dB)")

    # Phase 3: Full drift comparison (aggregate across prompts for t-test)
    drift_orig, drift_cut = {}, {}
    if not args.skip_drift:
        print(f"\n[3] Full per-layer drift comparison...")
        # Already computed in evaluate_prompt above
        # We need per-image drift dicts for cross-prompt t-test
        drifts_orig_all = {}
        drifts_cut_all = {}

        for r in results:
            pid = r["prompt_id"]
            if pid in generated:
                img_tensor, latent = generated[pid]
                prompt_embeds = pipe.encode_prompt(r["prompt"], DEVICE, 1, False)[0]

                avg_d, _ = analyze_layer_drift(pipe, latent, prompt_embeds, args.steps, seeds=[42])
                if avg_d:
                    drifts_orig_all[pid] = avg_d

                with SkipIntervention(pipe.unet, [2]):
                    avg_d_cut, _ = analyze_layer_drift(pipe, latent, prompt_embeds, args.steps, seeds=[42])
                if avg_d_cut:
                    drifts_cut_all[pid] = avg_d_cut

                torch.cuda.empty_cache()

        drift_orig = drifts_orig_all
        drift_cut = drifts_cut_all

        # Aggregate
        if drift_orig and drift_cut:
            agg_orig = aggregate_across_images(drift_orig)
            agg_cut = aggregate_across_images(drift_cut)

            common_imgs = sorted(set(drift_orig.keys()) & set(drift_cut.keys()))
            ttest = paired_ttest_per_layer(drift_orig, drift_cut, common_imgs)
            n_sig = sum(1 for v in ttest.values() if v["significant"])
            print(f"  Significant layers (p<0.05): {n_sig}/{len(ttest)}")

            # Count sig layers per result row
            for r in results:
                r["sig_layers"] = n_sig

    # Phase 4: Report
    print_report(results, drift_orig, drift_cut)

    # Phase 5: Figures
    print(f"\n[4] Generating figures...")
    plot_psnr_delta_histogram(results, OUT_DIR / "psnr_delta_histogram.png")
    plot_psnr_delta_histogram(results, OUT_DIR / "psnr_delta_histogram.pdf")
    plot_metric_bar(results, OUT_DIR / "psnr_per_prompt.png")
    plot_metric_bar(results, OUT_DIR / "psnr_per_prompt.pdf")

    # Phase 6: Save
    print(f"\n[5] Saving data...")
    # CSV
    csv_path = OUT_DIR / "results.csv"
    if results:
        fieldnames = ["prompt_id", "prompt",
                      "psnr_orig", "psnr_cut", "psnr_delta",
                      "ssim_orig", "ssim_cut", "ssim_delta",
                      "lpips_orig", "lpips_cut", "lpips_delta",
                      "peak_drift_orig", "peak_drift_cut",
                      "drift_delta", "drift_delta_pct", "sig_layers"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"[CSV] → {csv_path}")

    # Statistical summary JSON
    psnr_deltas = [r["psnr_delta"] for r in results if "psnr_delta" in r]
    ssim_deltas = [r["ssim_delta"] for r in results if "ssim_delta" in r]
    lpips_deltas = [r["lpips_delta"] for r in results if "lpips_delta" in r]

    summary = {
        "config": {"n_prompts": len(results), "steps": args.steps},
        "psnr": {
            "delta_mean": float(np.mean(psnr_deltas)),
            "delta_std": float(np.std(psnr_deltas)),
            "t_stat": float(stats.ttest_1samp(psnr_deltas, 0)[0]),
            "p_value": float(stats.ttest_1samp(psnr_deltas, 0)[1]),
            "cohens_d": float(np.mean(psnr_deltas) / np.std(psnr_deltas)) if np.std(psnr_deltas) > 0 else 0,
            "n_gt_1dB": int(sum(1 for d in psnr_deltas if d > 1.0)),
            "n_gt_2dB": int(sum(1 for d in psnr_deltas if d > 2.0)),
            "n_negative": int(sum(1 for d in psnr_deltas if d < 0)),
        },
        "ssim": {
            "delta_mean": float(np.mean(ssim_deltas)),
            "delta_std": float(np.std(ssim_deltas)),
        },
        "lpips": {
            "delta_mean": float(np.mean(lpips_deltas)),
            "delta_std": float(np.std(lpips_deltas)),
        },
    }

    # Bootstrap CI for PSNR delta
    if len(psnr_deltas) >= 10:
        boot_means = []
        rng = np.random.RandomState(42)
        for _ in range(10000):
            sample = rng.choice(psnr_deltas, size=len(psnr_deltas), replace=True)
            boot_means.append(np.mean(sample))
        summary["psnr"]["bootstrap_ci_95"] = [
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        ]

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Summary → {OUT_DIR / 'summary.json'}")

    print(f"\n{'='*60}")
    print("Task A complete.")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
