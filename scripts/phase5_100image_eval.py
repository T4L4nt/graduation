#!/usr/bin/env python
"""
SD 1.5 100-image expanded evaluation.

Expands Phase 5 from 19 to 100 COCO val2017 images.
Computes DDIM baseline + Ours_Corr + statistical tests.

Usage:
    python scripts/phase5_100image_eval.py
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import lpips
from tqdm import tqdm

from phase2_common import (
    load_pipeline, load_image, decode_latent,
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    FeatureCollector, FeatureCorrector,
    get_top_drift_layers, compute_metrics,
)

OUT_DIR = Path("outputs/phase5_100image")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = Path("data/coco_val100")
NUM_STEPS = 50
LAM = 0.7


def encode_empty_prompt(pipe):
    """Encode empty string to get null text embeddings."""
    text_input = pipe.tokenizer("", padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds = pipe.text_encoder(text_input.input_ids.to("cuda"))[0]
    return embeds


def run_baseline(pipe, latent, prompt_embeds):
    """DDIM baseline: invert + reconstruct, no correction."""
    noise = ddim_inversion(pipe, latent, prompt_embeds, NUM_STEPS)
    recon = ddim_reconstruction(pipe, noise, prompt_embeds, NUM_STEPS)
    return decode_latent(pipe, recon)


def run_ours(pipe, latent, top_layers, prompt_embeds):
    """Ours: invert with features + reconstruct with FeatureCorrector."""
    noise, saved_features = ddim_inversion_with_features(
        pipe, latent, prompt_embeds, NUM_STEPS, top_layers)
    # FeatureCorrector auto-registers hooks in __init__
    corrector = FeatureCorrector(pipe.unet, top_layers, LAM)
    recon = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, NUM_STEPS, saved_features, corrector)
    corrector.remove()
    return decode_latent(pipe, recon)


def main():
    print("Loading SD 1.5 pipeline...")
    pipe = load_pipeline()

    # Encode empty prompt for unconditional inversion
    prompt_embeds = encode_empty_prompt(pipe)
    print(f"Prompt embeds shape: {prompt_embeds.shape}")

    top_layers = get_top_drift_layers(k=5)
    print(f"Top-5 correction layers: {top_layers}")
    print(f"λ={LAM}, steps={NUM_STEPS}")

    # --- Gather images ---
    img_paths = sorted(IMAGE_DIR.glob("coco_*.jpg"))
    if len(img_paths) == 0:
        print(f"ERROR: No coco_*.jpg found in {IMAGE_DIR}")
        return
    print(f"Images: {len(img_paths)}")

    lpips_fn = lpips.LPIPS(net="alex").to("cuda")

    results_baseline = {}
    results_ours = {}

    for img_path in tqdm(img_paths, desc="Evaluating"):
        name = img_path.stem
        try:
            latent, orig_tensor = load_image(pipe, str(img_path))

            # DDIM baseline
            recon_tensor_b = run_baseline(pipe, latent, prompt_embeds)
            m_b = compute_metrics(orig_tensor, recon_tensor_b, lpips_fn)

            # Ours
            recon_tensor_c = run_ours(pipe, latent, top_layers, prompt_embeds)
            m_c = compute_metrics(orig_tensor, recon_tensor_c, lpips_fn)

            results_baseline[name] = m_b
            results_ours[name] = m_c

            # Save reconstructions for first 10 images
            idx = list(img_paths).index(img_path)
            if idx < 10:
                from phase2_common import save_recon_img
                save_recon_img(recon_tensor_b, str(OUT_DIR), name, NUM_STEPS,
                              "baseline")
                save_recon_img(recon_tensor_c, str(OUT_DIR), name, NUM_STEPS,
                              "ours")

        except Exception as e:
            print(f"  {name}: ERROR {e}")
            import traceback
            traceback.print_exc()

    n = len(results_baseline)
    if n == 0:
        print("No successful evaluations!")
        return

    # --- Aggregate ---
    psnr_b = np.array([results_baseline[k]["PSNR"] for k in results_baseline])
    psnr_c = np.array([results_ours[k]["PSNR"] for k in results_ours])
    ssim_b = np.array([results_baseline[k]["SSIM"] for k in results_baseline])
    ssim_c = np.array([results_ours[k]["SSIM"] for k in results_ours])
    lpips_b = np.array([results_baseline[k]["LPIPS"] for k in results_baseline])
    lpips_c = np.array([results_ours[k]["LPIPS"] for k in results_ours])

    delta_psnr = psnr_c - psnr_b
    delta_ssim = ssim_c - ssim_b
    delta_lpips = lpips_c - lpips_b

    # --- Statistical tests ---
    from scipy.stats import ttest_rel, wilcoxon

    t_p, p_p = ttest_rel(psnr_c, psnr_b)
    t_l, p_l = ttest_rel(lpips_c, lpips_b)
    cohens_d_psnr = delta_psnr.mean() / delta_psnr.std(ddof=1)
    cohens_d_lpips = delta_lpips.mean() / delta_lpips.std(ddof=1)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(delta_psnr[idx].mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Pearson r between baseline and ours (per-image)
    pearson_r = np.corrcoef(psnr_b, psnr_c)[0, 1]

    # --- Print ---
    print("\n" + "=" * 60)
    print(f"100-IMAGE EVALUATION (n={n})")
    print("=" * 60)
    print(f"\n  Baseline:    PSNR={psnr_b.mean():.2f}±{psnr_b.std():.2f}  "
          f"SSIM={ssim_b.mean():.4f}  LPIPS={lpips_b.mean():.4f}")
    print(f"  Ours (λ={LAM}): PSNR={psnr_c.mean():.2f}±{psnr_c.std():.2f}  "
          f"SSIM={ssim_c.mean():.4f}  LPIPS={lpips_c.mean():.4f}")
    print(f"  ΔPSNR: +{delta_psnr.mean():.2f} ± {delta_psnr.std():.2f}")
    print(f"  ΔSSIM: +{delta_ssim.mean():.4f}")
    print(f"  ΔLPIPS: {delta_lpips.mean():.4f}")
    print(f"  Paired t-test: t={t_p:.2f}, p={p_p:.2e}")
    print(f"  Cohen's d: {cohens_d_psnr:.3f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}] dB")
    print(f"  Pearson r: {pearson_r:.4f}")

    # --- Compare with 19-image ---
    print("\n" + "=" * 60)
    print("COMPARISON: 100-image vs 19-image")
    print("=" * 60)
    p5_path = Path("outputs/phase5_final/final_summary.json")
    if p5_path.exists():
        with open(p5_path) as f:
            p5 = json.load(f)
        p5_db = p5["ddim"]["psnr_mean"]
        p5_dc = p5["ours"]["psnr_mean"]
        p5_delta = p5_dc - p5_db
        print(f"  19-image:  baseline={p5_db:.2f}  ours={p5_dc:.2f}  Δ=+{p5_delta:.2f}")
        print(f"  100-image: baseline={psnr_b.mean():.2f}  ours={psnr_c.mean():.2f}  Δ=+{delta_psnr.mean():.2f}")
        print(f"  Consistency: {'OK ✓' if abs(delta_psnr.mean() - p5_delta) < 1.0 else 'DISCREPANCY'}")

    # --- Save ---
    output = {
        "config": {"n": n, "lam": LAM, "steps": NUM_STEPS, "top_layers": top_layers},
        "summary": {
            "baseline_PSNR": f"{psnr_b.mean():.2f}±{psnr_b.std():.2f}",
            "ours_PSNR": f"{psnr_c.mean():.2f}±{psnr_c.std():.2f}",
            "delta_PSNR": float(delta_psnr.mean()),
            "delta_PSNR_std": float(delta_psnr.std()),
            "baseline_LPIPS": float(lpips_b.mean()),
            "ours_LPIPS": float(lpips_c.mean()),
            "cohens_d": float(cohens_d_psnr),
            "paired_t_p": float(p_p),
            "bootstrap_CI95": [float(ci_lo), float(ci_hi)],
            "pearson_r": float(pearson_r),
        },
        "per_image": {
            k: {"baseline": results_baseline[k], "ours": results_ours[k]}
            for k in results_baseline
        },
    }
    out_path = OUT_DIR / "results_100image.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
