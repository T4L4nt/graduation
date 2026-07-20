#!/usr/bin/env python
"""
Phase 5 100-image evaluation recovery script.

Re-runs the full 100-image eval with incremental checkpoint saving.
Each image's metrics are saved immediately to prevent data loss on interruption.
Resumes from partial results if available.

Usage:
    python scripts/phase5_100image_recover.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import lpips
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_common import (
    load_pipeline, load_image, decode_latent,
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    FeatureCollector, FeatureCorrector,
    get_top_drift_layers, compute_metrics,
    save_recon_img,
)

OUT_DIR = Path("outputs/phase5_100image")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = Path("data/coco_val100")
NUM_STEPS = 50
LAM = 0.7

CHECKPOINT_PATH = OUT_DIR / "results_checkpoint.json"
FINAL_PATH = OUT_DIR / "results_100image.json"


def encode_empty_prompt(pipe):
    text_input = pipe.tokenizer("", padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeds = pipe.text_encoder(text_input.input_ids.to("cuda"))[0]
    return embeds


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"baseline": {}, "ours": {}}


def save_checkpoint(baseline, ours):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"baseline": baseline, "ours": ours}, f, indent=2)


def main():
    print("Loading SD 1.5 pipeline...")
    pipe = load_pipeline()
    prompt_embeds = encode_empty_prompt(pipe)

    top_layers = get_top_drift_layers(k=5)
    print(f"Top-5 correction layers: {top_layers}")
    print(f"λ={LAM}, steps={NUM_STEPS}")

    # Gather images
    img_paths = sorted(IMAGE_DIR.glob("coco_*.jpg"))
    if len(img_paths) == 0:
        print(f"ERROR: No coco_*.jpg found in {IMAGE_DIR}")
        return
    print(f"Images: {len(img_paths)}")

    # Load checkpoint
    ckpt = load_checkpoint()
    results_baseline = ckpt["baseline"]
    results_ours = ckpt["ours"]
    done = set(results_baseline.keys())

    if done:
        print(f"Resuming: {len(done)} images already computed, "
              f"{len(img_paths) - len(done)} remaining")

    lpips_fn = lpips.LPIPS(net="alex").to("cuda")
    save_count = 0

    for img_path in tqdm(img_paths, desc="Evaluating"):
        name = img_path.stem
        if name in done:
            continue

        try:
            latent, orig_tensor = load_image(pipe, str(img_path))

            # DDIM baseline
            noise = ddim_inversion(pipe, latent, prompt_embeds, NUM_STEPS)
            recon = ddim_reconstruction(pipe, noise, prompt_embeds, NUM_STEPS)
            recon_tensor_b = decode_latent(pipe, recon)
            m_b = compute_metrics(orig_tensor, recon_tensor_b, lpips_fn)

            # Ours
            noise, saved_features = ddim_inversion_with_features(
                pipe, latent, prompt_embeds, NUM_STEPS, top_layers)
            corrector = FeatureCorrector(pipe.unet, top_layers, LAM)
            recon = ddim_reconstruction_with_correction(
                pipe, noise, prompt_embeds, NUM_STEPS, saved_features, corrector)
            corrector.remove()
            recon_tensor_c = decode_latent(pipe, recon)
            m_c = compute_metrics(orig_tensor, recon_tensor_c, lpips_fn)

            results_baseline[name] = m_b
            results_ours[name] = m_c

            # Save first 10 reconstructions
            idx = list(img_paths).index(img_path)
            if idx < 10:
                (OUT_DIR / "recons").mkdir(exist_ok=True)
                save_recon_img(recon_tensor_b, str(OUT_DIR), name, NUM_STEPS,
                             "baseline")
                save_recon_img(recon_tensor_c, str(OUT_DIR), name, NUM_STEPS,
                             "ours")

            save_count += 1
            if save_count % 5 == 0:
                save_checkpoint(results_baseline, results_ours)

        except Exception as e:
            print(f"\n  {name}: ERROR {e}")
            import traceback
            traceback.print_exc()
            save_checkpoint(results_baseline, results_ours)

    # Final save
    save_checkpoint(results_baseline, results_ours)

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

    from scipy.stats import ttest_rel

    t_p, p_p = ttest_rel(psnr_c, psnr_b)
    cohens_d_psnr = delta_psnr.mean() / delta_psnr.std(ddof=1)
    cohens_d_lpips = delta_lpips.mean() / delta_lpips.std(ddof=1)

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(delta_psnr[idx].mean())
    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    pearson_r = np.corrcoef(psnr_b, psnr_c)[0, 1]

    print("\n" + "=" * 60)
    print(f"100-IMAGE EVALUATION (n={n})")
    print("=" * 60)
    print(f"  Baseline:  PSNR={psnr_b.mean():.2f}±{psnr_b.std():.2f}  "
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

    # Compare with 19-image
    p5_path = Path("outputs/phase5_final/final_summary.json")
    if p5_path.exists():
        with open(p5_path) as f:
            p5 = json.load(f)
        p5_db = p5["summaries"]["DDIM"]["PSNR"]["mean"]
        p5_dc = p5["summaries"]["Ours_Corr"]["PSNR"]["mean"]
        p5_delta = p5_dc - p5_db
        print(f"\n  19-image:  baseline={p5_db:.2f}  ours={p5_dc:.2f}  Δ=+{p5_delta:.2f}")
        print(f"  100-image: baseline={psnr_b.mean():.2f}  ours={psnr_c.mean():.2f}  Δ=+{delta_psnr.mean():.2f}")
        print(f"  Consistency: {'OK' if abs(delta_psnr.mean() - p5_delta) < 1.0 else 'DISCREPANCY'}")

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
    with open(FINAL_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {FINAL_PATH}")


if __name__ == "__main__":
    main()
