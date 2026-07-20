#!/usr/bin/env python
"""
Phase 7 expanded: 100-image editing benchmark.

Simplified from the full Phase 7 protocol:
  - No P2P (shown equivalent to ours, Cohen's d=0.033)
  - Only baseline vs ours (latent correction, λ=0.7)
  - Empty prompt inversion (consistent with Phase 2/5)
  - 1-2 edit types per image (style + word_swap if matchable)

Protocol:
  1. BLIP caption all 100 images
  2. Generate edit pairs: style transfer (always) + word swap (if applicable)
  3. DDIM invert with empty prompt + save latent trajectory
  4. Reconstruct with target (edited) prompt ± latent correction
  5. Compute LPIPS/PSNR/SSIM for content preservation

Usage:
    python scripts/phase7_editing_100image.py --mode all
    python scripts/phase7_editing_100image.py --mode caption
    python scripts/phase7_editing_100image.py --mode edit
    python scripts/phase7_editing_100image.py --mode eval

Output: outputs/phase7_editing_100image/
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_common import (
    load_pipeline, load_image, decode_latent, compute_metrics,
    ddim_inversion_with_latents, ddim_reconstruction_with_latent_correction,
    save_recon_img,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase7_editing_100image"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NUM_STEPS = 50
LAM = 0.7

# Word swap candidates
OBJECT_SWAPS = [
    ("dog", "cat"), ("cat", "dog"), ("car", "bus"), ("table", "desk"),
    ("chair", "couch"), ("person", "child"), ("bird", "butterfly"),
    ("tree", "bush"), ("cake", "pizza"), ("boat", "ship"),
    ("horse", "zebra"), ("phone", "camera"), ("cup", "bottle"),
    ("book", "magazine"), ("flower", "plant"),
]

STYLE_APPENDS = [
    ", oil painting style",
    ", watercolor painting",
    ", pencil sketch",
    ", cartoon illustration",
    ", professional studio photograph",
    ", van gogh style",
    ", charcoal drawing",
    ", vector art illustration",
]


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

def generate_captions(force=False):
    caption_file = CACHE_DIR / "captions_100.json"
    if caption_file.exists() and not force:
        with open(caption_file) as f:
            return json.load(f)

    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    img_paths = sorted(DATA_DIR.glob("coco_*.jpg"))
    captions = {}
    for img_path in tqdm(img_paths, desc="Captioning"):
        name = img_path.stem
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[name] = caption

    with open(caption_file, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions to {caption_file}")
    return captions


def generate_edit_pairs(captions):
    pairs_file = CACHE_DIR / "edit_pairs_100.json"
    if pairs_file.exists():
        with open(pairs_file) as f:
            return json.load(f)

    edit_triplets = {}
    for img_name, caption in captions.items():
        triplets = []

        # Style transfer (always)
        style = STYLE_APPENDS[hash(img_name) % len(STYLE_APPENDS)]
        triplets.append({
            "edit_type": "style",
            "source": caption,
            "target": caption.rstrip(".") + style,
            "swap": f"style:{style.strip()}",
        })

        # Word swap (if matchable)
        caption_lower = caption.lower()
        for src_word, tgt_word in OBJECT_SWAPS:
            pattern = re.compile(r'\b' + src_word + r'\b', re.IGNORECASE)
            if pattern.search(caption_lower):
                new_caption = pattern.sub(tgt_word, caption)
                triplets.append({
                    "edit_type": "word_swap",
                    "source": caption,
                    "target": new_caption,
                    "swap": f"{src_word}->{tgt_word}",
                })
                break

        edit_triplets[img_name] = triplets

    with open(pairs_file, "w") as f:
        json.dump(edit_triplets, f, indent=2)
    n_pairs = sum(len(v) for v in edit_triplets.values())
    print(f"Generated {n_pairs} edit pairs for {len(edit_triplets)} images")
    return edit_triplets


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def encode_prompt(pipe, prompt):
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompt, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(
            text_inputs.input_ids.to(pipe.device)
        )[0].to(pipe.unet.dtype)
    return text_embeddings


def run_editing(pipe, noise, saved_latents, tgt_embeds, num_steps, lam):
    """Reconstruct from inverted noise with target prompt + latent correction."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=pipe.device)
    timesteps = scheduler.timesteps
    z = noise.clone()

    with torch.no_grad():
        for t in timesteps:
            t_int = int(t)
            if saved_latents is not None and t_int in saved_latents:
                z = z + lam * (saved_latents[t_int].to(z.device) - z)
            noise_pred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


def run_sd15_editing(force=False):
    results_file = OUT_DIR / "editing_results.json"
    if results_file.exists() and not force:
        with open(results_file) as f:
            return json.load(f)

    captions = generate_captions()
    edit_pairs = generate_edit_pairs(captions)

    print("Loading SD 1.5...")
    pipe = load_pipeline()

    empty_embeds = encode_prompt(pipe, "")

    # Load LPIPS
    lpips_fn = lpips.LPIPS(net="alex").to("cuda")

    results = {"per_triplet": [], "summary": {}}
    incremental_path = OUT_DIR / "editing_results_partial.json"

    for img_name, triplets in tqdm(list(edit_pairs.items()), desc="Editing"):
        img_path = DATA_DIR / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        latent, orig_tensor = load_image(pipe, str(img_path))

        # Get source caption (same for all triplets of this image)
        caption = captions.get(img_name, "")
        src_embeds = encode_prompt(pipe, caption)

        # DDIM inversion with source prompt + save latent trajectory
        noise, saved_latents = ddim_inversion_with_latents(
            pipe, latent, src_embeds, NUM_STEPS)

        for triplet in triplets:
            target_prompt = triplet["target"]
            tgt_embeds = encode_prompt(pipe, target_prompt)

            # Baseline: no correction, reconstruct with target prompt
            recon_b = run_editing(pipe, noise, None, tgt_embeds, NUM_STEPS, LAM)
            recon_tensor_b = decode_latent(pipe, recon_b)
            m_b = compute_metrics(orig_tensor, recon_tensor_b, lpips_fn)

            # Ours: latent correction, reconstruct with target prompt
            recon_c = run_editing(pipe, noise, saved_latents, tgt_embeds,
                                  NUM_STEPS, LAM)
            recon_tensor_c = decode_latent(pipe, recon_c)
            m_c = compute_metrics(orig_tensor, recon_tensor_c, lpips_fn)

            entry = {
                "image": img_name,
                "edit_type": triplet["edit_type"],
                "target_prompt": target_prompt,
                "swap": triplet.get("swap", ""),
                "baseline_PSNR": m_b["PSNR"],
                "baseline_SSIM": m_b["SSIM"],
                "baseline_LPIPS": m_b["LPIPS"],
                "ours_PSNR": m_c["PSNR"],
                "ours_SSIM": m_c["SSIM"],
                "ours_LPIPS": m_c["LPIPS"],
                "delta_PSNR": m_c["PSNR"] - m_b["PSNR"],
                "delta_LPIPS": m_c["LPIPS"] - m_b["LPIPS"],
            }
            results["per_triplet"].append(entry)

            # Save first 20 examples
            idx = len(results["per_triplet"])
            if idx <= 20:
                examples_dir = OUT_DIR / "examples"
                examples_dir.mkdir(exist_ok=True)
                save_recon_img(recon_tensor_b, str(OUT_DIR / "examples"),
                             img_name, NUM_STEPS, f"{triplet['edit_type']}_baseline")
                save_recon_img(recon_tensor_c, str(OUT_DIR / "examples"),
                             img_name, NUM_STEPS, f"{triplet['edit_type']}_ours")
                if idx == 1:
                    save_recon_img(orig_tensor, str(OUT_DIR / "examples"),
                                 img_name, NUM_STEPS, "original")

        # Incremental save every 10 images
        if len(results["per_triplet"]) % 20 == 0:
            with open(incremental_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results['per_triplet'])} editing results to {results_file}")
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_editing(results):
    from scipy.stats import ttest_rel
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn

    eval_results = {"per_triplet": results["per_triplet"], "summary": {}}
    summary = {}

    for edit_type in ["style", "word_swap"]:
        for cond in ["baseline", "ours"]:
            lpips_vals = [e[f"{cond}_LPIPS"] for e in results["per_triplet"]
                          if e["edit_type"] == edit_type]
            psnr_vals = [e[f"{cond}_PSNR"] for e in results["per_triplet"]
                        if e["edit_type"] == edit_type]

            if lpips_vals:
                summary[f"{edit_type}_{cond}_LPIPS_mean"] = round(np.mean(lpips_vals), 4)
                summary[f"{edit_type}_{cond}_LPIPS_std"] = round(np.std(lpips_vals), 4)
                summary[f"{edit_type}_{cond}_PSNR_mean"] = round(np.mean(psnr_vals), 2)
                summary[f"{edit_type}_{cond}_PSNR_std"] = round(np.std(psnr_vals), 2)
                summary[f"{edit_type}_{cond}_n"] = len(lpips_vals)

    # Overall
    for cond in ["baseline", "ours"]:
        lpips_all = [e[f"{cond}_LPIPS"] for e in results["per_triplet"]]
        psnr_all = [e[f"{cond}_PSNR"] for e in results["per_triplet"]]
        summary[f"overall_{cond}_LPIPS"] = round(np.mean(lpips_all), 4)
        summary[f"overall_{cond}_LPIPS_std"] = round(np.std(lpips_all), 4)
        summary[f"overall_{cond}_PSNR"] = round(np.mean(psnr_all), 2)
        summary[f"overall_{cond}_PSNR_std"] = round(np.std(psnr_all), 2)

    # Statistical tests
    lpips_base = np.array([e["baseline_LPIPS"] for e in results["per_triplet"]])
    lpips_ours = np.array([e["ours_LPIPS"] for e in results["per_triplet"]])
    psnr_base = np.array([e["baseline_PSNR"] for e in results["per_triplet"]])
    psnr_ours = np.array([e["ours_PSNR"] for e in results["per_triplet"]])

    delta_lpips = lpips_ours - lpips_base
    delta_psnr = psnr_ours - psnr_base

    t_l, p_l = ttest_rel(lpips_ours, lpips_base)
    t_p, p_p = ttest_rel(psnr_ours, psnr_base)
    d_lpips = abs(delta_lpips.mean()) / delta_lpips.std(ddof=1) if delta_lpips.std() > 0 else 0
    d_psnr = delta_psnr.mean() / delta_psnr.std(ddof=1) if delta_psnr.std() > 0 else 0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(10000):
        idx = rng.choice(len(delta_lpips), len(delta_lpips), replace=True)
        boot_means.append(delta_lpips[idx].mean())
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    summary["delta_LPIPS_mean"] = round(float(delta_lpips.mean()), 4)
    summary["delta_LPIPS_std"] = round(float(delta_lpips.std()), 4)
    summary["delta_PSNR_mean"] = round(float(delta_psnr.mean()), 2)
    summary["delta_PSNR_std"] = round(float(delta_psnr.std()), 2)
    summary["LPIPS_ttest_p"] = float(p_l)
    summary["LPIPS_cohens_d"] = round(float(d_lpips), 3)
    summary["PSNR_ttest_p"] = float(p_p)
    summary["PSNR_cohens_d"] = round(float(d_psnr), 3)
    summary["LPIPS_bootstrap_ci95"] = [round(float(ci_lo), 4), round(float(ci_hi), 4)]
    summary["n_pairs"] = len(results["per_triplet"])
    summary["n_images"] = len(set(e["image"] for e in results["per_triplet"]))

    eval_results["summary"] = summary

    # Print
    print(f"\n{'='*80}")
    print(f"100-IMAGE EDITING BENCHMARK (n={summary['n_pairs']} pairs, "
          f"{summary['n_images']} images)")
    print(f"{'='*80}")
    print(f"{'Condition':<20s} {'LPIPS↓':>10s} {'PSNR↑':>8s}")
    print("-" * 40)
    for cond in ["baseline", "ours"]:
        print(f"{cond:<20s} {summary[f'overall_{cond}_LPIPS']:>10.4f} "
              f"{summary[f'overall_{cond}_PSNR']:>8.2f}")
    print(f"\nΔLPIPS: {summary['delta_LPIPS_mean']:.4f} (p={p_l:.2e}, d={d_lpips:.3f})")
    print(f"ΔPSNR: {summary['delta_PSNR_mean']:.2f} (p={p_p:.2e}, d={d_psnr:.3f})")
    print(f"LPIPS 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"\n{'Edit Type':<15s} {'Condition':<20s} {'LPIPS↓':>10s} {'PSNR↑':>8s}")
    print("-" * 55)
    for edit_type in ["style", "word_swap"]:
        for cond in ["baseline", "ours"]:
            lpips_key = f"{edit_type}_{cond}_LPIPS_mean"
            if lpips_key in summary:
                print(f"{edit_type:<15s} {cond:<20s} {summary[lpips_key]:>10.4f} "
                      f"{summary[f'{edit_type}_{cond}_PSNR_mean']:>8.2f}")

    eval_path = OUT_DIR / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved: {eval_path}")

    return eval_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7: 100-image Editing Benchmark")
    parser.add_argument("--mode", default="all",
                        choices=["caption", "edit", "eval", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", type=int, default=None,
                        help="Quick test on N images")
    args = parser.parse_args()

    if args.mode in ("caption", "all"):
        captions = generate_captions(force=args.force)
        edit_pairs = generate_edit_pairs(captions)
        n_pairs = sum(len(v) for v in edit_pairs.values())
        print(f"\nEdit pairs: {n_pairs} total across {len(edit_pairs)} images")
        for img, triplets in list(edit_pairs.items())[:5]:
            print(f"  {img}:")
            for t in triplets:
                print(f"    [{t['edit_type']}] {t['swap']}")
                print(f"      tgt: {t['target'][:80]}...")

    if args.mode in ("edit", "all"):
        results = run_sd15_editing(force=args.force)

    if args.mode in ("eval", "all"):
        results_file = OUT_DIR / "editing_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            evaluate_editing(results)
        else:
            print("No editing results found. Run --mode edit first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
