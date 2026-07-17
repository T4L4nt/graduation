#!/usr/bin/env python
"""
Phase 7 CLIP-Dir addendum: re-run editing with CLIP directional similarity.

Computes CLIP-Dir = cos(CLIP_img(edited) - CLIP_img(original),
                         CLIP_txt(target) - CLIP_txt(source))
for both baseline and ours across all 121 edit pairs.

Produces the dual-metric table: content preservation (LPIPS) x edit fidelity (CLIP-Dir).

Usage:
    python scripts/phase7_clip_dir.py

Output: outputs/phase7_editing_100image/evaluation_with_clipdir.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_common import (
    load_pipeline, load_image, decode_latent, compute_metrics,
    ddim_inversion_with_latents, ddim_reconstruction_with_latent_correction,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase7_editing_100image"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
CACHE_DIR = OUT_DIR / "cache"

NUM_STEPS = 50
LAM = 0.7
DEVICE = "cuda"


def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                       local_files_only=True).to(DEVICE)
    model.eval()
    from transformers import CLIPImageProcessor, AutoTokenizer
    img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                   local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                               local_files_only=True)
    return model, img_proc, tokenizer


@torch.no_grad()
def clip_image_embed(clip_model, img_proc, tensor):
    """tensor: [1,3,H,W] in [-1,1]"""
    img = ((tensor.squeeze(0) + 1) / 2).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    inputs = img_proc(images=Image.fromarray(img), return_tensors="pt").to(DEVICE)
    out = clip_model.get_image_features(**inputs)
    emb = out.pooler_output if hasattr(out, 'pooler_output') else out
    if hasattr(emb, 'image_embeds'):
        emb = emb.image_embeds
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


@torch.no_grad()
def clip_text_embed(clip_model, tokenizer, text):
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    out = clip_model.get_text_features(**inputs)
    emb = out.pooler_output if hasattr(out, 'pooler_output') else out
    if hasattr(emb, 'text_embeds'):
        emb = emb.text_embeds
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def run_editing(pipe, noise, saved_latents, tgt_embeds, num_steps, lam):
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


def main():
    # Load caches
    with open(CACHE_DIR / "captions_100.json") as f:
        captions = json.load(f)
    with open(CACHE_DIR / "edit_pairs_100.json") as f:
        edit_pairs = json.load(f)

    # Flatten triplets
    triplets = []
    for img_name, triplet_list in edit_pairs.items():
        for t in triplet_list:
            triplets.append({**t, "image": img_name})

    print(f"Total triplets: {len(triplets)}")

    # Load models
    print("Loading SD 1.5...")
    pipe = load_pipeline()
    print("Loading CLIP ViT-L/14...")
    clip_model, clip_img_proc, clip_tokenizer = load_clip()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    empty_embeds = encode_prompt(pipe, "")

    results = []
    incremental_path = OUT_DIR / "clipdir_results_partial.json"

    for img_name, triplet_list in tqdm(list(edit_pairs.items()), desc="Editing+CLIP"):
        img_path = DATA_DIR / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        latent, orig_tensor = load_image(pipe, str(img_path))
        clip_orig = clip_image_embed(clip_model, clip_img_proc, orig_tensor)

        caption = captions.get(img_name, "")
        src_embeds = encode_prompt(pipe, caption)

        # DDIM inversion with source prompt
        noise, saved_latents = ddim_inversion_with_latents(
            pipe, latent, src_embeds, NUM_STEPS)

        for triplet in triplet_list:
            target_prompt = triplet["target"]
            tgt_embeds = encode_prompt(pipe, target_prompt)

            # CLIP text direction
            clip_src_txt = clip_text_embed(clip_model, clip_tokenizer, caption)
            clip_tgt_txt = clip_text_embed(clip_model, clip_tokenizer, target_prompt)
            text_dir = clip_tgt_txt - clip_src_txt
            text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True)

            # Baseline: no correction
            recon_b = run_editing(pipe, noise, None, tgt_embeds, NUM_STEPS, LAM)
            tensor_b = decode_latent(pipe, recon_b)
            m_b = compute_metrics(orig_tensor, tensor_b, lpips_fn)
            clip_b = clip_image_embed(clip_model, clip_img_proc, tensor_b)
            img_dir_b = clip_b - clip_orig
            img_dir_b = img_dir_b / img_dir_b.norm(dim=-1, keepdim=True)
            clipdir_b = float((img_dir_b * text_dir).sum().item())

            # Ours: latent correction
            recon_c = run_editing(pipe, noise, saved_latents, tgt_embeds,
                                  NUM_STEPS, LAM)
            tensor_c = decode_latent(pipe, recon_c)
            m_c = compute_metrics(orig_tensor, tensor_c, lpips_fn)
            clip_c = clip_image_embed(clip_model, clip_img_proc, tensor_c)
            img_dir_c = clip_c - clip_orig
            img_dir_c = img_dir_c / img_dir_c.norm(dim=-1, keepdim=True)
            clipdir_c = float((img_dir_c * text_dir).sum().item())

            results.append({
                "image": img_name,
                "edit_type": triplet["edit_type"],
                "target_prompt": target_prompt,
                "swap": triplet.get("swap", ""),
                "baseline_PSNR": m_b["PSNR"],
                "baseline_SSIM": m_b["SSIM"],
                "baseline_LPIPS": m_b["LPIPS"],
                "baseline_CLIPDir": clipdir_b,
                "ours_PSNR": m_c["PSNR"],
                "ours_SSIM": m_c["SSIM"],
                "ours_LPIPS": m_c["LPIPS"],
                "ours_CLIPDir": clipdir_c,
                "delta_PSNR": m_c["PSNR"] - m_b["PSNR"],
                "delta_LPIPS": m_c["LPIPS"] - m_b["LPIPS"],
                "delta_CLIPDir": clipdir_c - clipdir_b,
                "text_direction_norm": float(text_dir.norm().item()),
            })

        if len(results) % 20 == 0:
            with open(incremental_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save full results
    results_path = OUT_DIR / "clipdir_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {results_path}")

    # ---- Summary ----
    from scipy.stats import ttest_rel

    clipdir_b = np.array([r["baseline_CLIPDir"] for r in results])
    clipdir_c = np.array([r["ours_CLIPDir"] for r in results])
    lpips_b = np.array([r["baseline_LPIPS"] for r in results])
    lpips_c = np.array([r["ours_LPIPS"] for r in results])
    psnr_b = np.array([r["baseline_PSNR"] for r in results])
    psnr_c = np.array([r["ours_PSNR"] for r in results])

    summary = {"n_pairs": len(results), "n_images": len(set(r["image"] for r in results))}

    for label, vals_b, vals_o in [("LPIPS", lpips_b, lpips_c),
                                   ("PSNR", psnr_b, psnr_c),
                                   ("CLIPDir", clipdir_b, clipdir_c)]:
        t, p = ttest_rel(vals_o, vals_b)
        d = float(abs(vals_o - vals_b).mean()) / float((vals_o - vals_b).std(ddof=1)) \
            if (vals_o - vals_b).std() > 0 else 0
        summary[f"baseline_{label}"] = round(float(vals_b.mean()), 4)
        summary[f"ours_{label}"] = round(float(vals_o.mean()), 4)
        summary[f"delta_{label}"] = round(float((vals_o - vals_b).mean()), 4)
        summary[f"{label}_p"] = float(p)
        summary[f"{label}_d"] = round(float(d), 3)

    # Per edit type
    for edit_type in ["style", "word_swap"]:
        idx = [r["edit_type"] == edit_type for r in results]
        for label, vals_b, vals_o in [("LPIPS", lpips_b, lpips_c),
                                       ("PSNR", psnr_b, psnr_c),
                                       ("CLIPDir", clipdir_b, clipdir_c)]:
            b_mean = float(vals_b[idx].mean())
            o_mean = float(vals_o[idx].mean())
            summary[f"{edit_type}_baseline_{label}"] = round(b_mean, 4)
            summary[f"{edit_type}_ours_{label}"] = round(o_mean, 4)
            summary[f"{edit_type}_delta_{label}"] = round(float(o_mean - b_mean), 4)

    eval_path = OUT_DIR / "evaluation_with_clipdir.json"
    with open(eval_path, "w") as f:
        json.dump({"per_triplet": results, "summary": summary}, f, indent=2)

    # ---- Dual-metric table ----
    print(f"\n{'='*80}")
    print(f"CONTENT × EDIT FIDELITY — 100-IMAGE BENCHMARK")
    print(f"({summary['n_pairs']} pairs, {summary['n_images']} images)")
    print(f"{'='*80}")
    print(f"{'Condition':<20s} {'LPIPS↓':>10s} {'PSNR↑':>8s} {'CLIP-Dir↑':>10s} {'ΔCLIP-Dir':>10s}")
    print("-" * 60)
    for cond, lp_key, ps_key, cd_key in [
        ("baseline", "baseline_LPIPS", "baseline_PSNR", "baseline_CLIPDir"),
        ("ours", "ours_LPIPS", "ours_PSNR", "ours_CLIPDir"),
    ]:
        print(f"{cond:<20s} {summary[lp_key]:>10.4f} {summary[ps_key]:>8.2f} "
              f"{summary[cd_key]:>10.4f} {'—':>10s}")
    print(f"{'Δ (ours - baseline)':<20s} {summary['delta_LPIPS']:>10.4f} "
          f"{summary['delta_PSNR']:>8.2f} {summary['delta_CLIPDir']:>10.4f} "
          f"{summary['delta_CLIPDir']:>10.4f}")
    print(f"\np(LPIPS)={summary['LPIPS_p']:.2e}  p(CLIP-Dir)={summary['CLIPDir_p']:.2e}")
    print(f"d(LPIPS)={summary['LPIPS_d']:.3f}  d(CLIP-Dir)={summary['CLIPDir_d']:.3f}")

    # Per edit type
    print(f"\n{'Edit Type':<15s} {'Condition':<20s} {'LPIPS↓':>10s} {'CLIP-Dir↑':>10s}")
    print("-" * 57)
    for et in ["style", "word_swap"]:
        for cond, lp_key, cd_key in [
            ("baseline", f"{et}_baseline_LPIPS", f"{et}_baseline_CLIPDir"),
            ("ours", f"{et}_ours_LPIPS", f"{et}_ours_CLIPDir"),
        ]:
            if lp_key in summary:
                print(f"{et:<15s} {cond:<20s} {summary[lp_key]:>10.4f} {summary[cd_key]:>10.4f}")

    print(f"\nSaved: {eval_path}")
    print("Done.")


if __name__ == "__main__":
    main()
