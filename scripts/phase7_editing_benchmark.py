#!/usr/bin/env python
"""
Phase 7: Editing benchmark — correction as a plugin for content-preserving editing.

Evaluates whether our residual correction improves content preservation when
the text prompt is changed during reconstruction (editing).

Protocol:
  1. BLIP captions each coco_val image → source prompt
  2. Generate 2-3 edited variants per image (word swap, attribute, style)
  3. For each (image, source_prompt, target_prompt) triplet:
     a. DDIM invert with source prompt
     b. Reconstruct with target prompt:
        - Baseline (no correction, no P2P)
        - Ours (latent correction only)
        - P2P (cross-attention injection only)
        - Ours + P2P (both)
  4. Metrics: LPIPS(orig, edited), CLIP_score(edited, target_prompt)

Usage:
    python scripts/phase7_editing_benchmark.py --mode caption   # generate captions
    python scripts/phase7_editing_benchmark.py --mode edit      # run editing (GPU)
    python scripts/phase7_editing_benchmark.py --mode eval      # evaluate (CPU)
    python scripts/phase7_editing_benchmark.py --mode all       # everything
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase7_editing"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COCO_19 = sorted([
    "coco_000000000139", "coco_000000000285", "coco_000000000632",
    "coco_000000000724", "coco_000000000776", "coco_000000000785",
    "coco_000000000802", "coco_000000000872", "coco_000000000885",
    "coco_000000001000", "coco_000000001353", "coco_000000001490",
    "coco_000000001532", "coco_000000001584", "coco_000000001675",
    "coco_000000001818", "coco_000000002153", "coco_000000002261",
    "coco_000000002532",
])

# Editing templates: function(source_caption) -> [(edit_type, target_prompt)]
# Generated programmatically from BLIP captions
OBJECT_SWAPS = [
    ("dog", "cat"), ("cat", "dog"), ("car", "bus"), ("table", "desk"),
    ("chair", "couch"), ("person", "child"), ("bird", "butterfly"),
    ("tree", "bush"), ("cake", "pizza"), ("boat", "ship"),
    ("horse", "zebra"), ("phone", "camera"), ("cup", "bottle"),
    ("book", "magazine"), ("flower", "plant"),
]

COLOR_SWAPS = [
    ("red", "blue"), ("white", "black"), ("green", "yellow"),
    ("brown", "gray"), ("pink", "purple"), ("orange", "red"),
]

STYLE_APPENDS = [
    ", oil painting style",
    ", watercolor painting",
    ", pencil sketch",
    ", cartoon illustration",
    ", professional studio photograph",
]


# ---------------------------------------------------------------------------
# Phase 7a: Captioning
# ---------------------------------------------------------------------------

def generate_captions(force=False):
    """Use BLIP to caption each coco_val image. Cache results."""
    caption_file = CACHE_DIR / "captions.json"
    if caption_file.exists() and not force:
        with open(caption_file) as f:
            return json.load(f)

    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    captions = {}
    for img_name in tqdm(COCO_19, desc="Captioning"):
        img_path = DATA_DIR / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = DATA_DIR / f"{img_name}.png"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[img_name] = caption

    with open(caption_file, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions to {caption_file}")
    return captions


def generate_edit_pairs(captions):
    """Generate (source_prompt, edit_type, target_prompt) triplets."""
    pairs_file = CACHE_DIR / "edit_pairs.json"
    if pairs_file.exists():
        with open(pairs_file) as f:
            return json.load(f)

    import re

    edit_triplets = {}  # img_name -> [{edit_type, source, target}]

    for img_name, caption in captions.items():
        triplets = []

        # 1. Word swap: find a matchable object and swap it
        swapped = False
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
                swapped = True
                break

        # 2. Color/attribute change
        for src_color, tgt_color in COLOR_SWAPS:
            pattern = re.compile(r'\b' + src_color + r'\b', re.IGNORECASE)
            if pattern.search(caption_lower):
                new_caption = pattern.sub(tgt_color, caption)
                triplets.append({
                    "edit_type": "attribute",
                    "source": caption,
                    "target": new_caption,
                    "swap": f"{src_color}->{tgt_color}",
                })
                break

        # 3. Style transfer (append style modifier)
        style = STYLE_APPENDS[hash(img_name) % len(STYLE_APPENDS)]
        triplets.append({
            "edit_type": "style",
            "source": caption,
            "target": caption.rstrip(".") + style,
            "swap": f"style:{style.strip()}",
        })

        edit_triplets[img_name] = triplets

    with open(pairs_file, "w") as f:
        json.dump(edit_triplets, f, indent=2)
    print(f"Generated {sum(len(v) for v in edit_triplets.values())} edit pairs "
          f"for {len(edit_triplets)} images")
    return edit_triplets


# ---------------------------------------------------------------------------
# Phase 7b: SD 1.5 editing
# ---------------------------------------------------------------------------

def encode_prompt(pipe, prompt):
    """Encode a text prompt for SD pipeline."""
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(
            text_inputs.input_ids.to(pipe.device)
        )[0].to(pipe.unet.dtype)
    return text_embeddings


def get_cross_attention_modules(unet):
    """Find all cross-attention modules in UNet."""
    attn_modules = []
    for name, module in unet.named_modules():
        if "attn2" in name and hasattr(module, "to_k"):
            attn_modules.append((name, module))
    return attn_modules


def ddim_invert_with_attn(pipe, latents, prompt_embeds, num_steps=50):
    """DDIM inversion with cross-attention map saving.

    Returns:
        noise: terminal latent
        saved_attn_maps: {timestep: {module_name: attn_tensor}}
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=pipe.device)
    timesteps = scheduler.timesteps
    extended_ts = timesteps.tolist() + [0]

    # Register hooks on cross-attention modules
    attn_modules = get_cross_attention_modules(pipe.unet)
    saved_attn_maps = {}
    hooks = []

    def make_hook(name, store_dict):
        def hook(module, input, output):
            store_dict[name] = output.detach().cpu()
        return hook

    z = latents.clone().to(pipe.unet.dtype)
    for i, t in enumerate(tqdm(timesteps, desc="Invert+Attn")):
        t_prev = extended_ts[i + 1]
        alpha_t = scheduler.alphas_cumprod[t]
        alpha_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        coeff1 = (alpha_t / alpha_prev).sqrt()
        coeff2 = (1 - alpha_prev).sqrt() - (alpha_t * (1 - alpha_prev) / alpha_prev).sqrt()

        # Register hooks for this step
        step_store = {}
        for name, module in attn_modules:
            h = module.register_forward_hook(make_hook(name, step_store))
            hooks.append(h)

        t_tensor = torch.full((1,), t, device=pipe.device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = pipe.unet(z, t_tensor, encoder_hidden_states=prompt_embeds).sample

        for h in hooks:
            h.remove()
        hooks.clear()

        saved_attn_maps[int(t)] = dict(step_store)
        z = coeff1 * z + coeff2 * noise_pred

    return z, saved_attn_maps


def ddim_reconstruct_with_editing(
    pipe, noise, source_embeds, target_embeds, num_steps=50,
    saved_attn_maps=None, attn_lam=0.8,
    latent_correction=None, corr_lam=0.7,
):
    """DDIM reconstruction with prompt editing.

    Args:
        pipe: SD pipeline
        noise: terminal latent from inversion
        source_embeds: prompt embeddings used during inversion
        target_embeds: prompt embeddings for reconstruction (edited)
        saved_attn_maps: cross-attention maps from inversion (for P2P)
        attn_lam: P2P attention injection strength
        latent_correction: saved latents from inversion (for our correction)
        corr_lam: latent correction strength

    Returns:
        PIL Image of edited reconstruction
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=pipe.device)
    timesteps = scheduler.timesteps

    attn_modules = get_cross_attention_modules(pipe.unet)
    hooks = []
    z = noise.clone().to(pipe.unet.dtype)

    # For P2P: register hooks only once, update reference per step
    if saved_attn_maps is not None:
        attn_stores = {}  # module_name -> reference tensor for current timestep

        def make_p2p_hook(name):
            def hook(module, input, output):
                if name in attn_stores:
                    ref = attn_stores[name].to(output.device, dtype=output.dtype)
                    return output + attn_lam * (ref - output)
                return output
            return hook

        for name, module in attn_modules:
            h = module.register_forward_hook(make_p2p_hook(name))
            hooks.append(h)

    for i, t in enumerate(tqdm(timesteps, desc="Recon+Edit")):
        t_int = int(t)

        # Update P2P reference for this timestep
        if saved_attn_maps is not None and t_int in saved_attn_maps:
            attn_stores = saved_attn_maps[t_int]

        # Apply latent correction
        if latent_correction is not None and t_int in latent_correction:
            z = z + corr_lam * (latent_correction[t_int].to(z.device) - z)

        t_tensor = torch.full((1,), t, device=pipe.device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = pipe.unet(z, t_tensor, encoder_hidden_states=target_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    for h in hooks:
        h.remove()

    # Decode
    with torch.no_grad():
        z = z / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(z, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))

    return image


def run_sd15_editing(force=False):
    """Run editing benchmark on SD 1.5."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    results_file = OUT_DIR / "sd15_editing_results.json"
    if results_file.exists() and not force:
        with open(results_file) as f:
            return json.load(f)

    # Load captions and edit pairs
    captions = generate_captions()
    edit_pairs = generate_edit_pairs(captions)

    # Load SD 1.5
    print("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()

    results = {"per_triplet": [], "summary": {}}
    NUM_STEPS = 50

    for img_name, triplets in tqdm(edit_pairs.items(), desc="Editing SD1.5"):
        img_path = DATA_DIR / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = DATA_DIR / f"{img_name}.png"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))

        # Encode image to latent
        with torch.no_grad():
            img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(pipe.device, dtype=pipe.vae.dtype)
            latents = pipe.vae.encode(img_tensor * 2 - 1).latent_dist.sample()
            latents = (latents * pipe.vae.config.scaling_factor).to(pipe.device, dtype=pipe.unet.dtype)

        for triplet in triplets:
            source_prompt = triplet["source"]
            target_prompt = triplet["target"]
            edit_type = triplet["edit_type"]

            source_embeds = encode_prompt(pipe, source_prompt)
            target_embeds = encode_prompt(pipe, target_prompt)

            # Inversion with attention saving
            noise, saved_attn = ddim_invert_with_attn(
                pipe, latents, source_embeds, num_steps=NUM_STEPS
            )

            # Also run standard inversion for latent correction baseline
            pipe.scheduler.set_timesteps(NUM_STEPS, device=pipe.device)
            timesteps = pipe.scheduler.timesteps
            extended_ts = timesteps.tolist() + [0]
            saved_latents = {}
            z = latents.clone().to(pipe.unet.dtype)
            for i, t in enumerate(timesteps):
                saved_latents[int(t)] = z.clone().cpu()
                t_prev = extended_ts[i + 1]
                alpha_t = pipe.scheduler.alphas_cumprod[t]
                alpha_prev = pipe.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
                coeff1 = (alpha_t / alpha_prev).sqrt()
                coeff2 = (1 - alpha_prev).sqrt() - (alpha_t * (1 - alpha_prev) / alpha_prev).sqrt()
                t_tensor = torch.full((1,), t, device=pipe.device, dtype=torch.long)
                with torch.no_grad():
                    noise_pred = pipe.unet(z, t_tensor, encoder_hidden_states=source_embeds).sample
                z = coeff1 * z + coeff2 * noise_pred

            # Reconstruct with each condition
            conditions = {
                "baseline": {"saved_attn_maps": None, "latent_correction": None},
                "ours": {"saved_attn_maps": None, "latent_correction": saved_latents},
                "p2p": {"saved_attn_maps": saved_attn, "latent_correction": None},
                "ours_p2p": {"saved_attn_maps": saved_attn, "latent_correction": saved_latents},
            }

            entry = {
                "image": img_name,
                "edit_type": edit_type,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "swap": triplet.get("swap", ""),
            }

            for cond_name, kwargs in conditions.items():
                edited_img = ddim_reconstruct_with_editing(
                    pipe, noise.clone(), source_embeds, target_embeds,
                    num_steps=NUM_STEPS, **kwargs,
                )
                # Save first few examples
                idx = len(results["per_triplet"])
                if idx < 20:
                    save_dir = OUT_DIR / "examples"
                    save_dir.mkdir(exist_ok=True)
                    edited_img.save(
                        save_dir / f"{img_name}_{edit_type}_{cond_name}.png"
                    )
                    if cond_name == "baseline":
                        image.save(save_dir / f"{img_name}_original.png")

                entry[f"{cond_name}_image"] = str(
                    OUT_DIR / "examples" / f"{img_name}_{edit_type}_{cond_name}.png"
                )

            results["per_triplet"].append(entry)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results['per_triplet'])} editing results to {results_file}")
    return results


# ---------------------------------------------------------------------------
# Phase 7c: Evaluation
# ---------------------------------------------------------------------------

def evaluate_editing(results):
    """Compute content preservation and edit quality metrics."""
    import lpips

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    eval_results = {"per_triplet": [], "summary": {}}

    for entry in tqdm(results["per_triplet"], desc="Evaluating"):
        img_name = entry["image"]
        edit_type = entry["edit_type"]
        target_prompt = entry["target_prompt"]

        # Load original image
        orig_path = DATA_DIR / f"{img_name}.jpg"
        if not orig_path.exists():
            orig_path = DATA_DIR / f"{img_name}.png"
        if not orig_path.exists():
            continue
        original = Image.open(orig_path).convert("RGB").resize((512, 512))

        eval_entry = {k: entry[k] for k in ["image", "edit_type", "source_prompt",
                                              "target_prompt", "swap"]}

        for cond in ["baseline", "ours", "p2p", "ours_p2p"]:
            edited_path = entry.get(f"{cond}_image", "")
            if not edited_path or not Path(edited_path).exists():
                continue

            edited = Image.open(edited_path).convert("RGB")

            # LPIPS: content preservation (lower = better preserved)
            orig_t = torch.from_numpy(np.array(original)).float() / 255.0
            orig_t = orig_t.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
            edit_t = torch.from_numpy(np.array(edited)).float() / 255.0
            edit_t = edit_t.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1

            lpips_val = float(lpips_fn(orig_t, edit_t).item())
            eval_entry[f"{cond}_LPIPS"] = round(lpips_val, 4)

            # PSNR
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            orig_np = np.array(original).astype(np.float32) / 255.0
            edit_np = np.array(edited).astype(np.float32) / 255.0
            psnr_val = psnr_fn(orig_np, edit_np, data_range=1.0)
            eval_entry[f"{cond}_PSNR"] = round(psnr_val, 2)

        eval_results["per_triplet"].append(eval_entry)

    # Summary per condition per edit type
    summary = {}
    for edit_type in ["word_swap", "attribute", "style"]:
        for cond in ["baseline", "ours", "p2p", "ours_p2p"]:
            lpips_vals = [e[f"{cond}_LPIPS"] for e in eval_results["per_triplet"]
                          if e["edit_type"] == edit_type and f"{cond}_LPIPS" in e]
            psnr_vals = [e[f"{cond}_PSNR"] for e in eval_results["per_triplet"]
                         if e["edit_type"] == edit_type and f"{cond}_PSNR" in e]

            if lpips_vals:
                summary[f"{edit_type}_{cond}_LPIPS_mean"] = round(np.mean(lpips_vals), 4)
                summary[f"{edit_type}_{cond}_LPIPS_std"] = round(np.std(lpips_vals), 4)
                summary[f"{edit_type}_{cond}_PSNR_mean"] = round(np.mean(psnr_vals), 2)
                summary[f"{edit_type}_{cond}_PSNR_std"] = round(np.std(psnr_vals), 2)
                summary[f"{edit_type}_{cond}_n"] = len(lpips_vals)

    # Overall
    for cond in ["baseline", "ours", "p2p", "ours_p2p"]:
        lpips_all = [e[f"{cond}_LPIPS"] for e in eval_results["per_triplet"]
                     if f"{cond}_LPIPS" in e]
        psnr_all = [e[f"{cond}_PSNR"] for e in eval_results["per_triplet"]
                    if f"{cond}_PSNR" in e]
        summary[f"overall_{cond}_LPIPS"] = round(np.mean(lpips_all), 4)
        summary[f"overall_{cond}_PSNR"] = round(np.mean(psnr_all), 2)

    eval_results["summary"] = summary

    # Print summary
    print(f"\n{'='*80}")
    print("EDITING BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Condition':<20s} {'LPIPS↓':>10s} {'PSNR↑':>8s}")
    print("-" * 40)
    for cond in ["baseline", "ours", "p2p", "ours_p2p"]:
        lpips_key = f"overall_{cond}_LPIPS"
        psnr_key = f"overall_{cond}_PSNR"
        if lpips_key in summary:
            print(f"{cond:<20s} {summary[lpips_key]:>10.4f} {summary[psnr_key]:>8.2f}")

    # Per edit type
    print(f"\n{'Edit Type':<15s} {'Condition':<20s} {'LPIPS↓':>10s} {'PSNR↑':>8s}")
    print("-" * 55)
    for edit_type in ["word_swap", "attribute", "style"]:
        for cond in ["baseline", "ours", "p2p", "ours_p2p"]:
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
    parser = argparse.ArgumentParser(description="Phase 7: Editing Benchmark")
    parser.add_argument("--mode", default="all",
                        choices=["caption", "edit", "eval", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.mode in ("caption", "all"):
        captions = generate_captions(force=args.force)
        edit_pairs = generate_edit_pairs(captions)
        for img, triplets in list(edit_pairs.items())[:3]:
            print(f"\n{img}:")
            for t in triplets:
                print(f"  [{t['edit_type']}] {t['swap']}")
                print(f"    src:  {t['source']}")
                print(f"    tgt:  {t['target']}")

    if args.mode in ("edit", "all"):
        results = run_sd15_editing(force=args.force)

    if args.mode in ("eval", "all"):
        results_file = OUT_DIR / "sd15_editing_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            evaluate_editing(results)
        else:
            print("No editing results found. Run --mode edit first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
