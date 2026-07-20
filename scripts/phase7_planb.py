#!/usr/bin/env python
"""
Phase 7 Plan B: Error–edit direction separation via source-prompt reconstruction.

Core insight: f_inv - f_recon^tgt (current correction) conflates inversion error
with edit direction, collapsing CLIP-Dir. Plan B extracts the pure inversion error
Δ = f_inv - f_recon^src via same-prompt reconstruction, then applies:
    f_out = f_recon^tgt + λ·Δ
during editing reconstruction. Hypothesis: LPIPS improves (error fixed) while
CLIP-Dir is preserved (edit direction untouched).

Includes λ sweep (0.0–1.0) for frontier curve.

Usage:
    python scripts/phase7_planb.py

Output: outputs/phase7_editing_100image/planb_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_common import load_pipeline, load_image, decode_latent, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase7_editing_100image"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
CACHE_DIR = OUT_DIR / "cache"
DEVICE = "cuda"
NUM_STEPS = 50
LAMBDA_SCAN = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                       local_files_only=True).to(DEVICE)
    model.eval()
    img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                   local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                               local_files_only=True)
    return model, img_proc, tokenizer


@torch.no_grad()
def clip_image_embed(clip_model, img_proc, tensor):
    img = ((tensor.squeeze(0) + 1) / 2).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    inputs = img_proc(images=Image.fromarray(img), return_tensors="pt").to(DEVICE)
    out = clip_model.get_image_features(**inputs)
    emb = out.pooler_output if hasattr(out, 'pooler_output') else out
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


@torch.no_grad()
def clip_text_embed(clip_model, tokenizer, text):
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    out = clip_model.get_text_features(**inputs)
    emb = out.pooler_output if hasattr(out, 'pooler_output') else out
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def encode_prompt(pipe, prompt):
    with torch.no_grad():
        inputs = pipe.tokenizer(prompt, padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
        emb = pipe.text_encoder(inputs.input_ids.to(pipe.device))[0].to(pipe.unet.dtype)
    return emb


# ---------------------------------------------------------------------------
# DDIM helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_invert_save(pipe, z0, prompt_embeds):
    """DDIM inversion, save full latent trajectory."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = z0.clone()
    traj = {}
    extended_ts = timesteps.tolist() + [0]
    for i in range(len(extended_ts) - 1, 0, -1):
        t_cur = extended_ts[i]
        t_next = extended_ts[i - 1]
        npred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample
        ac = scheduler.alphas_cumprod[t_cur]
        an = scheduler.alphas_cumprod[t_next]
        z = (an / ac).sqrt() * z + ((1 - an).sqrt() - (an / ac).sqrt() * (1 - ac).sqrt()) * npred
        traj[int(t_next)] = z.clone()
    return z, traj


@torch.no_grad()
def ddim_recon_save(pipe, noise, prompt_embeds):
    """DDIM reconstruction (no correction), save full latent trajectory."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    traj = {}
    for t in timesteps:
        t_int = int(t)
        npred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
        traj[t_int] = z.clone()
    return z, traj


@torch.no_grad()
def ddim_edit_planb_mid(pipe, noise, src_recon_traj, inv_traj, tgt_embeds, lam,
                          mid_frac=0.33):
    """Plan B variant: inject Δ[t] only during the middle fraction of timesteps."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    n_mid_start = int(len(timesteps) * (1 - mid_frac) / 2)
    n_mid_end = n_mid_start + int(len(timesteps) * mid_frac)
    z = noise.clone()
    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        npred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
        if n_mid_start <= step_idx < n_mid_end:
            if t_int in src_recon_traj and t_int in inv_traj:
                delta = inv_traj[t_int].to(DEVICE) - src_recon_traj[t_int].to(DEVICE)
                z = z + lam * delta
    return z


def ddim_edit_planb(pipe, noise, src_recon_traj, inv_traj, tgt_embeds, lam):
    """Reconstruct with target prompt + Plan B correction: f_out = f_recon + λ(f_inv - f_recon_src)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    for t in timesteps:
        t_int = int(t)
        npred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
        if t_int in src_recon_traj and t_int in inv_traj:
            delta = inv_traj[t_int].to(DEVICE) - src_recon_traj[t_int].to(DEVICE)
            z = z + lam * delta
    return z


@torch.no_grad()
def ddim_edit_baseline(pipe, noise, inv_traj, tgt_embeds, lam):
    """Reconstruct with target prompt + original correction: f_out = f_recon + λ(f_inv - f_recon)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    for t in timesteps:
        t_int = int(t)
        npred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
        if t_int in inv_traj:
            z = z + lam * (inv_traj[t_int].to(DEVICE) - z)
    return z


@torch.no_grad()
def ddim_edit_endpoint_correct(pipe, noise, inv_traj, tgt_embeds, lam):
    """Reconstruct with target prompt, then apply ENDPOINT-ONLY correction
    using the clean latent error (z_0 - z_recon_src)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    for t in timesteps:
        npred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
    # Endpoint correction: inv_traj[0] is the original clean latent
    if 0 in inv_traj:
        z = z + lam * (inv_traj[0].to(DEVICE) - z)
    return z


@torch.no_grad()
def ddim_edit_no_correction(pipe, noise, tgt_embeds):
    """Reconstruct with target prompt, no correction (baseline)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    for t in timesteps:
        npred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
        z = scheduler.step(npred, t, z).prev_sample
    return z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CACHE_DIR / "captions_100.json") as f:
        captions = json.load(f)
    with open(CACHE_DIR / "edit_pairs_100.json") as f:
        edit_pairs = json.load(f)

    print(f"Plan B: {len(edit_pairs)} images, "
          f"{sum(len(v) for v in edit_pairs.values())} edit pairs")
    print(f"λ scan: {LAMBDA_SCAN}")

    print("Loading SD 1.5...")
    pipe = load_pipeline()
    print("Loading CLIP...")
    clip_model, clip_img_proc, clip_tokenizer = load_clip()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    empty_emb = encode_prompt(pipe, "")

    results = []
    incremental_path = OUT_DIR / "planb_results_partial.json"

    # Reduced set for Plan B: first time, check on 20 images to get α estimate,
    # then full run
    USE_FULL = True  # Full 100-image run (2026-07-18)
    img_items = list(edit_pairs.items())
    if not USE_FULL:
        img_items = img_items[:20]

    for img_name, triplet_list in tqdm(img_items, desc="Plan B"):
        img_path = DATA_DIR / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        latent, orig_tensor = load_image(pipe, str(img_path))
        clip_orig = clip_image_embed(clip_model, clip_img_proc, orig_tensor)

        caption = captions.get(img_name, "")
        src_embeds = encode_prompt(pipe, caption)

        # ---- Step 1: Inversion with source caption ----
        noise, inv_traj = ddim_invert_save(pipe, latent, src_embeds)

        # ---- Step 2: Source-prompt reconstruction (for Plan B error vector) ----
        recon_src, src_recon_traj = ddim_recon_save(pipe, noise, src_embeds)

        # CLIP text directions for all triplets
        clip_src_txt = clip_text_embed(clip_model, clip_tokenizer, caption)
        clip_tgt_cache = {}
        text_dir_cache = {}
        for tpl in triplet_list:
            tgt = tpl["target"]
            if tgt not in clip_tgt_cache:
                clip_tgt_txt = clip_text_embed(clip_model, clip_tokenizer, tgt)
                text_dir = clip_tgt_txt - clip_src_txt
                text_dir = text_dir / text_dir.norm(dim=-1, keepdim=True)
                clip_tgt_cache[tgt] = clip_tgt_txt
                text_dir_cache[tgt] = text_dir

        for triplet in triplet_list:
            target_prompt = triplet["target"]
            tgt_embeds = encode_prompt(pipe, target_prompt)
            text_dir = text_dir_cache[target_prompt]

            # ---- Baseline (no correction) ----
            recon_bl = ddim_edit_no_correction(pipe, noise, tgt_embeds)
            tensor_bl = decode_latent(pipe, recon_bl)
            m_bl = compute_metrics(orig_tensor, tensor_bl, lpips_fn)
            clip_bl = clip_image_embed(clip_model, clip_img_proc, tensor_bl)
            img_dir_bl = clip_bl - clip_orig
            img_dir_bl = img_dir_bl / img_dir_bl.norm(dim=-1, keepdim=True)
            cd_bl = float((img_dir_bl * text_dir).sum().item())

            entry = {
                "image": img_name,
                "edit_type": triplet["edit_type"],
                "target_prompt": target_prompt,
                "swap": triplet.get("swap", ""),
                "baseline": {
                    "PSNR": m_bl["PSNR"], "SSIM": m_bl["SSIM"],
                    "LPIPS": m_bl["LPIPS"], "CLIPDir": cd_bl,
                },
                "ours": {},    # original correction λ sweep
                "planb": {},   # Plan B per-timestep
                "midstep": {}, # Plan B mid-step only
            }

            # ---- Original correction λ sweep ----
            for lam in LAMBDA_SCAN:
                recon_ours = ddim_edit_baseline(pipe, noise, inv_traj, tgt_embeds, lam)
                tensor_ours = decode_latent(pipe, recon_ours)
                m_ours = compute_metrics(orig_tensor, tensor_ours, lpips_fn)
                clip_ours = clip_image_embed(clip_model, clip_img_proc, tensor_ours)
                img_dir_ours = clip_ours - clip_orig
                img_dir_ours = img_dir_ours / img_dir_ours.norm(dim=-1, keepdim=True)
                cd_ours = float((img_dir_ours * text_dir).sum().item())
                entry["ours"][str(lam)] = {
                    "PSNR": m_ours["PSNR"], "SSIM": m_ours["SSIM"],
                    "LPIPS": m_ours["LPIPS"], "CLIPDir": cd_ours,
                }

            # ---- Plan B per-timestep (λ=0.1, 0.5) ----
            for lam in [0.1, 0.5]:
                recon_pb = ddim_edit_planb(pipe, noise, src_recon_traj, inv_traj,
                                            tgt_embeds, lam)
                tensor_pb = decode_latent(pipe, recon_pb)
                m_pb = compute_metrics(orig_tensor, tensor_pb, lpips_fn)
                clip_pb = clip_image_embed(clip_model, clip_img_proc, tensor_pb)
                img_dir_pb = clip_pb - clip_orig
                img_dir_pb = img_dir_pb / img_dir_pb.norm(dim=-1, keepdim=True)
                cd_pb = float((img_dir_pb * text_dir).sum().item())
                entry["planb"][str(lam)] = {
                    "PSNR": m_pb["PSNR"], "SSIM": m_pb["SSIM"],
                    "LPIPS": m_pb["LPIPS"], "CLIPDir": cd_pb,
                }

            # ---- Plan B mid-step only (λ=0.1, 0.5) ----
            for lam in [0.1, 0.5]:
                recon_ms = ddim_edit_planb_mid(pipe, noise, src_recon_traj,
                                                inv_traj, tgt_embeds, lam)
                tensor_ms = decode_latent(pipe, recon_ms)
                m_ms = compute_metrics(orig_tensor, tensor_ms, lpips_fn)
                clip_ms = clip_image_embed(clip_model, clip_img_proc, tensor_ms)
                img_dir_ms = clip_ms - clip_orig
                img_dir_ms = img_dir_ms / img_dir_ms.norm(dim=-1, keepdim=True)
                cd_ms = float((img_dir_ms * text_dir).sum().item())
                entry["midstep"][str(lam)] = {
                    "PSNR": m_ms["PSNR"], "SSIM": m_ms["SSIM"],
                    "LPIPS": m_ms["LPIPS"], "CLIPDir": cd_ms,
                }

            results.append(entry)

        if len(results) % 20 == 0:
            with open(incremental_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save
    out_path = OUT_DIR / "planb_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # ---- Summary ----
    from scipy.stats import ttest_rel

    lpips_bl = np.array([r["baseline"]["LPIPS"] for r in results])
    cd_bl = np.array([r["baseline"]["CLIPDir"] for r in results])

    print(f"\n{'='*80}")
    print(f"PLAN B — ERROR–EDIT SEPARATION ({len(results)} pairs)")
    print(f"{'='*80}")

    print(f"\n--- Original correction λ sweep ---")
    print(f"{'λ':>5s}  {'LPIPS↓':>10s}  {'CLIP-Dir↑':>10s}")
    print("-" * 32)
    for lam in LAMBDA_SCAN:
        lpips_vals = np.array([r["ours"][str(lam)]["LPIPS"] for r in results])
        cd_vals = np.array([r["ours"][str(lam)]["CLIPDir"] for r in results])
        print(f"{lam:>5.1f}  {lpips_vals.mean():>10.4f}  {cd_vals.mean():>10.4f}")

    print(f"\nbaseline (no correction): LPIPS={lpips_bl.mean():.4f} CLIP-Dir={cd_bl.mean():.4f}")

    # Plan B + Mid-step results
    for label, key, lams in [("Plan B per-timestep", "planb", [0.1, 0.5]),
                               ("Plan B mid-step", "midstep", [0.1, 0.5])]:
        print(f"\n--- {label} ---")
        for lam in lams:
            lpips_vals = np.array([r[key][str(lam)]["LPIPS"] for r in results])
            cd_vals = np.array([r[key][str(lam)]["CLIPDir"] for r in results])
            print(f"  λ={lam}: LPIPS={lpips_vals.mean():.4f} CLIP-Dir={cd_vals.mean():.4f}")

    # Save full frontier
    frontier = {}
    for lam in LAMBDA_SCAN:
        lpips_vals = [r["ours"][str(lam)]["LPIPS"] for r in results]
        cd_vals = [r["ours"][str(lam)]["CLIPDir"] for r in results]
        frontier[str(lam)] = {
            "LPIPS": float(np.mean(lpips_vals)),
            "CLIPDir": float(np.mean(cd_vals)),
        }

    for key in ["planb", "midstep"]:
        frontier[key] = {}
        for lam in [0.1, 0.5]:
            lpips_vals = [r[key][str(lam)]["LPIPS"] for r in results]
            cd_vals = [r[key][str(lam)]["CLIPDir"] for r in results]
            frontier[key][str(lam)] = {
                "LPIPS": float(np.mean(lpips_vals)),
                "CLIPDir": float(np.mean(cd_vals)),
            }

    summary = {
        "n_pairs": len(results),
        "n_images": len(set(r["image"] for r in results)),
        "baseline_LPIPS": float(lpips_bl.mean()),
        "baseline_CLIPDir": float(cd_bl.mean()),
        "lambda_frontier": frontier,
    }
    sum_path = OUT_DIR / "planb_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {sum_path}")
    print("Done.")


if __name__ == "__main__":
    main()
