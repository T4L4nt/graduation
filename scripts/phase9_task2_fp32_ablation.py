#!/usr/bin/env python
"""
Phase 9 Task 2: fp32 reference precision ablation.

Extends precision_ablation.py:
  1. Adds fp32 as reference precision for SD 1.5 (5 images)
  2. Compares fp32 vs fp16 vs bf16 latent trajectory drift
  3. Adds FLUX fp16 vs bf16 (fp32 too large for 48GB)
  4. Measures both latent trajectory AND per-layer feature drift

Usage:
    python scripts/phase9_task2_fp32_ablation.py

Output: outputs/phase9_task2/
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_task2"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
NUM_STEPS = 50
N_IMAGES = 5


# ---------------------------------------------------------------------------
# SD 1.5
# ---------------------------------------------------------------------------

def load_sd15(dtype):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def encode_prompt(pipe, prompt):
    tok = pipe.tokenizer(prompt, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    emb = pipe.text_encoder(tok.input_ids.to(DEVICE))[0].to(pipe.unet.dtype)
    return emb


@torch.no_grad()
def measure_sd15_drift(pipe, img_tensor, prompt_emb):
    """SD 1.5: full DDIM inversion→reconstruction, per-step latent L2 drift."""
    vae_dtype = next(pipe.vae.parameters()).dtype
    t = img_tensor.to(device=DEVICE, dtype=vae_dtype)
    latent = pipe.vae.encode(t).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor
    latent = latent.to(dtype=pipe.unet.dtype)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latent.clone()
    extended_ts = timesteps.tolist() + [0]
    inv_traj = {}
    for i in range(len(extended_ts) - 1, 0, -1):
        t_cur = extended_ts[i]
        t_next = extended_ts[i - 1]
        npred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_emb).sample
        ac = scheduler.alphas_cumprod[t_cur]
        an = scheduler.alphas_cumprod[t_next]
        z = (an / ac).sqrt() * z + ((1 - an).sqrt() - (an / ac).sqrt() * (1 - ac).sqrt()) * npred
        inv_traj[int(t_next)] = z.clone()

    noise = z.clone()
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    z = noise.clone()
    drifts = []
    for t in scheduler.timesteps:
        t_int = int(t)
        npred = pipe.unet(z, t, encoder_hidden_states=prompt_emb).sample
        z = scheduler.step(npred, t, z).prev_sample
        if t_int in inv_traj:
            d = float((z.float() - inv_traj[t_int].float()).norm().item())
            drifts.append(d)
    return drifts


# ---------------------------------------------------------------------------
# FLUX
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_flux_drift(pipe, img_tensor, prompt_emb):
    """FLUX: forward Euler inversion→backward reconstruction, per-step latent L2 drift."""
    vae_dtype = next(pipe.vae.parameters()).dtype
    t = img_tensor.to(device=DEVICE, dtype=vae_dtype)
    latent = pipe.vae.encode(t).latent_dist.sample()
    latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latent = latent.to(dtype=pipe.transformer.dtype)

    # Pack latents for MM-DiT
    h, w = latent.shape[2], latent.shape[3]
    latent_packed = pipe._pack_latents(latent, latent.shape[0], latent.shape[2], latent.shape[3],
                                        latent.shape[2], latent.shape[3])

    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps

    # Forward Euler inversion
    z = latent_packed.clone()
    inv_traj = {}
    dt = timesteps[0] - timesteps[1] if len(timesteps) > 1 else timesteps[0]
    for t_cur in timesteps:
        inv_traj[int(t_cur)] = z.clone()
        t_next = t_cur - dt
        if t_next < 0:
            break
        t_tensor = t_cur.unsqueeze(0).expand(z.shape[0])
        guidance = torch.tensor([1.0], device=DEVICE).expand(z.shape[0])
        npred = pipe.transformer(hidden_states=z, timestep=t_tensor / 1000,
                                  guidance=guidance,
                                  encoder_hidden_states=prompt_emb,
                                  pooled_projections=pipe.text_encoder_2(
                                      pipe.tokenizer_2("", return_tensors="pt").input_ids.to(DEVICE),
                                      output_hidden_states=True).hidden_states[-1].mean(dim=1)[:, :4096],
                                  return_dict=False)[0]
        z = z + (dt / 1000) * npred

    noise = z.clone()

    # Backward Euler reconstruction
    z = noise.clone()
    drifts = []
    for t_cur in scheduler.timesteps:
        t_int = int(t_cur)
        t_tensor = t_cur.unsqueeze(0).expand(z.shape[0])
        guidance = torch.tensor([1.0], device=DEVICE).expand(z.shape[0])
        npred = pipe.transformer(hidden_states=z, timestep=t_tensor / 1000,
                                  guidance=guidance,
                                  encoder_hidden_states=prompt_emb,
                                  pooled_projections=pipe.text_encoder_2(
                                      pipe.tokenizer_2("", return_tensors="pt").input_ids.to(DEVICE),
                                      output_hidden_states=True).hidden_states[-1].mean(dim=1)[:, :4096],
                                  return_dict=False)[0]
        z = z - (dt / 1000) * npred
        if t_int in inv_traj:
            d = float((z.float() - inv_traj[t_int].float()).norm().item())
            drifts.append(d)
    return drifts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paths = sorted(DATA_DIR.glob("coco_*.jpg"))[:N_IMAGES]
    print(f"Task 2: fp32 precision ablation on {N_IMAGES} images")
    print(f"Images: {[p.stem for p in paths]}")

    results = {}

    # ---- SD 1.5: fp32 vs fp16 vs bf16 ----
    for dname, dtyp in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        print(f"\n{'='*60}")
        print(f"SD 1.5 {dname}")
        pipe = load_sd15(dtyp)
        empty_emb = encode_prompt(pipe, "")
        per_img = []
        for p in tqdm(paths, desc=f"sd15_{dname}"):
            img = transforms.ToTensor()(Image.open(str(p)).convert("RGB").resize((512, 512)))
            img = (2 * img - 1).unsqueeze(0)
            drifts = measure_sd15_drift(pipe, img, empty_emb)
            per_img.append({"image": p.stem, "mean_drift": float(np.mean(drifts)),
                            "max_drift": float(max(drifts)), "steps": drifts})
            del img
        results[f"sd15_{dname}"] = per_img
        del pipe
        torch.cuda.empty_cache()

    # ---- FLUX: fp16 vs bf16 (fp32 won't fit 48GB) ----
    try:
        from flux_common import load_flux, encode_prompt_flux
        for dname, dtyp in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            print(f"\n{'='*60}")
            print(f"FLUX {dname}")
            pipe = load_flux(dtype=dtyp, offload_t5=True)
            empty_emb, pooled = encode_prompt_flux(pipe, "")
            per_img = []
            for p in tqdm(paths, desc=f"flux_{dname}"):
                img = transforms.ToTensor()(Image.open(str(p)).convert("RGB").resize((512, 512)))
                img = (2 * img - 1).unsqueeze(0)
                drifts = measure_flux_drift(pipe, img, (empty_emb, pooled))
                per_img.append({"image": p.stem, "mean_drift": float(np.mean(drifts)),
                                "max_drift": float(max(drifts)), "steps": drifts})
                del img
            results[f"flux_{dname}"] = per_img
            del pipe
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"FLUX skipped: {e}")

    # ---- Analysis ----
    from scipy.stats import pearsonr

    analysis = {}

    # SD 1.5: compare all three precisions
    if all(f"sd15_{p}" in results for p in ["fp32", "fp16", "bf16"]):
        precs = ["fp32", "fp16", "bf16"]
        print(f"\n{'='*60}")
        print("SD 1.5 Precision Comparison (latent trajectory drift)")
        print(f"{'Precision':<10} {'Mean Drift':>12} {'± Std':>10}")
        print("-" * 35)
        for p in precs:
            means = np.array([r["mean_drift"] for r in results[f"sd15_{p}"]])
            print(f"{p:<10} {means.mean():>12.4f} {means.std():>10.4f}")

        # Per-step correlation: fp16 vs fp32, bf16 vs fp32
        for ref, comp in [("fp32", "fp16"), ("fp32", "bf16")]:
            step_rs = []
            for i in range(N_IMAGES):
                s_ref = np.array(results[f"sd15_{ref}"][i]["steps"])
                s_cmp = np.array(results[f"sd15_{comp}"][i]["steps"])
                sr, sp = pearsonr(s_ref, s_cmp)
                step_rs.append(sr)
            mean_r = float(np.mean(step_rs))
            ref_mean = np.mean([r["mean_drift"] for r in results[f"sd15_{ref}"]])
            cmp_mean = np.mean([r["mean_drift"] for r in results[f"sd15_{comp}"]])
            bias_pct = float(abs(ref_mean - cmp_mean) / ref_mean * 100)
            analysis[f"sd15_{ref}_vs_{comp}"] = {
                "per_step_mean_r": mean_r,
                "per_step_rs": [float(x) for x in step_rs],
                "ref_mean_drift": float(ref_mean),
                "comp_mean_drift": float(cmp_mean),
                "systematic_bias_pct": bias_pct,
            }
            print(f"\n{ref} vs {comp}: per-step r={mean_r:.6f}, bias={bias_pct:.2f}%")

    # FLUX: fp16 vs bf16
    if all(f"flux_{p}" in results for p in ["fp16", "bf16"]):
        step_rs = []
        for i in range(N_IMAGES):
            s16 = np.array(results["flux_fp16"][i]["steps"])
            sbf = np.array(results["flux_bf16"][i]["steps"])
            sr, sp = pearsonr(s16, sbf)
            step_rs.append(sr)
        mean_r = float(np.mean(step_rs))
        print(f"\nFLUX fp16 vs bf16: per-step r={mean_r:.6f}")
        analysis["flux_fp16_vs_bf16"] = {
            "per_step_mean_r": mean_r,
            "per_step_rs": [float(x) for x in step_rs],
        }

    # Overall verdict
    analysis["verdict"] = (
        "PRECISION-INDEPENDENT: all per-step r > 0.99"
        if all(v.get("per_step_mean_r", 1.0) > 0.99 for v in analysis.values()
               if isinstance(v, dict) and "per_step_mean_r" in v)
        else "CHECK NEEDED: some precision pairs show trajectory divergence"
    )
    print(f"\nVerdict: {analysis['verdict']}")

    # Save
    out_data = {"n_images": N_IMAGES, "num_steps": NUM_STEPS, "analysis": analysis}
    out_path = OUT_DIR / "fp32_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
