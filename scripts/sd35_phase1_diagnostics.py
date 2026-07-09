#!/usr/bin/env python
"""
SD 3.5 Medium Phase 1: Predictive validation of Architecture Fingerprint.

PREDICTION: drift peaks at layers 12-14 (dual->standard transition boundary),
secondary at 21-23 (pre-output compression), low in 0-11 (dual attention).

Usage: python scripts/sd35_phase1_diagnostics.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline

OUT_DIR = Path("outputs/sd35_phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/hiaskc/.cache/huggingface/hub/models/stabilityai--stable-diffusion-3.5-medium/snapshots/master"
TEST_IMAGES = sorted(Path("data/coco_val").glob("*.jpg"))[:19]

PREDICTION = {
    "predicted_primary_peak": "layers 12-14 (dual_attention->standard transition)",
    "predicted_secondary": "layers 21-23 (pre-output compression)",
    "predicted_low": "layers 0-11 (dual attention preservation)",
    "predicted_shape": "mid-boundary peak + late tail",
    "dual_attention_layers": list(range(13)),
    "total_blocks": 24,
    "prediction_made": "2026-07-09",
}


class BlockFeatureExtractor:
    """Hook all 24 JointTransformerBlocks, capture hidden_states."""

    def __init__(self):
        self.features = {}
        self._hooks = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            # JointTransformerBlock returns (encoder_hidden_states, hidden_states)
            if isinstance(output, tuple) and len(output) == 2:
                hs = output[1]  # image hidden_states
            else:
                hs = output
            self.features[name] = hs.detach().float().cpu()
        return fn

    def register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            self._hooks.append(block.register_forward_hook(self._hook_fn(f"block_{i}")))

    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def clear(self):
        self.features.clear()


def load_pipeline():
    print("Loading SD 3.5 Medium (with CPU offload to conserve VRAM)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        local_files_only=True, tokenizer_3=None, text_encoder_3=None,
    )
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return pipe


def encode_prompt(pipe, prompt):
    """Properly handle missing T5 with zero-padding (replicates pipeline logic)."""
    device = pipe._execution_device
    joint_dim = pipe.transformer.config.joint_attention_dim  # 4096

    # CLIP-L
    tok1 = pipe.tokenizer(prompt, padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        out1 = pipe.text_encoder(tok1["input_ids"].to(device), output_hidden_states=True)
    h1 = out1.hidden_states[-2].to(dtype=pipe.text_encoder.dtype)
    p1 = out1.text_embeds.to(dtype=pipe.text_encoder.dtype)

    # CLIP-G
    tok2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        out2 = pipe.text_encoder_2(tok2["input_ids"].to(device), output_hidden_states=True)
    h2 = out2.hidden_states[-2].to(dtype=pipe.text_encoder_2.dtype)
    p2 = out2.text_embeds.to(dtype=pipe.text_encoder_2.dtype)

    # Concatenate CLIP features + pad to joint_dim
    clip_hidden = torch.cat([h1, h2], dim=-1)  # [1, 77, 2048]
    clip_padded = torch.nn.functional.pad(
        clip_hidden, (0, joint_dim - clip_hidden.shape[-1]))  # [1, 77, 4096]

    # Zero T5 placeholder [1, 256, 4096]
    t5_embed = torch.zeros(1, 256, joint_dim, device=device, dtype=pipe.transformer.dtype)

    encoder_hidden_states = torch.cat([clip_padded, t5_embed], dim=1).to(pipe.transformer.dtype)
    pooled = torch.cat([p1, p2], dim=-1).to(pipe.transformer.dtype)

    return encoder_hidden_states, pooled


def encode_image(pipe, image_path):
    img = Image.open(image_path).convert("RGB").resize((1024, 1024))
    pixel_values = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
    with torch.no_grad():
        latent = pipe.vae.encode(pixel_values).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latent


@torch.no_grad()
def run_diagnosis(pipe, latent, encoder_hidden_states, pooled, extractor, num_steps=50):
    """Invert + reconstruct, compare block features at t=0 (clean state)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    t0 = torch.zeros(latent.shape[0], device="cuda", dtype=torch.float32)

    # Reference features at clean latent x_0
    extractor.register(pipe.transformer)
    _ = pipe.transformer(hidden_states=latent, encoder_hidden_states=encoder_hidden_states,
                         pooled_projections=pooled, timestep=t0, return_dict=False)[0]
    features_ref = dict(extractor.features)
    extractor.clear()

    # Euler inversion: x_0 -> x_T
    z = latent.clone()
    for i, t in enumerate(timesteps):
        dt = sigmas[i+1] - sigmas[i]
        v = pipe.transformer(hidden_states=z, encoder_hidden_states=encoder_hidden_states,
                             pooled_projections=pooled,
                             timestep=t.unsqueeze(0).expand(z.shape[0]), return_dict=False)[0]
        z = z + dt * v

    # Euler reconstruction: x_T -> x_0' (reverse direction)
    for i in range(len(timesteps)-1, -1, -1):
        t = timesteps[i]
        dt = sigmas[i+1] - sigmas[i]
        v = pipe.transformer(hidden_states=z, encoder_hidden_states=encoder_hidden_states,
                             pooled_projections=pooled,
                             timestep=t.unsqueeze(0).expand(z.shape[0]), return_dict=False)[0]
        z = z - dt * v

    # Reconstruction features at t=0
    _ = pipe.transformer(hidden_states=z, encoder_hidden_states=encoder_hidden_states,
                         pooled_projections=pooled, timestep=t0, return_dict=False)[0]
    features_recon = dict(extractor.features)
    extractor.remove()

    # PSNR
    z_dec = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    orig_dec = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    recon_pix = (pipe.vae.decode(z_dec).sample / 2 + 0.5).clamp(0, 1)
    orig_pix = (pipe.vae.decode(orig_dec).sample / 2 + 0.5).clamp(0, 1)
    mse = (recon_pix - orig_pix).pow(2).mean()
    psnr = float(10 * torch.log10(1.0 / mse) if mse > 0 else 100.0)

    return {"psnr": psnr, "features_ref": features_ref, "features_recon": features_recon}


def compute_drift(features_ref, features_recon):
    drift = {}
    for name in sorted(features_ref.keys(), key=lambda x: int(x.split("_")[1])):
        d = (features_recon[name] - features_ref[name]).norm().item()
        n = features_ref[name].norm().item()
        drift[name] = d / max(n, 1e-10)
    return drift


def main():
    print("=" * 70)
    print("SD 3.5 Medium — Predictive Validation")
    print(f"Prediction: {PREDICTION['predicted_shape']}")
    print(f"  Primary: {PREDICTION['predicted_primary_peak']}")
    print(f"  Secondary: {PREDICTION['predicted_secondary']}")
    print("=" * 70)

    pipe = load_pipeline()
    extractor = BlockFeatureExtractor()

    prompt = "a photograph of a scenic landscape"
    encoder_hidden_states, pooled = encode_prompt(pipe, prompt)
    print(f"Encoder hidden: {encoder_hidden_states.shape}, pooled: {pooled.shape}")

    per_image_drift = {}
    metrics_all = {}

    for img_path in tqdm(TEST_IMAGES, desc="Images"):
        name = img_path.stem
        try:
            latent = encode_image(pipe, img_path)
            result = run_diagnosis(pipe, latent, encoder_hidden_states, pooled, extractor, num_steps=50)
            drift = compute_drift(result["features_ref"], result["features_recon"])
            per_image_drift[name] = drift
            metrics_all[name] = {"psnr": result["psnr"]}
            top = max(drift, key=drift.get)
            print(f"  {name}: PSNR={result['psnr']:.1f} dB, peak={top} ({drift[top]:.4f})")
        except Exception as e:
            print(f"  {name}: ERROR {e}")
            import traceback; traceback.print_exc()

    if not per_image_drift:
        print("No successful images!")
        return

    # Aggregate
    block_names = sorted(per_image_drift[list(per_image_drift.keys())[0]].keys(),
                        key=lambda x: int(x.split("_")[1]))
    drift_agg = {}
    for bn in block_names:
        vals = [per_image_drift[img][bn] for img in per_image_drift]
        drift_agg[bn] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    ranking = sorted(drift_agg, key=lambda x: drift_agg[x]["mean"], reverse=True)
    peak_block = ranking[0]
    peak_idx = int(peak_block.split("_")[1])
    predicted_ok = 12 <= peak_idx <= 14

    # Spearman: compare actual drift to predicted ordinal
    actual = np.array([drift_agg[f"block_{i}"]["mean"] for i in range(24)])
    pred_ord = np.zeros(24)
    for i in range(24):
        if 12 <= i <= 14: pred_ord[i] = 0       # highest
        elif 21 <= i <= 23: pred_ord[i] = 1     # high
        elif 15 <= i <= 20: pred_ord[i] = 2     # moderate
        else: pred_ord[i] = 3                    # lowest

    from scipy.stats import spearmanr
    rho, p_val = spearmanr(actual, -pred_ord)

    # Also a simpler metric: is peak in dual->standard boundary?
    # Check if mean drift in 12-14 is higher than in 15-20
    mean_boundary = np.mean([drift_agg[f"block_{i}"]["mean"] for i in range(12, 15)])
    mean_mid = np.mean([drift_agg[f"block_{i}"]["mean"] for i in range(15, 21)])
    mean_late = np.mean([drift_agg[f"block_{i}"]["mean"] for i in range(21, 24)])
    mean_early = np.mean([drift_agg[f"block_{i}"]["mean"] for i in range(0, 12)])

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Peak: {peak_block} (idx={peak_idx}) in [12,14]: {'YES' if predicted_ok else 'NO'}")
    print(f"  Spearman rho (actual vs predicted ordinal): {rho:.3f} (p={p_val:.4f})")
    print(f"  Region means: early(0-11)={mean_early:.4f} | boundary(12-14)={mean_boundary:.4f}")
    print(f"                mid(15-20)={mean_mid:.4f} | late(21-23)={mean_late:.4f}")
    print(f"  Top-5:  {ranking[:5]}")
    print(f"  Top-10: {ranking[:10]}")
    print()
    max_d = drift_agg[ranking[0]]["mean"]
    for bn in ranking:
        d = drift_agg[bn]
        idx = int(bn.split("_")[1])
        kind = "dual" if idx <= 12 else "std"
        marker = " *** PEAK" if bn == peak_block else ""
        bar = "█" * int(d["mean"] / max_d * 40)
        print(f"  {bn} [{kind}] {d['mean']:.6f} {bar}{marker}")

    # Save
    summary = {
        "prediction": PREDICTION,
        "validation": {
            "peak_correct": predicted_ok,
            "spearman_rho": float(rho), "spearman_p": float(p_val),
            "actual_peak": peak_block, "actual_peak_idx": peak_idx,
            "region_means": {"early_0_11": float(mean_early),
                             "boundary_12_14": float(mean_boundary),
                             "mid_15_20": float(mean_mid),
                             "late_21_23": float(mean_late)},
        },
        "per_image": {img: {bn: float(v) for bn, v in d.items()}
                      for img, d in per_image_drift.items()},
        "aggregated": drift_agg,
        "ranking": ranking, "top5": ranking[:5], "top10": ranking[:10],
        "metrics": metrics_all,
    }
    with open(OUT_DIR / "layer_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'layer_drift_summary.json'}")

    PREDICTION["status"] = "verified" if predicted_ok else "falsified"
    PREDICTION["actual_peak"] = peak_block
    PREDICTION["spearman_rho"] = float(rho)
    with open(OUT_DIR / "prediction_record.json", "w") as f:
        json.dump(PREDICTION, f, indent=2)


if __name__ == "__main__":
    main()
