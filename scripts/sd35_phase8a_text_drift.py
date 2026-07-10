#!/usr/bin/env python
"""
Phase 8a: SD 3.5 Medium text drift measurement + prediction refinement.

Adds text token drift to the Phase 1 diagnosis. The prediction:
  "cross-modal acquisition (dual->standard) → text drift should DECREASE"
  Because standard joint attention provides cross-modal context that stabilizes text.

Also quantifies the corrected prediction's match and writes prediction_record.json.

Usage:
    python scripts/sd35_phase8a_text_drift.py
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

# Original prediction AND corrected prediction (after seeing partial results)
PREDICTION_V1 = {
    "predicted_primary_peak": "layers 12-14 (dual_attention->standard transition)",
    "predicted_secondary": "layers 21-23 (pre-output compression)",
    "predicted_low": "layers 0-11 (dual attention preservation)",
    "predicted_shape": "mid-boundary peak + late tail",
    "status": "partially_falsified",
    "actual": "peak at block_22 (late tail), boundary 12-14 is global MINIMUM",
}

PREDICTION_V2 = {
    "framework": "three-element predictive framework",
    "analysis": (
        "SD 3.5's dual->standard transition is a CROSS-MODAL ACQUISITION, "
        "not a loss like FLUX's joint->single. Standard joint attention GAINS "
        "cross-modal context, which should STABILIZE features. Hence the "
        "boundary (12-14) is a drift VALLEY, not a peak."
    ),
    "corrected_prediction_image_drift": {
        "primary_peak": "layers 21-23 (pre-output compression) -- CONFIRMED by Phase 1",
        "valley": "layers 12-14 (cross-modal acquisition stabilizes features) -- CONFIRMED",
        "shape": "late-tail dominant, with mid-architecture valley",
    },
    "corrected_prediction_text_drift": {
        "primary_peak": "layers 0-12 (dual attention — separate modality processing)",
        "valley": "layers 13-20 (joint attention stabilizes text via cross-modal context)",
        "late_rise": "layers 21-23 (pre-output text compression)",
        "mechanism": (
            "In dual-attention layers (0-12), text is processed independently "
            "before joint fusion — less cross-modal stabilization → higher text drift. "
            "In standard joint layers (13-23), text+image are processed together → "
            "cross-modal context stabilizes text, reducing drift. "
            "Prediction: mean text drift(0-12) > mean text drift(13-20)."
        ),
    },
    "prediction_made": "2026-07-09",
}


class DualFeatureExtractor:
    """Hook all 24 blocks, capture BOTH image (output[1]) and text (output[0]) features."""

    def __init__(self):
        self.img_features = {}
        self.txt_features = {}
        self._hooks = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                if output[0] is not None:
                    self.txt_features[name] = output[0].detach().float().cpu()
                if output[1] is not None:
                    self.img_features[name] = output[1].detach().float().cpu()
            elif output is not None:
                self.img_features[name] = output.detach().float().cpu()
        return fn

    def register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            self._hooks.append(block.register_forward_hook(self._hook_fn(f"block_{i}")))

    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def clear(self):
        self.img_features.clear()
        self.txt_features.clear()


def load_pipeline():
    print("Loading SD 3.5 Medium (CPU offload)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        local_files_only=True, tokenizer_3=None, text_encoder_3=None,
    )
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    return pipe


def encode_prompt(pipe, prompt):
    device = pipe._execution_device
    joint_dim = pipe.transformer.config.joint_attention_dim  # 4096
    tok1 = pipe.tokenizer(prompt, padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt")
    tok2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        out1 = pipe.text_encoder(tok1["input_ids"].to(device), output_hidden_states=True)
        out2 = pipe.text_encoder_2(tok2["input_ids"].to(device), output_hidden_states=True)
    h1 = out1.hidden_states[-2].to(dtype=pipe.text_encoder.dtype)
    h2 = out2.hidden_states[-2].to(dtype=pipe.text_encoder_2.dtype)
    p1 = out1.text_embeds.to(dtype=pipe.text_encoder.dtype)
    p2 = out2.text_embeds.to(dtype=pipe.text_encoder_2.dtype)
    clip = torch.cat([h1, h2], dim=-1)
    clip = torch.nn.functional.pad(clip, (0, joint_dim - clip.shape[-1]))
    t5z = torch.zeros(1, 256, joint_dim, device=device, dtype=torch.float16)
    enc = torch.cat([clip, t5z], dim=1).to(torch.float16)
    pooled = torch.cat([p1, p2], dim=-1).to(torch.float16)
    return enc, pooled


def encode_image(pipe, image_path):
    img = Image.open(image_path).convert("RGB").resize((1024, 1024))
    pixel_values = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
    with torch.no_grad():
        latent = pipe.vae.encode(pixel_values).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latent


@torch.no_grad()
def diagnose_with_text_drift(pipe, latent, enc, pooled, extractor, num_steps=50):
    """Full inversion+reconstruction, capture image AND text features at t=0."""
    scheduler = pipe.scheduler; scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = scheduler.timesteps; sigmas = scheduler.sigmas
    t0 = torch.zeros(1, device="cuda", dtype=torch.float32)

    # Reference at t=0
    extractor.register(pipe.transformer)
    _ = pipe.transformer(hidden_states=latent, encoder_hidden_states=enc,
                         pooled_projections=pooled, timestep=t0, return_dict=False)[0]
    img_ref = dict(extractor.img_features)
    txt_ref = dict(extractor.txt_features)
    extractor.clear()

    # Inversion (no hooks needed for drift measurement)
    z = latent.clone()
    for i in range(num_steps):
        dt = sigmas[i+1] - sigmas[i]
        z = z + dt * pipe.transformer(hidden_states=z, encoder_hidden_states=enc,
            pooled_projections=pooled, timestep=timesteps[i].unsqueeze(0).expand(z.shape[0]),
            return_dict=False)[0]
        if i % 20 == 0: torch.cuda.empty_cache()

    # Reconstruction
    for i in range(num_steps-1, -1, -1):
        dt = sigmas[i+1] - sigmas[i]
        z = z - dt * pipe.transformer(hidden_states=z, encoder_hidden_states=enc,
            pooled_projections=pooled, timestep=timesteps[i].unsqueeze(0).expand(z.shape[0]),
            return_dict=False)[0]
        if i % 20 == 0: torch.cuda.empty_cache()

    # Recon features at t=0
    _ = pipe.transformer(hidden_states=z, encoder_hidden_states=enc,
                         pooled_projections=pooled, timestep=t0, return_dict=False)[0]
    img_recon = dict(extractor.img_features)
    txt_recon = dict(extractor.txt_features)
    extractor.remove()

    return img_ref, img_recon, txt_ref, txt_recon


def compute_drift(ref_dict, recon_dict):
    drift = {}
    for name in sorted(ref_dict.keys(), key=lambda x: int(x.split("_")[1])):
        d = (recon_dict[name] - ref_dict[name]).norm().item()
        n = ref_dict[name].norm().item()
        drift[name] = d / max(n, 1e-10)
    return drift


def main():
    print("=" * 70)
    print("Phase 8a: SD 3.5 Medium — Text Drift Measurement")
    print("Corrected Prediction: cross-modal acquisition → text drift valley")
    print("=" * 70)

    pipe = load_pipeline()
    extractor = DualFeatureExtractor()

    prompt = "a photograph of a scenic landscape"
    enc, pooled = encode_prompt(pipe, prompt)

    per_image_img_drift = {}
    per_image_txt_drift = {}

    for img_path in tqdm(TEST_IMAGES, desc="Images"):
        name = img_path.stem
        try:
            latent = encode_image(pipe, img_path)
            img_ref, img_recon, txt_ref, txt_recon = diagnose_with_text_drift(
                pipe, latent, enc, pooled, extractor, num_steps=50)
            per_image_img_drift[name] = compute_drift(img_ref, img_recon)
            per_image_txt_drift[name] = compute_drift(txt_ref, txt_recon)
            itop = max(per_image_img_drift[name], key=per_image_img_drift[name].get)
            ttop = max(per_image_txt_drift[name], key=per_image_txt_drift[name].get)
            print(f"  {name}: img_peak={itop} txt_peak={ttop}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    if not per_image_img_drift:
        print("No results!")
        return

    # --- Aggregate ---
    block_names = sorted(per_image_img_drift[list(per_image_img_drift.keys())[0]].keys(),
                        key=lambda x: int(x.split("_")[1]))

    def aggregate(drift_dict, block_names):
        agg = {}
        for bn in block_names:
            vals = [drift_dict[img][bn] for img in drift_dict if bn in drift_dict.get(img, {})]
            if vals:
                agg[bn] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return agg

    img_agg = aggregate(per_image_img_drift, block_names)
    txt_agg = aggregate(per_image_txt_drift, block_names)

    # --- Region analysis ---
    def region_mean(agg, i_range):
        vals = [agg[f"block_{i}"]["mean"] for i in i_range if f"block_{i}" in agg]
        return np.mean(vals) if vals else float('nan')

    regions = {
        "dual_0_12": range(0, 13),
        "boundary_13": [13],
        "mid_14_20": range(14, 21),
        "late_21_23": range(21, 24),
    }

    # --- Validate corrected prediction ---
    # Prediction 1: text drift in dual-attention (0-12) > text drift in standard (14-20)
    txt_dual = region_mean(txt_agg, regions["dual_0_12"])
    txt_mid = region_mean(txt_agg, regions["mid_14_20"])
    txt_boundary = txt_agg["block_13"]["mean"]
    txt_late = region_mean(txt_agg, regions["late_21_23"])

    # Prediction 2: image drift valley at boundary 12-14
    img_dual = region_mean(img_agg, regions["dual_0_12"])
    img_boundary_12_14 = np.mean([img_agg["block_12"]["mean"], img_agg["block_13"]["mean"], img_agg["block_14"]["mean"]])
    img_mid = region_mean(img_agg, regions["mid_14_20"])
    img_late = region_mean(img_agg, regions["late_21_23"])

    # Is text drift higher in dual-attention than standard joint?
    text_cross_modal_effect = (txt_dual - txt_mid) / max(txt_dual, txt_mid, 1e-10)
    # Is image drift lowest at the boundary?
    min_region = min(img_dual, img_boundary_12_14, img_mid, img_late)
    img_boundary_is_valley = (img_boundary_12_14 <= min_region * 1.05)

    print("\n" + "=" * 70)
    print("TEXT DRIFT REGION ANALYSIS")
    print("=" * 70)
    print(f"  Text drift:")
    print(f"    dual (0-12):     {txt_dual:.6f}")
    print(f"    boundary (13):    {txt_boundary:.6f}")
    print(f"    mid (14-20):      {txt_mid:.6f}")
    print(f"    late (21-23):     {txt_late:.6f}")
    print(f"    dual vs mid ratio: {txt_dual/txt_mid:.2f}x")
    print(f"    cross-modal effect: {text_cross_modal_effect:+.3f}")
    print()
    print(f"  Image drift:")
    print(f"    dual (0-12):     {img_dual:.6f}")
    print(f"    boundary (12-14): {img_boundary_12_14:.6f}")
    print(f"    mid (14-20):      {img_mid:.6f}")
    print(f"    late (21-23):     {img_late:.6f}")

    # --- Print all per-block ---
    print(f"\n  {'Block':<10} {'Img Drift':>10} {'Txt Drift':>10} {'dual/ std':<8}")
    print(f"  {'-'*40}")
    img_ranked = sorted(img_agg, key=lambda x: img_agg[x]["mean"], reverse=True)
    for bn in block_names:
        idx = int(bn.split("_")[1])
        mode = "dual" if idx <= 12 else "std"
        if bn in img_agg and bn in txt_agg:
            print(f"  {bn:<10} {img_agg[bn]['mean']:>10.6f} {txt_agg[bn]['mean']:>10.6f} {mode:<8}")
        elif bn in img_agg:
            print(f"  {bn:<10} {img_agg[bn]['mean']:>10.6f} {'--':>10} {mode:<8}")

    # --- Prediction validation ---
    print("\n" + "=" * 70)
    print("PREDICTION VALIDATION")
    print("=" * 70)
    print(f"  P1: text drift(dual 0-12) > text drift(mid 14-20): "
          f"{'CONFIRMED' if txt_dual > txt_mid else 'FALSIFIED'}")
    print(f"       {txt_dual:.6f} vs {txt_mid:.6f} (ratio {txt_dual/txt_mid:.2f}x)")
    print(f"  P2: image drift valley at boundary 12-14: "
          f"{'CONFIRMED' if img_boundary_is_valley else 'FALSIFIED'}")
    print(f"       boundary={img_boundary_12_14:.6f} vs min_other={min(img_dual, img_mid, img_late):.6f}")

    # Save
    summary = {
        "prediction_v1": PREDICTION_V1,
        "prediction_v2": PREDICTION_V2,
        "prediction_validation": {
            "text_cross_modal_effect": float(text_cross_modal_effect),
            "text_dual_gt_mid": bool(txt_dual > txt_mid),
            "text_dual_mean": float(txt_dual),
            "text_mid_mean": float(txt_mid),
            "image_boundary_is_valley": bool(img_boundary_is_valley),
            "image_boundary_mean": float(img_boundary_12_14),
        },
        "image_drift": img_agg,
        "text_drift": txt_agg,
        "per_image_img_drift": {k: {b: float(v) for b, v in d.items()} for k, d in per_image_img_drift.items()},
        "per_image_txt_drift": {k: {b: float(v) for b, v in d.items()} for k, d in per_image_txt_drift.items()},
        "config": {"num_steps": 50, "prompt": prompt},
    }
    with open(OUT_DIR / "prediction_record.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'prediction_record.json'}")


if __name__ == "__main__":
    main()
