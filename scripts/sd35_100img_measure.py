#!/usr/bin/env python
"""
SD 3.5 Medium — 100-image remeasurement for bootstrap + paired test.
Uses same protocol as sd35_phase1_diagnostics.py:
  - Euler 50-step inversion/reconstruction
  - Feature drift measured at t=0 (clean latent)
  - Normalized L2: ||f_recon - f_ref|| / max(||f_ref||, 1e-10)

Output: outputs/sd35_phase1/layer_drift_summary_100img.json
"""

import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

OUT_DIR = Path("outputs/sd35_phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/home/hiaskc/.cache/huggingface/hub/models/stabilityai--stable-diffusion-3.5-medium/snapshots/master"
DATA_DIR = Path("data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
N_STEPS = 50

print(f"SD3.5 100-image measurement: {len(IMAGES)} images, {N_STEPS} Euler steps")


class BlockFeatureExtractor:
    def __init__(self):
        self.features = {}
        self._hooks = []

    def _hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                hs = output[1]  # image hidden_states for MMDiT
            else:
                hs = output
            self.features[name] = hs.detach().float().cpu()
        return fn

    def register(self, transformer):
        for i, block in enumerate(transformer.transformer_blocks):
            self._hooks.append(block.register_forward_hook(self._hook_fn(f"block_{i}")))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.features.clear()


def load_pipeline():
    from diffusers import StableDiffusion3Pipeline
    print("Loading SD 3.5 Medium (fp16, cpu_offload)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        local_files_only=True, tokenizer_3=None, text_encoder_3=None,
    )
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    return pipe


def encode_empty_prompt(pipe):
    device = pipe._execution_device
    joint_dim = pipe.transformer.config.joint_attention_dim

    tok1 = pipe.tokenizer("", padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt")
    with torch.no_grad():
        out1 = pipe.text_encoder(tok1["input_ids"].to(device), output_hidden_states=True)
    h1 = out1.hidden_states[-2].to(dtype=pipe.text_encoder.dtype)
    p1 = out1.text_embeds.to(dtype=pipe.text_encoder.dtype)

    tok2 = pipe.tokenizer_2("", padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
    with torch.no_grad():
        out2 = pipe.text_encoder_2(tok2["input_ids"].to(device), output_hidden_states=True)
    h2 = out2.hidden_states[-2].to(dtype=pipe.text_encoder_2.dtype)
    p2 = out2.text_embeds.to(dtype=pipe.text_encoder_2.dtype)

    clip_hidden = torch.cat([h1, h2], dim=-1)
    clip_padded = torch.nn.functional.pad(clip_hidden, (0, joint_dim - clip_hidden.shape[-1]))
    t5_embed = torch.zeros(1, 256, joint_dim, device=device, dtype=pipe.transformer.dtype)
    encoder_hidden_states = torch.cat([clip_padded, t5_embed], dim=1).to(pipe.transformer.dtype)
    pooled = torch.cat([p1, p2], dim=-1).to(pipe.transformer.dtype)
    return encoder_hidden_states, pooled


@torch.no_grad()
def measure_one(pipe, img_path, encoder_hidden_states, pooled, extractor):
    img = Image.open(img_path).convert("RGB").resize((1024, 1024))
    pixel_values = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
    latent = pipe.vae.encode(pixel_values).latent_dist.sample()
    latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    scheduler = pipe.scheduler
    scheduler.set_timesteps(N_STEPS, device="cuda")
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    t0 = torch.zeros(latent.shape[0], device="cuda", dtype=torch.float32)

    # Reference at t=0
    extractor.register(pipe.transformer)
    _ = pipe.transformer(hidden_states=latent, encoder_hidden_states=encoder_hidden_states,
                         pooled_projections=pooled, timestep=t0, return_dict=False)[0]
    feat_ref = dict(extractor.features)
    extractor.clear()

    # Euler inversion: x_0 → x_T
    z = latent.clone()
    for i, t in enumerate(timesteps):
        dt = sigmas[i+1] - sigmas[i]
        v = pipe.transformer(hidden_states=z, encoder_hidden_states=encoder_hidden_states,
                             pooled_projections=pooled,
                             timestep=t.unsqueeze(0).expand(z.shape[0]), return_dict=False)[0]
        z = z + dt * v

    # Euler reconstruction: x_T → x_0'
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
    feat_recon = dict(extractor.features)
    extractor.remove()
    extractor.clear()

    # Normalized drift
    drift = {}
    for name in sorted(feat_ref.keys(), key=lambda x: int(x.split("_")[1])):
        d = (feat_recon[name] - feat_ref[name]).norm().item()
        n = feat_ref[name].norm().item()
        drift[name] = d / max(n, 1e-10)
    return drift


def main():
    pipe = load_pipeline()
    extractor = BlockFeatureExtractor()
    encoder_hidden_states, pooled = encode_empty_prompt(pipe)

    per_image = {}
    failed = 0
    for img_path in tqdm(IMAGES, desc="SD3.5 100img"):
        try:
            drift = measure_one(pipe, img_path, encoder_hidden_states, pooled, extractor)
            per_image[img_path.stem] = drift
        except Exception as e:
            failed += 1
            print(f"  FAIL {img_path.stem}: {e}")

    del pipe; torch.cuda.empty_cache()

    print(f"\nDone: {len(per_image)}/{len(IMAGES)} images, {failed} failed")

    # Aggregate
    block_names = sorted(per_image[list(per_image.keys())[0]].keys(),
                         key=lambda x: int(x.split("_")[1]))
    drift_agg = {}
    per_raw = {}
    for bn in block_names:
        vals = [per_image[img][bn] for img in per_image]
        drift_agg[bn] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                          "min": float(np.min(vals)), "max": float(np.max(vals))}
        per_raw[bn] = vals

    ranking = sorted(drift_agg, key=lambda x: drift_agg[x]["mean"], reverse=True)
    peak_block = ranking[0]
    peak_idx = int(peak_block.split("_")[1])

    # Bootstrap argmax distribution
    np.random.seed(42)
    B = 10000
    n_img = len(per_image)
    argmax_dist = np.zeros(24, dtype=int)
    for _ in range(B):
        idx = np.random.choice(n_img, size=n_img, replace=True)
        boot_means = np.array([np.mean([per_image[list(per_image.keys())[j]][f"block_{i}"]
                                        for j in idx]) for i in range(24)])
        argmax_dist[boot_means.argmax()] += 1

    # Paired test block_22 vs block_23
    d22 = np.array([per_image[img]["block_22"] for img in per_image])
    d23 = np.array([per_image[img]["block_23"] for img in per_image])
    d21 = np.array([per_image[img]["block_21"] for img in per_image])
    from scipy import stats
    t22_23, p22_23 = stats.ttest_rel(d22, d23)
    d22_23 = d22.mean() - d23.mean()
    d22_21 = d22.mean() - d21.mean()
    cohen = d22_23 / np.sqrt((d22.var() + d23.var()) / 2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SD3.5 100-image validation results")
    print(f"{'='*70}")
    print(f"  Images: {len(per_image)}")
    print(f"  Peak: {peak_block} (idx={peak_idx})")
    print(f"  Block_22 mean={d22.mean():.6f} ± {d22.std(ddof=1):.6f}")
    print(f"  Block_23 mean={d23.mean():.6f} ± {d23.std(ddof=1):.6f}")
    print(f"  Δ(22-23): {d22_23:.6f}  Δ(22-21): {d22_21:.6f}")
    print(f"  t-test(22vs23): t={t22_23:.4f}, p={p22_23:.8f}")
    print(f"  Cohen d(22vs23): {cohen:.4f}")
    print(f"  block_22 > block_23: {(d22>d23).sum()}/{len(d22)} images")
    print(f"\n  Bootstrap argmax distribution (B={B}, n={n_img}):")
    for i in range(24):
        pct = 100 * argmax_dist[i] / B
        if pct > 0:
            bar = "█" * int(pct)
            print(f"    block_{i:2d}: {argmax_dist[i]:5d}  ({pct:5.1f}%) {bar}")

    # Save
    summary = {
        "n_images": len(per_image),
        "images": sorted(per_image.keys()),
        "steps": N_STEPS,
        "protocol": "Euler inversion/reconstruction, drift=||f_recon-f_ref||/||f_ref|| at t=0",
        "peak": {"layer": peak_block, "index": peak_idx, "position": peak_idx/24},
        "bootstrap": {"B": B, "n_samples": n_img,
                      "argmax_distribution": {f"block_{i}": int(argmax_dist[i]) for i in range(24)}},
        "paired_test_22_23": {
            "mean_diff": float(d22_23), "t_stat": float(t22_23), "p_value": float(p22_23),
            "cohens_d": float(cohen), "n_22_gt_23": int((d22 > d23).sum()), "n_total": len(d22)
        },
        "aggregated": drift_agg,
        "ranking": ranking,
        "top5": ranking[:5],
        "per_image": {img: {bn: float(v) for bn, v in d.items()} for img, d in per_image.items()},
    }
    out_path = OUT_DIR / "layer_drift_summary_100img.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
