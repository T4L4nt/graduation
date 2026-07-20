#!/usr/bin/env python
"""
FLUX fp16 re-measurement of per-block drift profile.

Replaces the bf16 measurement in phase6_unified/results.json.
Motivation: fp32 vs fp16 r=0.999998 (lossless), fp32 vs bf16 bias=9.3%.
fp16 is the correct measurement precision — eliminates the 9% bf16 bias.

Runs flux_invert + block drift computation on 5 coco_val100 images.
Saves per-block drift vector for integration into unified fingerprint.

Usage:
    python scripts/phase9_flux_fp16_remeasure.py

Output: outputs/phase9_flux_fp16/
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux_common import load_flux_pipeline, flux_invert, compute_block_drift

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_flux_fp16"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
NUM_STEPS = 28
N_IMAGES = 5


def main():
    paths = sorted(DATA_DIR.glob("coco_*.jpg"))[:N_IMAGES]
    print(f"FLUX fp16 remeasurement: {len(paths)} images, {NUM_STEPS} Euler steps")
    print(f"Images: {[p.stem for p in paths]}")

    # Load FLUX in fp16
    print("\nLoading FLUX in fp16...")
    pipe = load_flux_pipeline(device=DEVICE, dtype=torch.float16, offload_t5=True)

    # Per-image drift accumulation
    all_drifts = []  # list of {block_name: hidden_drift}

    for p in tqdm(paths, desc="FLUX fp16"):
        img = Image.open(str(p)).convert("RGB")

        out = flux_invert(pipe, img, num_steps=NUM_STEPS, extract_features=True)
        drift = compute_block_drift(out["features_inv"], out["features_recon"])

        # Extract hidden (image token) drift per block
        per_block = {name: d["hidden_drift"] for name, d in drift.items()}
        all_drifts.append(per_block)

        # Clean up features
        del out
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    # Aggregate: mean drift per block
    block_names = sorted(all_drifts[0].keys())
    mean_drift = {}
    std_drift = {}
    for bn in block_names:
        vals = [d[bn] for d in all_drifts]
        mean_drift[bn] = float(np.mean(vals))
        std_drift[bn] = float(np.std(vals))

    drift_vec = np.array([mean_drift[bn] for bn in block_names])

    # Print summary
    print(f"\n{'='*60}")
    print(f"FLUX fp16 drift profile ({N_IMAGES} images, {NUM_STEPS} steps)")
    print(f"  Blocks: {len(block_names)} (19 joint + 38 single)")
    print(f"  Mean drift: {drift_vec.mean():.4f}")
    print(f"  Max drift:  {drift_vec.max():.4f} at {block_names[int(np.argmax(drift_vec))]}")
    print(f"  Min drift:  {drift_vec.min():.4f} at {block_names[int(np.argmin(drift_vec))]}")

    # Compare vs bf16 measurement
    with open(PROJECT_ROOT / "outputs" / "phase6_unified" / "results.json") as f:
        unified = json.load(f)
    flux_bf16_data = unified["architectures"]["FLUX (Transformer, Flow Match)"]
    flux_bf16_layers = flux_bf16_data["layers"]
    bf16_drift = np.array([l["drift"] for l in flux_bf16_layers])

    # Map block names to unified format (j0..j18, s0..s37)
    # flux_common uses "joint_0".."joint_18", "single_0".."single_37"
    # phase6_unified uses "j0".."j18", "s0".."s37"
    unified_names = [f"j{i}" for i in range(19)] + [f"s{i}" for i in range(38)]
    if len(unified_names) == len(block_names):
        # Compute correlation and bias
        from scipy.stats import pearsonr
        r, p = pearsonr(drift_vec, bf16_drift)
        bias_pct = float(np.mean(np.abs(drift_vec - bf16_drift)) / np.mean(bf16_drift) * 100)
        print(f"\nComparison with bf16:")
        print(f"  Pearson r = {r:.6f}")
        print(f"  Mean abs bias = {bias_pct:.1f}%")
        print(f"  fp16/bf16 ratio: {drift_vec.mean()/bf16_drift.mean():.3f}")

    # Save
    output = {
        "model": "FLUX.1-dev",
        "dtype": "float16",
        "num_steps": NUM_STEPS,
        "n_images": N_IMAGES,
        "images": [p.stem for p in paths],
        "n_blocks": len(block_names),
        "blocks": block_names,
        "mean_drift": mean_drift,
        "std_drift": std_drift,
        "drift_vector": drift_vec.tolist(),
        "comparison_with_bf16": {
            "pearson_r": float(r) if 'r' in dir() else None,
            "mean_abs_bias_pct": bias_pct if 'bias_pct' in dir() else None,
            "fp16_div_bf16_ratio": float(drift_vec.mean() / bf16_drift.mean()),
        },
    }

    out_path = OUT_DIR / "flux_fp16_drift.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Also generate layer list in unified format for easy integration
    unified_layers = []
    for bn in block_names:
        if bn.startswith("joint_"):
            idx = int(bn.split("_")[1])
            unified_layers.append({"name": f"j{idx}", "drift": mean_drift[bn]})
        elif bn.startswith("single_"):
            idx = int(bn.split("_")[1])
            unified_layers.append({"name": f"s{idx}", "drift": mean_drift[bn]})
    unified_layers.sort(key=lambda x: (0 if x["name"].startswith("j") else 1,
                                          int(x["name"][1:])))

    unified_out = {
        "n_layers": len(unified_layers),
        "drift_min": float(drift_vec.min()),
        "drift_max": float(drift_vec.max()),
        "drift_mean": float(drift_vec.mean()),
        "drift_std": float(drift_vec.std()),
        "layers": unified_layers,
    }
    unified_path = OUT_DIR / "flux_fp16_unified_format.json"
    with open(unified_path, "w") as f:
        json.dump(unified_out, f, indent=2)
    print(f"Saved (unified format): {unified_path}")


if __name__ == "__main__":
    main()
