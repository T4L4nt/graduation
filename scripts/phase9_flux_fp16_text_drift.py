#!/usr/bin/env python
"""
FLUX fp16 text drift re-measurement.

Captures both hidden (image token) AND encoder (text token) drift per block.
Required to replace bf16-era claims about joint_18 text drift jump (0.12→0.44)
with fp16-verified numbers.

Usage:
    python scripts/phase9_flux_fp16_text_drift.py

Output: outputs/phase9_flux_fp16/text_drift.json
"""

import json, sys
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
    print(f"FLUX fp16 text drift: {len(paths)} images, {NUM_STEPS} Euler steps")

    print("Loading FLUX in fp16...")
    pipe = load_flux_pipeline(device=DEVICE, dtype=torch.float16, offload_t5=True)

    all_hidden_drifts = []
    all_encoder_drifts = []

    for p in tqdm(paths, desc="FLUX fp16 text"):
        img = Image.open(str(p)).convert("RGB")
        out = flux_invert(pipe, img, num_steps=NUM_STEPS, extract_features=True)
        drift = compute_block_drift(out["features_inv"], out["features_recon"])

        hidden = {}
        encoder = {}
        for name, d in drift.items():
            hidden[name] = d["hidden_drift"]
            if "encoder_drift" in d:
                encoder[name] = d["encoder_drift"]

        all_hidden_drifts.append(hidden)
        all_encoder_drifts.append(encoder)
        del out
        torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()

    # Sort blocks properly
    def sort_key(name):
        parts = name.split("_")
        return (0 if parts[0] == "joint" else 1, int(parts[1]))

    block_names = sorted(all_hidden_drifts[0].keys(), key=sort_key)

    # Aggregate
    mean_hidden = {}
    mean_encoder = {}
    std_hidden = {}
    std_encoder = {}
    for bn in block_names:
        hv = [d[bn] for d in all_hidden_drifts]
        ev = [d.get(bn, 0.0) for d in all_encoder_drifts]
        mean_hidden[bn] = float(np.mean(hv))
        mean_encoder[bn] = float(np.mean(ev))
        std_hidden[bn] = float(np.std(hv))
        std_encoder[bn] = float(np.std(ev))

    # ---- Analysis ----
    joint_blocks = [b for b in block_names if "joint" in b]
    single_blocks = [b for b in block_names if "single" in b]

    # Joint block encoder drift
    joint_enc = [(b, mean_encoder[b]) for b in joint_blocks]
    joint_enc_sorted = sorted(joint_enc, key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print("JOINT BLOCK ENCODER (TEXT) DRIFT — fp16")
    print(f"{'='*60}")
    print(f"{'Block':<12} {'encoder_drift':>14} {'hidden_drift':>14}")
    print("-" * 44)
    for b, ev in joint_enc_sorted:
        hv = mean_hidden[b]
        marker = " <-- PEAK" if b == joint_enc_sorted[0][0] else ""
        print(f"{b:<12} {ev:>14.6f} {hv:>14.6f}{marker}")

    # The key question: does text drift jump at joint_18?
    # Compare joint_18 vs mean of joint_0..joint_17
    joint_enc_except_last = [mean_encoder[b] for b in joint_blocks if b != "joint_18"]
    joint_18_enc = mean_encoder["joint_18"]
    mean_other_enc = np.mean(joint_enc_except_last)
    jump_ratio = joint_18_enc / mean_other_enc if mean_other_enc > 0 else float("inf")

    print(f"\n{'='*60}")
    print("TEXT DRIFT JUMP AT joint_18")
    print(f"{'='*60}")
    print(f"joint_18 encoder drift:     {joint_18_enc:.4f}")
    print(f"joint_0..17 mean:           {mean_other_enc:.4f}")
    print(f"jump ratio:                 {jump_ratio:.2f}x")
    print(f"joint_17 encoder drift:     {mean_encoder['joint_17']:.4f}")
    print(f"joint_18 hidden drift:      {mean_hidden['joint_18']:.4f}")

    # Compare with bf16 claims
    print(f"\n{'='*60}")
    print("COMPARISON WITH bf16 ERA CLAIMS")
    print(f"{'='*60}")
    print(f"Old claim: text drift jump 0.12 → 0.44 at joint_18")
    print(f"fp16 result: text drift mean(joint_0..17)={mean_other_enc:.4f} → joint_18={joint_18_enc:.4f}")
    print(f"Jump confirmed: {'YES' if jump_ratio > 2.0 else 'WEAK' if jump_ratio > 1.5 else 'NO'} ({jump_ratio:.1f}x)")

    # Top-5 joint blocks by text drift
    print(f"\nTop-5 joint blocks by text drift (fp16):")
    for b, ev in joint_enc_sorted[:5]:
        print(f"  {b}: encoder={ev:.4f}, hidden={mean_hidden[b]:.4f}")

    # Special token analysis: which text token positions drive the drift?
    # (Would need per-token analysis — flag for protocol documentation)

    # Single blocks: they also have encoder features even though they're
    # single-stream (encoder is passed through unchanged)
    single_with_enc = [(b, mean_encoder[b]) for b in single_blocks if b in mean_encoder]
    if single_with_enc:
        single_enc_mean = np.mean([v for _, v in single_with_enc])
        print(f"\nSingle block text drift (pass-through): mean={single_enc_mean:.6f}")
        print(f"  (expected near 0 — single blocks don't process text)")

    # Save
    output = {
        "model": "FLUX.1-dev",
        "dtype": "float16",
        "num_steps": NUM_STEPS,
        "n_images": N_IMAGES,
        "images": [p.stem for p in paths],
        "n_blocks": len(block_names),
        "blocks": block_names,
        "mean_hidden_drift": mean_hidden,
        "mean_encoder_drift": mean_encoder,
        "std_hidden_drift": std_hidden,
        "std_encoder_drift": std_encoder,
        "text_drift_jump": {
            "joint_18_encoder": joint_18_enc,
            "joint_0_17_mean_encoder": float(mean_other_enc),
            "jump_ratio": float(jump_ratio),
            "joint_17_encoder": float(mean_encoder["joint_17"]),
            "jump_confirmed": bool(jump_ratio > 1.5),
        },
        "measurement_protocol": {
            "features": "FluxFeatureExtractor hooks on all 57 transformer blocks",
            "hidden_drift": "MSE(f_inv.hidden, f_recon.hidden) at turnaround t=1, mean over all image tokens",
            "encoder_drift": "MSE(f_inv.encoder, f_recon.encoder) at turnaround t=1, mean over all text tokens (77 tokens from T5)",
            "normalization": "Raw MSE, no normalization applied",
            "aggregation": f"Mean over {N_IMAGES} images, {NUM_STEPS}-step Euler inversion",
        },
    }

    out_path = OUT_DIR / "text_drift.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
