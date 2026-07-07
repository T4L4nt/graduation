#!/usr/bin/env python
"""
Phase 6a: FLUX.1-dev per-block feature drift diagnosis.

Mirrors Phase 1 (SD 1.5 UNet 196-layer diagnosis) but for FLUX's 57
transformer blocks (19 joint + 38 single).

Key question: Does feature drift exist in flow matching models?
If so, what is its structural fingerprint?

Usage:
    python scripts/flux_phase6_diagnosis.py
    python scripts/flux_phase6_diagnosis.py --num-steps 20 --num-images 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from flux_common import (
    load_flux_pipeline,
    flux_invert,
    FluxFeatureExtractor,
    compute_block_drift,
    compute_metrics,
    apply_correction_latent,
    OUTPUT_DIR,
    DATA_DIR,
    N_JOINT_BLOCKS,
    N_SINGLE_BLOCKS,
)


def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 6a: Drift Diagnosis")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=19)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--offload-t5", action="store_true",
                        help="Offload T5-XXL to CPU (saves ~10GB VRAM)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load images
    image_paths = sorted(DATA_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {DATA_DIR}")
        image_paths = sorted(DATA_DIR.glob("*.png"))
    if not image_paths:
        print(f"No images found in {DATA_DIR}")
        sys.exit(1)

    image_paths = image_paths[: args.num_images]
    print(f"Using {len(image_paths)} images from {DATA_DIR}")

    # Load model
    pipe = load_flux_pipeline(device=args.device, offload_t5=args.offload_t5)

    # -------------------------------------------------------------------
    # Part 1: Per-block drift diagnosis (1 image, full feature extraction)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 1: Per-block drift diagnosis")
    print("=" * 60)

    img0 = Image.open(image_paths[0]).convert("RGB")
    print(f"Diagnosis image: {image_paths[0].name} ({img0.size})")

    extractor = FluxFeatureExtractor(pipe.transformer)

    # Inversion with feature hooks
    print("\nRunning inversion with feature hooks...")
    out_inv = flux_invert(
        pipe, img0, num_steps=args.num_steps, extract_features=True
    )

    drift = compute_block_drift(out_inv["features_inv"], out_inv["features_recon"])

    # Rank blocks by hidden (image token) drift
    sorted_by_hidden = sorted(
        drift.items(), key=lambda x: x[1]["hidden_drift"], reverse=True
    )
    sorted_by_encoder = sorted(
        [x for x in drift.items() if x[0].startswith("joint")],
        key=lambda x: x[1]["encoder_drift"],
        reverse=True,
    )

    print(f"\n{'='*60}")
    print("Top-20 blocks by image token drift:")
    print(f"{'Block':<20s} {'hidden_drift':>14s} {'encoder_drift':>14s}")
    print("-" * 50)
    for name, d in sorted_by_hidden[:20]:
        hd = d["hidden_drift"]
        ed = d.get("encoder_drift", 0.0)
        print(f"{name:<20s} {hd:>14.6f} {ed:>14.6f}")

    print(f"\nJoint blocks ranked by text token drift:")
    print(f"{'Block':<20s} {'encoder_drift':>14s} {'hidden_drift':>14s}")
    print("-" * 50)
    for name, d in sorted_by_encoder[:10]:
        print(f"{name:<20s} {d['encoder_drift']:>14.6f} {d['hidden_drift']:>14.6f}")

    # Compute group statistics
    joint_hidden_drifts = [
        d["hidden_drift"] for n, d in drift.items() if n.startswith("joint")
    ]
    single_hidden_drifts = [
        d["hidden_drift"] for n, d in drift.items() if n.startswith("single")
    ]
    joint_encoder_drifts = [
        d["encoder_drift"] for n, d in drift.items() if n.startswith("joint")
    ]

    print(f"\nGroup statistics (hidden/image token drift):")
    print(f"  Joint blocks  (n={len(joint_hidden_drifts)}): "
          f"mean={np.mean(joint_hidden_drifts):.6f}, max={np.max(joint_hidden_drifts):.6f}")
    print(f"  Single blocks (n={len(single_hidden_drifts)}): "
          f"mean={np.mean(single_hidden_drifts):.6f}, max={np.max(single_hidden_drifts):.6f}")
    print(f"  Joint text token drift: "
          f"mean={np.mean(joint_encoder_drifts):.6f}, max={np.max(joint_encoder_drifts):.6f}")

    # -------------------------------------------------------------------
    # Part 2: Latent-space correction evaluation (all images)
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PART 2: Latent-space correction evaluation")
    print("=" * 60)

    metrics_no_corr = []
    metrics_corr = []

    for img_path in tqdm(image_paths, desc="Evaluating"):
        img = Image.open(img_path).convert("RGB")

        # Without correction
        out = flux_invert(pipe, img, num_steps=args.num_steps)
        m = compute_metrics(img, out["image_recon"])
        m["image"] = img_path.name
        metrics_no_corr.append(m)

        # With latent-space correction
        z_inv = out["z_0"]            # raw initial VAE latent
        z_recon = out["z_0_recon_raw"]  # raw reconstructed VAE latent
        z_corrected = apply_correction_latent(z_recon, z_inv, lam=args.lam)

        z_corrected = z_corrected.to(pipe.device, dtype=pipe.vae.dtype)
        z_corrected = (
            z_corrected / pipe.vae.config.scaling_factor
            + pipe.vae.config.shift_factor
        )

        with torch.no_grad():
            img_corr = pipe.vae.decode(z_corrected, return_dict=False)[0]
            img_corr = pipe.image_processor.postprocess(img_corr, output_type="pil")[0]

        # Save first 3 reconstructions for visual inspection
        idx = image_paths.index(img_path)
        if idx < 3:
            img.save(OUTPUT_DIR / f"{img_path.stem}_original.png")
            out["image_recon"].save(OUTPUT_DIR / f"{img_path.stem}_recon_nocorr.png")
            img_corr.save(OUTPUT_DIR / f"{img_path.stem}_recon_corr.png")

        m_corr = compute_metrics(img, img_corr)
        m_corr["image"] = img_path.name
        metrics_corr.append(m_corr)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY (n={len(image_paths)}, steps={args.num_steps}, λ={args.lam})")
    print(f"{'='*60}")

    summary = {}
    for key in ["PSNR", "SSIM", "LPIPS"]:
        vals = [m[key] for m in metrics_no_corr]
        summary[f"{key}_nocorr_mean"] = round(np.mean(vals), 3)
        summary[f"{key}_nocorr_std"] = round(np.std(vals), 3)

        vals_c = [m[key] for m in metrics_corr]
        summary[f"{key}_corr_mean"] = round(np.mean(vals_c), 3)
        summary[f"{key}_corr_std"] = round(np.std(vals_c), 3)

        print(f"{key}: {summary[f'{key}_nocorr_mean']:.3f} ± {summary[f'{key}_nocorr_std']:.3f} "
              f"-> {summary[f'{key}_corr_mean']:.3f} ± {summary[f'{key}_corr_std']:.3f}")

    delta = summary["PSNR_corr_mean"] - summary["PSNR_nocorr_mean"]
    summary["delta_PSNR"] = round(delta, 3)
    print(f"\nΔPSNR = {delta:+.2f} dB")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    drift_json = {}
    for name, d in drift.items():
        drift_json[name] = {
            "hidden_drift": d["hidden_drift"],
            "encoder_drift": d["encoder_drift"],
            "block_type": "joint" if name.startswith("joint") else "single",
            "block_index": int(name.split("_")[1]),
        }

    result = {
        "args": vars(args),
        "drift": drift_json,
        "drift_top20": [
            {"name": n, **d} for n, d in sorted_by_hidden[:20]
        ],
        "drift_statistics": {
            "joint_hidden_mean": float(np.mean(joint_hidden_drifts)),
            "joint_hidden_max": float(np.max(joint_hidden_drifts)),
            "single_hidden_mean": float(np.mean(single_hidden_drifts)),
            "single_hidden_max": float(np.max(single_hidden_drifts)),
            "joint_encoder_mean": float(np.mean(joint_encoder_drifts)),
            "joint_encoder_max": float(np.max(joint_encoder_drifts)),
        },
        "metrics_no_corr": metrics_no_corr,
        "metrics_corr": metrics_corr,
        "summary": summary,
    }

    out_path = OUTPUT_DIR / "diagnosis_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {out_path}")

    return result


if __name__ == "__main__":
    main()
