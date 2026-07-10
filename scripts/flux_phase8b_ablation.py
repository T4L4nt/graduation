#!/usr/bin/env python
"""
Phase 8b: FLUX layer group ablation + lambda scan.
Uses flux_common.run_correction_feature for robust pipeline handling.

Usage: python scripts/flux_phase8b_ablation.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from flux_common import load_flux_pipeline, run_correction_feature

OUT_DIR = Path("outputs/phase8b_flux")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = sorted(Path("data/coco_val").glob("*.jpg"))[:19]
N_JOINT = 19
N_SINGLE = 38


def psnr_from_result(result, key="recon"):
    """Extract PSNR from run_correction_feature result."""
    m = result.get(f"metrics_{key}", result.get("metrics", {}))
    return m.get("PSNR", float("nan"))


def main():
    print("=" * 70)
    print("Phase 8b: FLUX Layer Group Ablation + Lambda Scan")
    print("=" * 70)

    # Load pipeline
    print("Loading FLUX.1-dev...")
    pipe = load_flux_pipeline(offload_t5=True)
    # No CPU offload — let run_correction_feature manage GPU directly
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    NUM_STEPS = 28  # FLUX default

    # === Part 1: Lambda Scan (5 images) ===
    print("\n" + "=" * 70)
    print("Part 1: Lambda Scan (5 images)")
    print("=" * 70)

    lam_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    lam_results = {lam: [] for lam in lam_values}
    baseline_psnrs = []

    for i, img_path in enumerate(tqdm(TEST_IMAGES[:5], desc="λ scan")):
        try:
            img = Image.open(img_path).convert("RGB")

            # Baseline: run with lam_hidden=0 (no correction)
            result = run_correction_feature(pipe, img, num_steps=NUM_STEPS,
                                           prompt="", lam_hidden=0.0, seed=42)
            baseline_psnrs.append(psnr_from_result(result, "recon"))

            for lam in lam_values:
                result = run_correction_feature(pipe, img, num_steps=NUM_STEPS,
                                               prompt="", lam_hidden=lam, seed=42)
                lam_results[lam].append(psnr_from_result(result, "corrected"))

        except Exception as e:
            print(f"  {img_path.name}: ERROR {e}")
        torch.cuda.empty_cache()

    base_mean = np.mean(baseline_psnrs) if baseline_psnrs else float("nan")
    print(f"\n  Baseline PSNR: {base_mean:.2f} dB")
    best_delta = -999
    best_lam = 0.7
    for lam in lam_values:
        vals = lam_results[lam]
        if vals:
            delta = np.mean(vals) - base_mean
            print(f"  λ={lam}: PSNR={np.mean(vals):.2f} ± {np.std(vals):.2f}, Δ=+{delta:.2f}")
            if delta > best_delta:
                best_delta = delta
                best_lam = lam
    print(f"  Best λ: {best_lam}")

    # === Part 2: Layer Group Ablation (all 19 images) ===
    print("\n" + "=" * 70)
    print("Part 2: Layer Group Ablation (19 images)")
    print("=" * 70)

    # Get top-5 drift blocks
    diag_path = Path("outputs/phase6_flux/diagnosis_summary.json")
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        drift_vals = {n: diag["drift"][n]["hidden_drift"] for n in diag["drift"]}
        ranked = sorted(drift_vals, key=lambda x: drift_vals[x], reverse=True)
        top5_names = ranked[:5]
        top5_joints = [int(n.replace("joint_", "")) for n in top5_names if "joint" in n]
        top5_singles = [int(n.replace("single_", "")) for n in top5_names if "single" in n]
    else:
        top5_names = ["joint_18", "single_2", "single_0", "single_1", "single_3"]
        top5_joints = [18]
        top5_singles = [2, 0, 1, 3]

    print(f"  Top-5 drift: {top5_names}")
    print(f"  Using λ = {best_lam}")

    # Each ablation condition = a specific layer group
    conditions = [
        ("baseline", None, []),
        ("latent_only", list(range(N_JOINT)), list(range(N_SINGLE))),
        ("joint_only", list(range(N_JOINT)), []),
        ("single_only", [], list(range(N_SINGLE))),
        ("early_single", [], list(range(0, 19))),
        ("late_single", [], list(range(19, 38))),
        ("top5", top5_joints, top5_singles),
        ("joint_plus_early", list(range(N_JOINT)), list(range(0, 19))),
    ]

    ablation_results = {name: [] for name, _, _ in conditions}

    for img_path in tqdm(TEST_IMAGES, desc="Ablation"):
        try:
            img = Image.open(img_path).convert("RGB")

            for cond_name, joints, singles in conditions:
                try:
                    if cond_name == "baseline":
                        result = run_correction_feature(pipe, img, num_steps=NUM_STEPS,
                                                       prompt="", lam_hidden=0.0, seed=42)
                        ablation_results[cond_name].append(
                            psnr_from_result(result, "recon"))
                    else:
                        result = run_correction_feature(
                            pipe, img, num_steps=NUM_STEPS, prompt="",
                            lam_hidden=best_lam,
                            joint_indices=joints, single_indices=singles,
                            seed=42)
                        ablation_results[cond_name].append(
                            psnr_from_result(result, "corrected"))
                except Exception as e2:
                    pass  # Skip individual condition failures

        except Exception as e:
            print(f"  {img_path.name}: ERROR {e}")
        torch.cuda.empty_cache()

    # --- Print results ---
    base = np.nanmean(ablation_results["baseline"])
    print(f"\n  Baseline PSNR: {base:.2f} dB")
    for cond_name, _, _ in conditions[1:]:  # skip baseline
        vals = ablation_results[cond_name]
        if len(vals) == 0:
            continue
        m = np.nanmean(vals); delta = m - base
        print(f"  {cond_name:25s}: PSNR={m:.2f}, Δ=+{delta:.2f}")

    # Key comparisons
    s_vals = ablation_results["single_only"]
    j_vals = ablation_results["joint_only"]
    if s_vals and j_vals:
        s_m = np.nanmean(s_vals); j_m = np.nanmean(j_vals)
        s_d = s_m - base; j_d = j_m - base
        print(f"\n  single-only Δ=+{s_d:.2f}  joint-only Δ=+{j_d:.2f}")
        print(f"  single/joint ratio: {s_d/j_d:.2f}x" if j_d > 0 else "")
        print(f"  Prediction (single > joint): {'CONFIRMED' if s_d > j_d else 'FALSIFIED'}")

    e_vals = ablation_results["early_single"]
    l_vals = ablation_results["late_single"]
    if e_vals and l_vals:
        print(f"  early-single Δ=+{np.nanmean(e_vals)-base:.2f}  "
              f"late-single Δ=+{np.nanmean(l_vals)-base:.2f}")

    # Save
    output = {
        "config": {"num_steps": NUM_STEPS, "best_lambda": float(best_lam)},
        "lambda_scan": {
            "baseline_psnr": float(base_mean),
            "results": {str(lam): {"mean": float(np.nanmean(v)) if v else None}
                       for lam, v in lam_results.items()},
        },
        "ablation": {
            "baseline_psnr": float(base),
            "conditions": {name: {"n": len(v), "mean": float(np.nanmean(v)) if v else None}
                          for name, v in ablation_results.items()},
        },
    }
    with open(OUT_DIR / "ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
