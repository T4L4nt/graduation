"""
Phase 5: Failure case analysis and lambda stability.

1. Failure cases: identify images where correction provides minimal benefit
2. Lambda stability: verify that λ=0.7 is robust across images

Usage:
  python scripts/phase5_failure_lambda.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/phase5_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COCO_IMAGES_19 = sorted([
    f"coco_000000000139", f"coco_000000000285", f"coco_000000000632",
    f"coco_000000000724", f"coco_000000000776", f"coco_000000000785",
    f"coco_000000000802", f"coco_000000000872", f"coco_000000000885",
    f"coco_000000001000", f"coco_000000001353", f"coco_000000001490",
    f"coco_000000001532", f"coco_000000001584", f"coco_000000001675",
    f"coco_000000001818", f"coco_000000002153", f"coco_000000002261",
    f"coco_000000002532",
])


def load_per_image_delta():
    """Compute per-image ΔPSNR from Phase 2 full data."""
    with open("outputs/phase2_full/metrics.json") as f:
        data = json.load(f)

    per_image = {}
    for img in COCO_IMAGES_19:
        ddim_psnr = None
        corr_psnr = None
        for x in data:
            if x["image"] != img or x.get("steps") != 50:
                continue
            lmbda = str(x.get("lambda", ""))
            sched = x.get("scheduler", "none")
            if lmbda == "baseline" and sched == "none":
                ddim_psnr = x["PSNR"]
            elif lmbda == "0.7" and sched == "constant":
                corr_psnr = x["PSNR"]

        if ddim_psnr is not None and corr_psnr is not None:
            per_image[img] = {
                "ddim_psnr": ddim_psnr,
                "corr_psnr": corr_psnr,
                "delta": corr_psnr - ddim_psnr,
            }
    return per_image


def analyze_failure_cases(per_image):
    """Identify and analyze failure cases (where correction provides minimal benefit)."""
    print("=" * 60)
    print("FAILURE CASE ANALYSIS")
    print("=" * 60)

    # Sort by delta (worst first)
    sorted_imgs = sorted(per_image.items(), key=lambda x: x[1]["delta"])

    print("\nTop-5 failure cases (lowest ΔPSNR):")
    print(f"{'Image':<25s} {'DDIM':>6s} {'Corr':>6s} {'Δ':>7s}")
    print("-" * 45)
    for img, info in sorted_imgs[:5]:
        print(f"{img:<25s} {info['ddim_psnr']:>6.2f} {info['corr_psnr']:>6.2f} {info['delta']:>+7.2f}")

    print("\nTop-5 success cases (highest ΔPSNR):")
    print(f"{'Image':<25s} {'DDIM':>6s} {'Corr':>6s} {'Δ':>7s}")
    print("-" * 45)
    for img, info in sorted_imgs[-5:]:
        print(f"{img:<25s} {info['ddim_psnr']:>6.2f} {info['corr_psnr']:>6.2f} {info['delta']:>+7.2f}")

    # Pattern analysis: does baseline PSNR predict delta?
    baselines = [v["ddim_psnr"] for v in per_image.values()]
    deltas = [v["delta"] for v in per_image.values()]

    corr = np.corrcoef(baselines, deltas)[0, 1]
    print(f"\nCorrelation(DDIM_PSNR, ΔPSNR) = {corr:.4f}")
    if corr < -0.3:
        print("  → FAILURE PATTERN: Correction helps MORE when baseline is WORSE.")
        print("    High-DDIM images have less room for improvement.")
    elif corr > 0.3:
        print("  → FAILURE PATTERN: Correction helps MORE when baseline is BETTER.")
        print("    Low-DDIM images are too degraded for correction to help.")
    else:
        print("  → No strong linear relationship between baseline quality and correction benefit.")

    # Generate failure case figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: sorted bar chart
    ax = axes[0]
    img_labels = [img.replace("coco_0000000", "") for img, _ in sorted_imgs]
    delta_vals = [info["delta"] for _, info in sorted_imgs]
    colors = ["#e74c3c" if d < 0.5 else "#f39c12" if d < 1.5 else "#2ecc71" for d in delta_vals]

    bars = ax.bar(range(len(delta_vals)), delta_vals, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=np.mean(delta_vals), color="blue", linestyle="--", alpha=0.5,
               label=f"Mean Δ={np.mean(delta_vals):.2f} dB")
    ax.set_xticks(range(len(delta_vals)))
    ax.set_xticklabels(img_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔPSNR (dB)", fontsize=12)
    ax.set_title("Per-Image Correction Benefit (sorted)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: baseline vs delta scatter
    ax = axes[1]
    ax.scatter(baselines, deltas, c="#3498db", s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("DDIM Baseline PSNR (dB)", fontsize=12)
    ax.set_ylabel("ΔPSNR from Correction (dB)", fontsize=12)
    ax.set_title(f"Baseline vs Correction Benefit (r={corr:.3f})", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Highlight failure region
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.3, label="Δ=1.0 (low benefit)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "failure_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] failure_analysis.png")

    # Summary statistics
    deltas_arr = np.array(deltas)
    print(f"\nΔPSNR distribution: mean={deltas_arr.mean():.2f}, std={deltas_arr.std():.2f}, "
          f"min={deltas_arr.min():.2f}, max={deltas_arr.max():.2f}")
    print(f"Images with Δ < 1.0 dB: {sum(deltas_arr < 1.0)}/{len(deltas_arr)}")
    print(f"Images with Δ > 5.0 dB: {sum(deltas_arr > 5.0)}/{len(deltas_arr)}")


def analyze_lambda_stability():
    """Check if λ=0.7 is robust across images and alternative lambdas."""
    print("\n" + "=" * 60)
    print("LAMBDA STABILITY ANALYSIS")
    print("=" * 60)

    with open("outputs/phase2_full/metrics.json") as f:
        data = json.load(f)

    # Check what lambdas are available
    all_lambdas = sorted(set(str(x.get("lambda", "")) for x in data
                             if x.get("steps") == 50 and str(x.get("lambda", "")) != "baseline"))
    print(f"Available λ values at step=50: {all_lambdas}")

    # Get per-image results at different lambdas
    per_lambda = {}
    for x in data:
        img = x["image"]
        if img not in COCO_IMAGES_19:
            continue
        if x.get("steps") != 50:
            continue
        lmbda = str(x.get("lambda", ""))
        sched = x.get("scheduler", "none")
        if lmbda == "baseline":
            continue
        if sched != "constant":
            continue
        if lmbda not in per_lambda:
            per_lambda[lmbda] = {}
        per_lambda[lmbda][img] = x["PSNR"]

    if len(per_lambda) <= 1:
        print("Only one lambda value available — checking phase2 lambda_tuning data...")
        # Try lambda_tuning.json
        tuning_path = Path("outputs/phase2_full/lambda_tuning.json")
        if tuning_path.exists():
            with open(tuning_path) as f:
                tuning = json.load(f)
            print(f"Lambda tuning data: {len(tuning)} entries")
            # Check structure
            if tuning:
                print(f"First entry keys: {list(tuning[0].keys()) if isinstance(tuning, list) else list(tuning.keys())[:5]}")
        return

    # For each lambda, compute mean PSNR across images
    print(f"\nPer-λ summary (step=50):")
    print(f"{'λ':>6s}  {'PSNR':>7s}  {'vs λ=0.7':>9s}")
    print("-" * 25)
    ref = per_lambda.get("0.7", {})
    ref_mean = np.mean(list(ref.values())) if ref else 0

    for lmbda in sorted(per_lambda.keys(), key=float):
        vals = list(per_lambda[lmbda].values())
        mean_val = np.mean(vals)
        diff = mean_val - ref_mean if ref else 0
        marker = " <-- best" if mean_val == max(np.mean(list(per_lambda[lm].values())) for lm in per_lambda) else ""
        print(f"{float(lmbda):>5.1f}  {mean_val:>7.2f}  {diff:>+9.2f}{marker}")

    # Per-image lambda preference
    print("\nPer-image best lambda:")
    best_counts = {}
    for img in COCO_IMAGES_19:
        best_l = None
        best_p = -1
        for lmbda in per_lambda:
            if img in per_lambda[lmbda]:
                if per_lambda[lmbda][img] > best_p:
                    best_p = per_lambda[lmbda][img]
                    best_l = lmbda
        if best_l:
            best_counts[best_l] = best_counts.get(best_l, 0) + 1

    for lmbda in sorted(best_counts.keys(), key=float):
        print(f"  λ={float(lmbda):.1f} is best for {best_counts[lmbda]} images")


def main():
    print("=" * 60)
    print("Phase 5: Failure Case & Lambda Stability Analysis")
    print("=" * 60)

    # 1. Failure cases
    per_image = load_per_image_delta()
    analyze_failure_cases(per_image)

    # 2. Lambda stability
    analyze_lambda_stability()

    print(f"\nDone! Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
