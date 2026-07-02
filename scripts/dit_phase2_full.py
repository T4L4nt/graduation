"""
DiT Phase 2: 零训练残差校正（HunyuanDiT）

在 HunyuanDiT 上验证 Phase 2 残差校正：
- Lambda 扫描（校正强度）
- 消融实验（不同层组）
- 与 Phase 1 诊断结果联动

核心公式：f_out = f_recon + λ · (f_inv - f_recon)
"""
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dit_phase2_common import (
    DEVICE, DTYPE, MODEL_ID,
    load_pipeline, load_and_encode, decode_latent, encode_prompt_dit,
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion_baseline, ddim_reconstruction_baseline,
    compute_metrics, save_recon_img, save_results_csv, save_results_json,
    LAYER_GROUPS, TOP_5_LAYERS, TOP_10_LAYERS, TRANSITION_ZONE,
)

OUT_DIR = Path("outputs/dit_phase2")

# coco_val images for quantitative evaluation
COCO_VAL = [
    "data/coco_val/coco_000000000285.jpg",
    "data/coco_val/coco_000000000724.jpg",
    "data/coco_val/coco_000000000802.jpg",
    "data/coco_val/coco_000000001584.jpg",
    "data/coco_val/coco_000000002153.jpg",
]
LAMBDA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
NUM_STEPS = 50


def run_lambda_scan(pipe, images, cond, out_dir):
    """Lambda scan: test correction at different strengths."""
    print(f"\n{'='*60}")
    print("Lambda Scan")
    print(f"{'='*60}")

    results = []
    for lam in LAMBDA_VALUES:
        print(f"\n  λ = {lam}")
        img_results = []
        for img_path in images:
            img_name = Path(img_path).stem
            latents, tensor, _ = load_and_encode(pipe, img_path)

            # Inversion with feature collection on top-5 layers
            noise, saved_features = ddim_inversion_with_features(
                pipe, latents, cond, NUM_STEPS, TOP_5_LAYERS,
            )

            # Reconstruction with correction (step-matched features)
            corrector = FeatureCorrector(pipe.transformer, TOP_5_LAYERS,
                                         LambdaScheduler(lam, NUM_STEPS))
            recon_latents = ddim_reconstruction_with_correction(
                pipe, noise, cond, NUM_STEPS, corrector, saved_features,
            )
            corrector.remove()

            recon_tensor = decode_latent(pipe, recon_latents)
            metrics = compute_metrics(tensor, recon_tensor)
            metrics["lambda"] = lam
            metrics["image"] = img_name
            img_results.append(metrics)
            print(f"    {img_name}: PSNR={metrics['PSNR']:.2f} LPIPS={metrics['LPIPS']:.4f}")

        avg = {k: np.mean([r[k] for r in img_results])
               for k in ["PSNR", "SSIM", "LPIPS"]}
        avg["lambda"] = lam
        avg["image"] = "average"
        print(f"    Average: PSNR={avg['PSNR']:.2f} LPIPS={avg['LPIPS']:.4f}")
        results.extend(img_results)
        results.append(avg)

    save_results_csv(results, out_dir / "lambda_scan.csv")
    save_results_json(results, out_dir / "lambda_scan.json")
    return results


def run_ablation(pipe, images, cond, out_dir):
    """Ablation: compare different layer groups."""
    print(f"\n{'='*60}")
    print("Ablation: Layer Groups")
    print(f"{'='*60}")

    # First compute baselines per image
    print("\n  Computing baselines...")
    baselines = {}
    for img_path in images:
        img_name = Path(img_path).stem
        latents, tensor, _ = load_and_encode(pipe, img_path)

        noise = ddim_inversion_baseline(pipe, latents, cond, NUM_STEPS)
        recon_latents = ddim_reconstruction_baseline(pipe, noise, cond, NUM_STEPS)
        recon_tensor = decode_latent(pipe, recon_latents)
        baselines[img_name] = {
            "metrics": compute_metrics(tensor, recon_tensor),
            "latents": latents, "tensor": tensor,
        }
        print(f"    {img_name}: PSNR={baselines[img_name]['metrics']['PSNR']:.2f}")

    # Test each layer group with lambda=0.7
    groups_to_test = ["top5", "top10", "transition", "region_transition",
                      "region_bottom", "region_top"]
    lam = 0.7

    results = []
    for group_name in groups_to_test:
        layers = LAYER_GROUPS[group_name]
        print(f"\n  [{group_name}] {len(layers)} layers: {layers[:3]}...")

        group_results = []
        for img_name, bd in baselines.items():
            latents = bd["latents"]
            tensor = bd["tensor"]
            baseline_metrics = bd["metrics"]

            noise, saved_features = ddim_inversion_with_features(
                pipe, latents, cond, NUM_STEPS, layers,
            )

            corrector = FeatureCorrector(pipe.transformer, layers,
                                         LambdaScheduler(lam, NUM_STEPS))
            recon_latents = ddim_reconstruction_with_correction(
                pipe, noise, cond, NUM_STEPS, corrector, saved_features,
            )
            corrector.remove()

            recon_tensor = decode_latent(pipe, recon_latents)
            metrics = compute_metrics(tensor, recon_tensor)
            metrics["group"] = group_name
            metrics["image"] = img_name
            metrics["delta_psnr"] = metrics["PSNR"] - baseline_metrics["PSNR"]
            group_results.append(metrics)
            print(f"    {img_name}: PSNR={metrics['PSNR']:.2f} ΔPSNR={metrics['delta_psnr']:+.2f}")

        avg = {k: np.mean([r[k] for r in group_results])
               for k in ["PSNR", "SSIM", "LPIPS", "delta_psnr"]}
        avg["group"] = group_name
        avg["image"] = "average"
        print(f"    Average: ΔPSNR={avg['delta_psnr']:+.2f} dB")
        results.extend(group_results)
        results.append(avg)

    save_results_csv(results, out_dir / "ablation.csv")
    save_results_json(results, out_dir / "ablation.json")
    return results


def plot_lambda_curve(results, out_dir):
    """Plot PSNR/LPIPS vs lambda."""
    avg_results = [r for r in results if r["image"] == "average"]
    if not avg_results:
        return
    lambdas = [r["lambda"] for r in avg_results]
    psnrs = [r["PSNR"] for r in avg_results]
    lpipss = [r["LPIPS"] for r in avg_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(lambdas, psnrs, "o-", color="steelblue", label="PSNR")
    ax2.plot(lambdas, lpipss, "s--", color="darkorange", label="LPIPS")
    ax1.set_xlabel("λ")
    ax1.set_ylabel("PSNR (dB)", color="steelblue")
    ax2.set_ylabel("LPIPS", color="darkorange")
    ax1.set_title(f"HunyuanDiT: λ Scan ({NUM_STEPS} steps, top-5 layers)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_curve.png", dpi=150)
    plt.close()


def plot_ablation(results, out_dir):
    """Plot ablation delta PSNR bar chart."""
    avg_results = [r for r in results if r["image"] == "average"]
    if not avg_results:
        return
    groups = [r["group"] for r in avg_results]
    deltas = [r["delta_psnr"] for r in avg_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(groups, deltas, color="steelblue")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ PSNR (dB)")
    ax.set_title(f"HunyuanDiT: Ablation ({NUM_STEPS} steps, λ=0.7)")
    # Annotate each bar
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{d:+.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_delta_psnr.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lambda", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--n-images", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HunyuanDiT Phase 2: Residual Correction")
    print(f"Model: {MODEL_ID}")
    print(f"Steps: {NUM_STEPS}")
    print("=" * 60)

    print("\n[1/3] Loading pipeline...")
    pipe = load_pipeline()
    cond = encode_prompt_dit(pipe, "")

    images = COCO_VAL[:args.n_images] if args.n_images else COCO_VAL
    print(f"\n[2/3] Test images ({len(images)}):")
    for p in images:
        print(f"  {p}")

    if not args.skip_lambda:
        lambda_results = run_lambda_scan(pipe, images, cond, OUT_DIR)
        plot_lambda_curve(lambda_results, OUT_DIR)

    if not args.skip_ablation:
        ablation_results = run_ablation(pipe, images, cond, OUT_DIR)
        plot_ablation(ablation_results, OUT_DIR)

    print(f"\n[3/3] Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
