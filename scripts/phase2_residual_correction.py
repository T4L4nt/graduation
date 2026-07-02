"""
第二阶段第一步：零训练残差校正模块原型
基于第一阶段诊断的 top-K 漂移层，在 DDIM 重建时注入反演路径特征作为校正信号。

校正公式: f_corrected = f_recon + λ * (f_inv - f_recon)
"""

import argparse, os, time, csv, json
from pathlib import Path

import torch
import lpips

from phase2_common import (
    DEVICE, DTYPE, MODEL_ID,
    load_pipeline, load_image, decode_latent,
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion, ddim_reconstruction,
    compute_metrics, save_recon_img, save_results_csv,
    make_grid_image, TOP_5_LAYERS,
)

OUT_DIR = Path("outputs/phase2")

# 第一阶段诊断：漂移最大的 5 个 ResNet 层（均在 decoder + bottleneck）
TOP_DRIFT_LAYERS = TOP_5_LAYERS


def run_corrected(pipe, original_latent, original_tensor, prompt_embeds,
                   num_steps, lam, layers, lpips_fn=None, compute_dists=False):
    t0 = time.perf_counter()
    noise, saved_features = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, layers
    )
    corrector = FeatureCorrector(pipe.unet, layers, lam=lam)
    recon_latent = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, num_steps, saved_features, corrector
    )
    corrector.remove()
    elapsed = time.perf_counter() - t0
    recon_tensor = decode_latent(pipe, recon_latent)
    metrics = compute_metrics(original_tensor, recon_tensor, lpips_fn, compute_dists=compute_dists)
    del noise, saved_features, recon_latent
    torch.cuda.empty_cache()
    return metrics, recon_tensor, elapsed


def run_baseline(pipe, original_latent, original_tensor, prompt_embeds,
                  num_steps, lpips_fn=None, compute_dists=False):
    t0 = time.perf_counter()
    noise = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    elapsed = time.perf_counter() - t0
    recon_tensor = decode_latent(pipe, recon_latent)
    metrics = compute_metrics(original_tensor, recon_tensor, lpips_fn, compute_dists=compute_dists)
    del noise, recon_latent
    torch.cuda.empty_cache()
    return metrics, recon_tensor, elapsed


def main():
    parser = argparse.ArgumentParser(description="Phase 2: 零训练残差校正模块原型")
    parser.add_argument("--image", type=str, default=None,
                        help="测试图路径，默认跑 face2 + watercolor")
    parser.add_argument("--steps", type=int, default=50, help="反演步数（默认 50）")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7, 1.0], help="校正强度 λ 列表")
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                        help="注入层列表（默认 top-5）")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dists", action="store_true", help="计算 DISTS 指标")
    args = parser.parse_args()

    test_images = [args.image] if args.image else [
        "data/basetest/face2.jpg",
        "data/watercolor.jpeg",
    ]
    layers = args.layers if args.layers else TOP_DRIFT_LAYERS

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[设备] {DEVICE}")
    print(f"[输出] {OUT_DIR.resolve()}")
    print(f"[图片] {test_images}")
    print(f"[步数] {args.steps}")
    print(f"[λ] {args.lambdas}")
    print(f"[DISTS] {'ON' if args.dists else 'OFF'}")

    print("[0] 加载模型...")
    pipe = load_pipeline()

    lpips_fn = None
    if not args.skip_lpips:
        print("[1] 加载 LPIPS...")
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    all_results = []

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"[跳过] {img_path}")
            continue

        img_name = Path(img_path).stem
        print(f"\n{'='*60}")
        print(f"[图片] {img_name}")
        print(f"{'='*60}")

        original_latent, original_tensor = load_image(pipe, img_path)

        # 基线（无校正）
        print(f"\n  [基线] 无校正...", end=" ", flush=True)
        base_metrics, base_recon, base_time = run_baseline(
            pipe, original_latent, original_tensor, prompt_embeds,
            args.steps, lpips_fn, compute_dists=args.dists
        )
        save_recon_img(base_recon, OUT_DIR, img_name, args.steps, "baseline")
        lpips_str = f" LPIPS={base_metrics.get('LPIPS', 0):.4f}" if lpips_fn else ""
        print(f"PSNR={base_metrics['PSNR']:.2f} SSIM={base_metrics['SSIM']:.4f}"
              f"{lpips_str} 耗时={base_time:.2f}s")

        if args.dists:
            print(f"  DISTS={base_metrics.get('DISTS', 'N/A')}")

        all_results.append({
            "image": img_name, "lambda": "baseline", **base_metrics,
            "time_s": base_time, "delta_psnr": 0.0, "delta_ssim": 0.0,
        })

        images_grid = {"Original": (original_tensor + 1) / 2,
                       "Baseline": (base_recon + 1) / 2}
        metrics_grid = {"Original": {}, "Baseline": base_metrics}

        # λ 扫描
        for lam in args.lambdas:
            tag = f"λ={lam:.1f}"
            print(f"  [{tag}] 校正中...", end=" ", flush=True)

            metrics, recon, elapsed = run_corrected(
                pipe, original_latent, original_tensor, prompt_embeds,
                args.steps, lam, layers, lpips_fn, compute_dists=args.dists
            )
            save_recon_img(recon, OUT_DIR, img_name, args.steps, f"lambda{lam:.1f}")

            delta_psnr = metrics["PSNR"] - base_metrics["PSNR"]
            delta_ssim = metrics["SSIM"] - base_metrics["SSIM"]
            lpips_str = f" LPIPS={metrics.get('LPIPS', 0):.4f}" if lpips_fn else ""
            overhead = (elapsed - base_time) / base_time * 100
            print(f"PSNR={metrics['PSNR']:.2f} (Δ{delta_psnr:+.2f}) "
                  f"SSIM={metrics['SSIM']:.4f} (Δ{delta_ssim:+.4f})"
                  f"{lpips_str} 耗时={elapsed:.2f}s (+{overhead:.1f}%)")

            all_results.append({
                "image": img_name, "lambda": f"{lam:.1f}", **metrics,
                "time_s": elapsed, "delta_psnr": delta_psnr, "delta_ssim": delta_ssim,
            })

            images_grid[tag] = (recon + 1) / 2
            metrics_grid[tag] = metrics

            del metrics, recon
            torch.cuda.empty_cache()

        # 生成并排对比网格
        grids_dir = OUT_DIR / "grids"
        os.makedirs(grids_dir, exist_ok=True)
        make_grid_image(images_grid, grids_dir / f"{img_name}_comparison.png",
                        ncols=4, reference_tensor=(original_tensor + 1) / 2,
                        metrics_dict=metrics_grid)
        print(f"  [Grid] {grids_dir / f'{img_name}_comparison.png'}")

        del original_latent, original_tensor

    save_results_csv(all_results, OUT_DIR, "metrics.csv")

    # 最佳 λ 总结
    print(f"\n{'='*60}")
    print("最佳 λ 总结:")
    for img_name in set(r["image"] for r in all_results if r["lambda"] != "baseline"):
        img_results = [r for r in all_results
                       if r["image"] == img_name and r["lambda"] != "baseline"]
        if img_results:
            best = max(img_results, key=lambda r: r["PSNR"])
            base = next(r for r in all_results
                       if r["image"] == img_name and r["lambda"] == "baseline")
            delta = best["PSNR"] - base["PSNR"]
            print(f"  {img_name}: best λ={best['lambda']}, "
                  f"PSNR={best['PSNR']:.2f} (Δ{delta:+.2f}), SSIM={best['SSIM']:.4f}")

    print(f"\n完成。输出目录: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
