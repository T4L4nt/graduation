"""
第二阶段完整系统实验：零训练残差校正
- 全图×全步数×λ 扫描 + λ-Scheduler + 消融 + ArcFace + DISTS

用法:
  python scripts/phase2_full.py --mode full       # 完整实验
  python scripts/phase2_full.py --mode ablation    # 消融实验（仅 50 步）
  python scripts/phase2_full.py --mode quick       # 快速验证（face2×50步）
"""

import argparse, os, time, json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lpips

from phase2_common import (
    DEVICE, DTYPE,
    load_pipeline, load_image, decode_latent,
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_latents, ddim_reconstruction_with_latent_correction,
    compute_metrics, compute_arcface_similarity,
    make_grid_image, save_recon_img, save_results_csv,
    TOP_5_LAYERS, LAYER_GROUPS,
    get_image_split, get_top_drift_layers,
)

OUT_DIR = Path("outputs/phase2_full")

STEP_LIST = [4, 10, 20, 50, 100]
LAMBDA_LIST = [0.0, 0.3, 0.5, 0.7]
SCHEDULER_TYPES = ["constant", "linear"]


# ---------------------------------------------------------------------------
# 实验运行
# ---------------------------------------------------------------------------

def run_corrected(pipe, original_latent, original_tensor, prompt_embeds,
                   num_steps, lam, layers, sched_mode, lpips_fn=None,
                   compute_arcface=False, compute_dists=False):
    t0 = time.perf_counter()
    noise, saved_features = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, layers
    )
    lam_sched = LambdaScheduler(lam, num_steps, sched_mode)
    corrector = FeatureCorrector(pipe.unet, layers, lam_sched)
    recon_latent = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, num_steps, saved_features, corrector
    )
    corrector.remove()
    elapsed = time.perf_counter() - t0
    recon_tensor = decode_latent(pipe, recon_latent)
    metrics = compute_metrics(original_tensor, recon_tensor, lpips_fn,
                               compute_arcface, compute_dists)
    del noise, saved_features, recon_latent
    torch.cuda.empty_cache()
    return metrics, recon_tensor, elapsed


def run_baseline(pipe, original_latent, original_tensor, prompt_embeds,
                  num_steps, lpips_fn=None, compute_arcface=False, compute_dists=False):
    t0 = time.perf_counter()
    noise = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    elapsed = time.perf_counter() - t0
    recon_tensor = decode_latent(pipe, recon_latent)
    metrics = compute_metrics(original_tensor, recon_tensor, lpips_fn,
                               compute_arcface, compute_dists)
    del noise, recon_latent
    torch.cuda.empty_cache()
    return metrics, recon_tensor, elapsed


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

def run_full(pipe, lpips_fn, split="all", tune_lambda=False, compute_dists=False):
    """完整实验：分集支持 + 可选 λ 自动调优。

    Args:
        split: "val" | "test" | "all"
        tune_lambda: 若 True，在 val 集上扫描 λ，选最优值后用于 test 集
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    all_results = []

    # Determine which images and whether to do lambda scan
    if tune_lambda:
        val_images = get_image_split("val")
        test_images = get_image_split(split) if split != "val" else val_images
        print(f"[Split] val tuning: {len(val_images)} images")
        print(f"[Split] {split} evaluation: {len(test_images)} images")
    else:
        test_images = get_image_split(split)
        val_images = test_images
        print(f"[Split] {split}: {len(test_images)} images")

    # --- Phase A: Lambda tuning on validation set ---
    selected_lambda = 0.5  # default fallback
    if tune_lambda:
        print(f"\n{'='*60}")
        print("[Tune] λ 扫描 (验证集)...")
        val_prompt = pipe.encode_prompt("", DEVICE, 1, False)[0]
        val_layer = layers
        val_results = []

        for lam in LAMBDA_LIST:
            if lam == 0.0:
                continue  # skip λ=0 (identity)
            psnrs = []
            for img_path in val_images:
                if not os.path.exists(img_path):
                    continue
                img_name = Path(img_path).stem
                is_face = "face" in img_name.lower()
                orig_latent, orig_tensor = load_image(pipe, img_path)
                for steps in [50]:  # tune on 50 steps only
                    m, _, _ = run_corrected(
                        pipe, orig_latent, orig_tensor, val_prompt,
                        steps, lam, val_layer, "constant", lpips_fn,
                        compute_arcface=is_face, compute_dists=compute_dists
                    )
                    psnrs.append(m["PSNR"])
                del orig_latent, orig_tensor
                torch.cuda.empty_cache()

            mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
            val_results.append({"lambda": lam, "mean_psnr": mean_psnr,
                                "n_images": len(psnrs)})
            print(f"  λ={lam:.1f}: mean PSNR={mean_psnr:.2f} (n={len(psnrs)})")

        best = max(val_results, key=lambda x: x["mean_psnr"])
        selected_lambda = best["lambda"]
        print(f"[Tune] Selected λ={selected_lambda} "
              f"(mean PSNR={best['mean_psnr']:.2f})")

        tuning_path = OUT_DIR / "lambda_tuning.json"
        with open(tuning_path, "w") as f:
            json.dump({
                "val_images": [str(Path(p).name) for p in val_images if os.path.exists(p)],
                "val_results": val_results,
                "selected_lambda": selected_lambda,
            }, f, indent=2)
        print(f"[Tune] Saved → {tuning_path}")

        # Save selected lambda for other scripts to load
        sel_path = OUT_DIR / "selected_lambda.json"
        with open(sel_path, "w") as f:
            json.dump({"selected_lambda": selected_lambda}, f)
        print(f"[Tune] Global λ → {sel_path}")

    # --- Phase B: Run on target split with selected/fixed lambda ---
    if tune_lambda:
        lam_list = [selected_lambda]
    elif split == "test":
        # Auto-load tuned lambda for test set evaluation
        sel_path = OUT_DIR / "selected_lambda.json"
        if sel_path.exists():
            with open(sel_path) as f:
                lam_list = [json.load(f)["selected_lambda"]]
            print(f"[λ] Auto-loaded from tuning: λ={lam_list[0]}")
        else:
            lam_list = [0.5]  # fallback
            print("[λ] No tuning found, using λ=0.5")
    else:
        lam_list = LAMBDA_LIST

    total = len(test_images) * len(STEP_LIST) * (1 + len(lam_list) * len(SCHEDULER_TYPES))
    count = 0

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        original_latent, original_tensor = load_image(pipe, img_path)

        for steps in STEP_LIST:
            # 基线
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 baseline...", end="", flush=True)
            base_m, base_r, base_t = run_baseline(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, lpips_fn, compute_arcface=is_face, compute_dists=compute_dists
            )
            row = {"image": img_name, "steps": steps, "lambda": "baseline",
                    "scheduler": "none", **base_m, "time_s": base_t}
            all_results.append(row)
            save_recon_img(base_r, OUT_DIR, img_name, steps, "baseline")

            # λ × scheduler
            for lam in lam_list:
                if lam == 0.0:
                    sched_modes = ["constant"]
                else:
                    sched_modes = SCHEDULER_TYPES

                for sched in sched_modes:
                    count += 1
                    tag = f"λ={lam:.1f}/{sched}"
                    print(f"\r[{count}/{total}] {img_name} {steps}步 {tag}...", end="", flush=True)
                    m, r, t = run_corrected(
                        pipe, original_latent, original_tensor, prompt_embeds,
                        steps, lam, layers, sched, lpips_fn,
                        compute_arcface=is_face, compute_dists=compute_dists
                    )
                    row = {"image": img_name, "steps": steps, "lambda": f"{lam:.1f}",
                            "scheduler": sched, **m, "time_s": t}
                    all_results.append(row)
                    save_recon_img(r, OUT_DIR, img_name, steps, f"lam{lam:.1f}_{sched}")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    print()
    save_results_csv(all_results, OUT_DIR, "metrics.csv")
    plot_full_summary(all_results)
    return all_results


def run_ablation(pipe, lpips_fn, compute_dists=False):
    """消融实验：层组 + λ=1.0 + latent插值，固定 50 步（仅验证集）"""
    os.makedirs(OUT_DIR / "ablation", exist_ok=True)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    steps = 50
    sched = "constant"
    all_results = []
    test_images = get_image_split("val")

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        original_latent, original_tensor = load_image(pipe, img_path)

        print(f"  {img_name} baseline...")
        base_m, base_r, base_t = run_baseline(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, lpips_fn, compute_arcface=is_face, compute_dists=compute_dists
        )
        all_results.append({"image": img_name, "layers": "baseline",
                            **base_m, "time_s": base_t, "delta_psnr": 0.0})
        save_recon_img(base_r, OUT_DIR, img_name, steps, "baseline", subdir="ablation")

        # --- Layer group ablations (λ=0.5) ---
        for group_name, group_layers in LAYER_GROUPS.items():
            print(f"  {img_name} {group_name} ({len(group_layers)}层)...")
            m, r, t = run_corrected(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, 0.5, group_layers, sched, lpips_fn,
                compute_arcface=is_face, compute_dists=compute_dists
            )
            m["delta_psnr"] = m["PSNR"] - base_m["PSNR"]
            all_results.append({"image": img_name, "layers": group_name,
                                **m, "time_s": t})
            save_recon_img(r, OUT_DIR, img_name, steps, f"ablation_{group_name}",
                           subdir="ablation")

        # --- λ=1.0: full feature replacement ---
        print(f"  {img_name} λ=1.0 (full replacement)...")
        m, r, t = run_corrected(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, 1.0, get_top_drift_layers(5), sched, lpips_fn,
            compute_arcface=is_face, compute_dists=compute_dists
        )
        m["delta_psnr"] = m["PSNR"] - base_m["PSNR"]
        all_results.append({"image": img_name, "layers": "lambda=1.0_full_replace",
                            **m, "time_s": t})
        save_recon_img(r, OUT_DIR, img_name, steps, "ablation_lam1.0",
                       subdir="ablation")

        # --- Latent-space interpolation ---
        for lam_latent in [0.5, 0.7]:
            print(f"  {img_name} latent-interp λ={lam_latent}...")
            t0 = time.perf_counter()
            noise, saved_latents = ddim_inversion_with_latents(
                pipe, original_latent, prompt_embeds, steps)
            recon_latent = ddim_reconstruction_with_latent_correction(
                pipe, noise, prompt_embeds, steps, saved_latents, lam=lam_latent)
            elapsed = time.perf_counter() - t0
            recon = decode_latent(pipe, recon_latent)
            m = compute_metrics(original_tensor, recon, lpips_fn,
                                compute_arcface=is_face, compute_dists=compute_dists)
            m["delta_psnr"] = m["PSNR"] - base_m["PSNR"]
            all_results.append({"image": img_name,
                                "layers": f"latent_interp_λ={lam_latent}",
                                **m, "time_s": elapsed})
            save_recon_img(recon, OUT_DIR, img_name, steps,
                           f"ablation_latent_lam{lam_latent}", subdir="ablation")
            del noise, saved_latents, recon_latent, recon
            torch.cuda.empty_cache()

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    save_results_csv(all_results, OUT_DIR, "metrics.csv", subdir="ablation")
    plot_ablation_summary(all_results)
    return all_results


def run_quick(pipe, lpips_fn, compute_dists=False):
    """快速验证：face2 × 50 步，对比 constant vs linear scheduler"""
    os.makedirs(OUT_DIR, exist_ok=True)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = TOP_5_LAYERS
    steps = 50
    img_path = "data/basetest/face2.jpg"

    original_latent, original_tensor = load_image(pipe, img_path)

    print(" baseline...")
    base_m, base_r, base_t = run_baseline(
        pipe, original_latent, original_tensor, prompt_embeds,
        steps, lpips_fn, compute_dists=compute_dists
    )
    print(f"  PSNR={base_m['PSNR']:.2f} SSIM={base_m['SSIM']:.4f} LPIPS={base_m['LPIPS']:.4f}")

    images_dict = {"Original": (original_tensor + 1) / 2, "Baseline": (base_r + 1) / 2}
    metrics_dict = {"Original": {}, "Baseline": base_m}

    for lam in [0.3, 0.5, 0.7]:
        for sched in ["constant", "linear"]:
            print(f" λ={lam:.1f} {sched}...", end=" ")
            m, r, t = run_corrected(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, lam, layers, sched, lpips_fn, compute_dists=compute_dists
            )
            delta = m["PSNR"] - base_m["PSNR"]
            print(f"PSNR={m['PSNR']:.2f} (Δ{delta:+.2f}) LPIPS={m['LPIPS']:.4f}")
            save_recon_img(r, OUT_DIR, "face2", steps, f"lam{lam:.1f}_{sched}")
            tag = f"λ={lam:.1f}/{sched}"
            images_dict[tag] = (r + 1) / 2
            metrics_dict[tag] = m

    # 生成并排对比网格
    os.makedirs(OUT_DIR / "grids", exist_ok=True)
    make_grid_image(images_dict, OUT_DIR / "grids" / "face2_comparison.png",
                    ncols=4, reference_tensor=(original_tensor + 1) / 2,
                    metrics_dict=metrics_dict)
    print(f"[Grid] {OUT_DIR / 'grids' / 'face2_comparison.png'}")

    del original_latent, original_tensor


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_full_summary(results):
    if not results:
        return
    images = sorted(set(r["image"] for r in results))
    n_img = len(images)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, img_name in enumerate(images):
        if i >= len(axes):
            break
        ax = axes[i]
        img_results = [r for r in results if r["image"] == img_name]

        groups = defaultdict(list)
        for r in img_results:
            if r["lambda"] == "baseline":
                groups["baseline"].append(r)
            else:
                key = f"λ={r['lambda']}/{r['scheduler']}"
                groups[key].append(r)

        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for (gname, gdata), c in zip(groups.items(), colors):
            pts = sorted(gdata, key=lambda x: x["steps"])
            x = [p["steps"] for p in pts]
            y = [p["PSNR"] for p in pts]
            ls = "-" if "baseline" in gname else "--"
            ax.plot(x, y, "o" + ls, color=c, label=gname, markersize=4)

        ax.set_title(img_name)
        ax.set_xlabel("Steps")
        ax.set_ylabel("PSNR (dB)")
        ax.legend(fontsize=6)

    for j in range(len(images), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "psnr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] PSNR 曲线 → {OUT_DIR / 'psnr_curves.png'}")


def plot_ablation_summary(results):
    if not results:
        return
    images = sorted(set(r["image"] for r in results))
    layer_names = sorted(set(r["layers"] for r in results if r["layers"] != "baseline"))
    n_layers = len(layer_names)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.5), 5))
    x = np.arange(len(images))
    width = 0.8 / (n_layers + 1)

    baseline_vals = []
    for img in images:
        b = next((r for r in results if r["image"] == img and r["layers"] == "baseline"), None)
        baseline_vals.append(b["PSNR"] if b else 0)

    ax.bar(x - 0.4 + width, baseline_vals, width, label="baseline", color="#999")

    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    for j, lname in enumerate(layer_names):
        vals = []
        for img in images:
            r = next((r for r in results if r["image"] == img and r["layers"] == lname), None)
            vals.append(r["PSNR"] if r else 0)
        ax.bar(x - 0.4 + width * (j + 2), vals, width, label=lname, color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(images)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Ablation: Layer Groups (50 steps, λ=0.5)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ablation/ablation_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 消融柱状图 → {OUT_DIR / 'ablation/ablation_bars.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 完整系统实验")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "ablation"],
                        help="实验模式")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "val", "test"],
                        help="图片分集: val=验证集(λ调参), test=测试集(最终评估), all=全部")
    parser.add_argument("--tune-lambda", action="store_true",
                        help="在验证集上扫描 λ，选出最优值后用于目标分集")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dists", action="store_true", help="计算 DISTS 指标")
    args = parser.parse_args()

    print(f"[设备] {DEVICE}")
    print(f"[模式] {args.mode}")
    print(f"[分集] {args.split}")
    print(f"[Tune λ] {'ON' if args.tune_lambda else 'OFF'}")
    print(f"[DISTS] {'ON' if args.dists else 'OFF'}")
    print(f"[输出] {OUT_DIR.resolve()}")

    print("[0] 加载模型...")
    pipe = load_pipeline()

    lpips_fn = None
    if not args.skip_lpips:
        print("[1] 加载 LPIPS...")
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    if args.mode == "quick":
        run_quick(pipe, lpips_fn, compute_dists=args.dists)
    elif args.mode == "full":
        run_full(pipe, lpips_fn, split=args.split,
                 tune_lambda=args.tune_lambda, compute_dists=args.dists)
    elif args.mode == "ablation":
        run_ablation(pipe, lpips_fn, compute_dists=args.dists)

    print(f"\n完成。输出: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
