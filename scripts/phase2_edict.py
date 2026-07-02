"""
第二阶段第三步：EDICT 基线对比
EDICT (CVPR 2023): 双向量耦合实现数学精确可逆反演

对比:
  1. DDIM (基线)
  2. DDIM + 残差校正 (我们的方法)
  3. EDICT (精确可逆基线)

用法:
  python scripts/phase2_edict.py
"""

import argparse, os, time, csv, json
from pathlib import Path

import torch
import numpy as np
import lpips
from torchvision import transforms

from phase2_common import (
    DEVICE, DTYPE, MODEL_ID,
    load_pipeline, load_image, decode_latent,
    FeatureCollector, FeatureCorrector,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion, ddim_reconstruction,
    compute_metrics, save_recon_img, save_results_csv,
    make_grid_image, DEFAULT_TEST_IMAGES, get_top_drift_layers,
)

OUT_DIR = Path("outputs/phase2_edict")

STEP_LIST = [10, 20, 50, 100]
EDICT_P = 0.93  # mixing factor

# Global correction lambda (from val-set tuning, not per-image optimal)
EDICT_CORR_LAMBDA = 0.5


# ---------------------------------------------------------------------------
# EDICT 反演 & 重建
# ---------------------------------------------------------------------------

def edict_inversion(pipe, latents, prompt_embeds, num_steps, p=0.93):
    """EDICT inversion (Algorithm 2): 从 x_0 到 (x_T, y_T)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    extended_ts = timesteps.tolist() + [0]

    x = latents.clone()
    y = latents.clone() + 0.001 * torch.randn_like(latents)

    inv_det = 1.0 / (2 * p - 1)

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            x_int = inv_det * (p * x + (p - 1) * y)
            y_int = inv_det * ((p - 1) * x + p * y)

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur

            eps_x = pipe.unet(x_int, t_cur, encoder_hidden_states=prompt_embeds).sample
            x_next = coeff1 * x_int + coeff2 * eps_x

            eps_y = pipe.unet(y_int, t_cur, encoder_hidden_states=prompt_embeds).sample
            y_prime = coeff1 * y_int + coeff2 * eps_y
            y_next = 2 * x_next - y_prime

            x, y = x_next, y_next

    return x, y


def edict_reconstruction(pipe, x_T, y_T, prompt_embeds, num_steps, p=0.93):
    """EDICT reconstruction (Algorithm 1): 从 (x_T, y_T) 回到 x_0."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    x = x_T.clone()
    y = y_T.clone()

    with torch.no_grad():
        for t in timesteps:
            def denoise_step(latent):
                eps = pipe.unet(latent, t, encoder_hidden_states=prompt_embeds).sample
                return scheduler.step(eps, t, latent).prev_sample

            x_inter = denoise_step(x)
            y_prime = 2 * x - y
            y_inter = denoise_step(y_prime)

            x_next = p * x_inter + (1 - p) * y_inter
            y_next = p * y_inter + (1 - p) * x_inter

            x, y = x_next, y_next

    return x, y


# ---------------------------------------------------------------------------
# 总结
# ---------------------------------------------------------------------------

def print_summary(results):
    print(f"\n{'='*70}")
    print("三方对比总结 (50步)")
    print(f"{'Image':15s} {'DDIM':>8s} {'DDIM+Corr':>10s} {'EDICT':>8s}  "
          f"{'CorrΔ':>6s} {'EDICTΔ':>6s}  {'EDICT/DDIM':>10s}")
    print("-" * 70)

    images = sorted(set(r["image"] for r in results))
    for img in images:
        ddim_r = next(r for r in results if r["image"] == img and r["steps"] == 50
                      and r["method"] == "DDIM")
        corr_r = next(r for r in results if r["image"] == img and r["steps"] == 50
                      and r["method"].startswith("DDIM+Corr"))
        edict_r = next(r for r in results if r["image"] == img and r["steps"] == 50
                       and r["method"] == "EDICT")

        corr_delta = corr_r["PSNR"] - ddim_r["PSNR"]
        edict_delta = edict_r["PSNR"] - ddim_r["PSNR"]

        print(f"{img:15s} {ddim_r['PSNR']:8.2f} {corr_r['PSNR']:10.2f} "
              f"{edict_r['PSNR']:8.2f}  {corr_delta:+6.2f} {edict_delta:+6.2f}  "
              f"{edict_r['time_s']/ddim_r['time_s']:6.1f}x")

    ddim_avg = np.mean([r["PSNR"] for r in results
                        if r["steps"] == 50 and r["method"] == "DDIM"])
    corr_avg = np.mean([r["PSNR"] for r in results
                        if r["steps"] == 50 and r["method"].startswith("DDIM+Corr")])
    edict_avg = np.mean([r["PSNR"] for r in results
                         if r["steps"] == 50 and r["method"] == "EDICT"])
    print("-" * 70)
    print(f"{'AVERAGE':15s} {ddim_avg:8.2f} {corr_avg:10.2f} {edict_avg:8.2f}  "
          f"{corr_avg-ddim_avg:+6.2f} {edict_avg-ddim_avg:+6.2f}")


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDICT 基线对比")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dists", action="store_true", help="计算 DISTS 指标")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[设备] {DEVICE}")
    print(f"[p] = {EDICT_P}")
    print(f"[DISTS] {'ON' if args.dists else 'OFF'}")

    print("[0] 加载模型...")
    pipe = load_pipeline()

    lpips_fn = None
    if not args.skip_lpips:
        print("[1] 加载 LPIPS...")
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    all_results = []

    test_images = DEFAULT_TEST_IMAGES
    layers = get_top_drift_layers(5)

    # Load global lambda from Phase 2 tuning if available
    lam = EDICT_CORR_LAMBDA
    tuning_path = Path("outputs/phase2_full/selected_lambda.json")
    if tuning_path.exists():
        with open(tuning_path) as f:
            data = json.load(f)
        lam = data.get("selected_lambda", EDICT_CORR_LAMBDA)
        print(f"[λ] Loaded from tuning: λ={lam}")

    total = len(test_images) * len(STEP_LIST) * 3
    count = 0

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)

        # for grid at best step
        grid_images = {"Original": (original_tensor + 1) / 2}
        grid_metrics = {"Original": {}}

        for steps in STEP_LIST:
            # --- DDIM 基线 ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 DDIM...", end="", flush=True)
            t0 = time.perf_counter()
            noise = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
            recon_ddim = ddim_reconstruction(pipe, noise, prompt_embeds, steps)
            t_ddim = time.perf_counter() - t0
            recon_ddim_img = decode_latent(pipe, recon_ddim)
            m_ddim = compute_metrics(original_tensor, recon_ddim_img,
                                      lpips_fn, compute_dists=args.dists)
            all_results.append({"image": img_name, "steps": steps, "method": "DDIM",
                                **m_ddim, "time_s": t_ddim})
            save_recon_img(recon_ddim_img, OUT_DIR, img_name, steps, "ddim")
            del noise, recon_ddim
            torch.cuda.empty_cache()

            # --- DDIM + 残差校正 ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 DDIM+Corr(λ={lam:.1f})...",
                  end="", flush=True)
            t0 = time.perf_counter()
            noise_c, saved_f = ddim_inversion_with_features(
                pipe, original_latent, prompt_embeds, steps, layers)
            corrector = FeatureCorrector(pipe.unet, layers, lam=lam)
            recon_corr = ddim_reconstruction_with_correction(
                pipe, noise_c, prompt_embeds, steps, saved_f, corrector)
            corrector.remove()
            t_corr = time.perf_counter() - t0
            m_corr = compute_metrics(original_tensor, decode_latent(pipe, recon_corr),
                                      lpips_fn, compute_dists=args.dists)
            all_results.append({"image": img_name, "steps": steps,
                                "method": f"DDIM+Corr(λ={lam:.1f})",
                                **m_corr, "time_s": t_corr})
            recon_corr_img = decode_latent(pipe, recon_corr)
            save_recon_img(recon_corr_img, OUT_DIR, img_name, steps,
                           f"ddim_corr_lam{lam:.1f}")
            del noise_c, saved_f, recon_corr
            torch.cuda.empty_cache()

            # --- EDICT ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 EDICT...", end="", flush=True)
            t0 = time.perf_counter()
            z_T, z_hat_T = edict_inversion(pipe, original_latent, prompt_embeds, steps, EDICT_P)
            recon_edict, _ = edict_reconstruction(pipe, z_T, z_hat_T, prompt_embeds, steps, EDICT_P)
            t_edict = time.perf_counter() - t0
            m_edict = compute_metrics(original_tensor, decode_latent(pipe, recon_edict),
                                       lpips_fn, compute_dists=args.dists)
            all_results.append({"image": img_name, "steps": steps, "method": "EDICT",
                                **m_edict, "time_s": t_edict})
            recon_edict_img = decode_latent(pipe, recon_edict)
            save_recon_img(recon_edict_img, OUT_DIR, img_name, steps, "edict")
            del z_T, z_hat_T, recon_edict
            torch.cuda.empty_cache()

            # Save best-step results for grid (use pre-decoded images)
            if steps == 50:
                grid_images["DDIM"] = (recon_ddim_img + 1) / 2
                grid_images["DDIM+Corr"] = (recon_corr_img + 1) / 2
                grid_images["EDICT"] = (recon_edict_img + 1) / 2
                grid_metrics["DDIM"] = m_ddim
                grid_metrics["DDIM+Corr"] = m_corr
                grid_metrics["EDICT"] = m_edict

        # Generate comparison grid for 50-step results
        if len(grid_images) >= 3:
            grid_dir = OUT_DIR / "grids"
            os.makedirs(grid_dir, exist_ok=True)
            make_grid_image(grid_images, grid_dir / f"{img_name}_comparison.png",
                            ncols=4, reference_tensor=(original_tensor + 1) / 2,
                            metrics_dict=grid_metrics)
            print(f"\n  [Grid] {grid_dir / f'{img_name}_comparison.png'}")

        del original_latent, original_tensor

    print()
    save_results_csv(all_results, OUT_DIR, "metrics.csv")
    print_summary(all_results)

    print(f"\n完成。输出: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
