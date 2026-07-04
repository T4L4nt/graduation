"""
Phase 4 三类场景验证：人像、建筑、艺术字体

对每类场景运行 Phase 2 残差校正 + Phase 3 风格注入+钉扎，
生成定量指标和对比可视化网格。

用法:
  python scripts/phase4_scenes.py
  python scripts/phase4_scenes.py --scene portraits  # 仅人像
"""

import argparse, json, sys, time, os
from pathlib import Path
from collections import defaultdict

import torch, numpy as np
import lpips

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_image, decode_latent,
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion, ddim_reconstruction,
    compute_metrics, save_recon_img, make_grid_image,
    get_top_drift_layers, get_drift_weights,
)
from phase3_common import (
    CLIPFeatureExtractor, build_style_cross_attn_tokens,
    StyleFeatureInjector,
)

OUT_DIR = Path("outputs/phase4_sota/scenes")
SCENES = {
    "portraits": ("data/portraits", "人像"),
    "architecture": ("data/architecture", "建筑"),
    "typography": ("data/typography", "艺术字体"),
}

STYLE_PRESETS = {
    "portraits": "a vibrant digital illustration",
    "architecture": "an oil painting in impressionist style",
    "typography": "a neon-lit cyberpunk scene",
}

STEPS = 50
CORR_LAM = 0.7
STYLE_STRENGTH = 0.5


def run_baseline(pipe, lat, ten, prompt_embeds):
    noise = ddim_inversion(pipe, lat, prompt_embeds, STEPS)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, STEPS)
    return decode_latent(pipe, recon_latent)


def run_correction(pipe, lat, ten, prompt_embeds, layers, weights):
    noise, saved = ddim_inversion_with_features(
        pipe, lat, prompt_embeds, STEPS, layers)
    sched = LambdaScheduler(CORR_LAM, STEPS, "constant")
    corr = FeatureCorrector(pipe.unet, layers, sched, per_layer_weights=weights)
    recon_latent = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, STEPS, saved, corr)
    corr.remove()
    return decode_latent(pipe, recon_latent)


def run_style_pin(pipe, lat, ten, prompt_embeds, styled_embeds,
                   layers, weights, extractor, v_content, v_style):
    """Phase 3: correction + style + pinning."""
    from phase3_common import run_correction_with_style_and_pinning

    noise, saved = ddim_inversion_with_features(
        pipe, lat, prompt_embeds, STEPS, layers)
    sched = LambdaScheduler(CORR_LAM, STEPS, "constant")
    corr = FeatureCorrector(pipe.unet, layers, sched, per_layer_weights=weights)

    # Style injector for feature bias (scene-specific v_style from caller)
    style_inj = StyleFeatureInjector(pipe.unet, layers, v_style, strength=STYLE_STRENGTH)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(STEPS, device=DEVICE)
    timesteps = scheduler.timesteps

    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(ten), v_content)

    z = noise.clone()
    pinning_log = []
    effective_strength = STYLE_STRENGTH

    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        if t_int in saved:
            corr.set_reference(saved[t_int], step_idx)
        else:
            corr.set_reference({}, step_idx)

        # Pinning check every 5 steps
        if step_idx > 0 and step_idx % 5 == 0:
            with torch.no_grad():
                current_img = decode_latent(pipe, z.clone())
                v_current = extractor.encode_image_from_tensor(current_img)
                cur_proj = extractor.compute_content_projection(v_current, v_content)
                deviation = abs(cur_proj - ref_proj)
                pinning_log.append((step_idx, cur_proj, deviation, effective_strength))
                if deviation > 0.02:
                    scale = max(0.0, 1.0 - 0.5 * deviation / max(ref_proj, 0.01))
                    effective_strength = STYLE_STRENGTH * scale
                    style_inj.set_strength(effective_strength)

        inp = scheduler.scale_model_input(z, t)
        with torch.no_grad():
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=styled_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    corr.remove()
    style_inj.remove()

    return decode_latent(pipe, z), pinning_log


def process_scene(scene_name, img_dir, pipe, layers, weights, extractor, lpips_fn):
    """Run Phase 2 + Phase 3 on all images in a scene directory."""
    print(f"\n{'='*60}")
    print(f"Scene: {SCENES[scene_name][1]} ({scene_name})")
    print(f"{'='*60}")

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")

    style_text = STYLE_PRESETS[scene_name]
    v_style = extractor.encode_text(style_text)
    styled_embeds = build_style_cross_attn_tokens(
        pipe, "", v_style, strength=STYLE_STRENGTH, mode="extra_token")
    print(f"Style: \"{style_text}\"")

    img_paths = sorted(Path(img_dir).glob("*.jpg"))
    scene_dir = OUT_DIR / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    images_for_grid = {}

    for img_path in img_paths:
        img_name = img_path.stem
        lat, ten = load_image(pipe, str(img_path))
        is_face = "portrait" in scene_name

        print(f"\n  {img_name}:")

        # 1) Baseline
        t0 = time.perf_counter()
        recon_base = run_baseline(pipe, lat, ten, prompt_embeds)
        bm = compute_metrics(ten, recon_base, lpips_fn, compute_arcface=is_face)
        bt = time.perf_counter() - t0
        print(f"    baseline:      PSNR={bm['PSNR']:.2f} LPIPS={bm['LPIPS']:.3f}"
              f"{' ArcFace='+str(bm.get('ArcFace',0))[:6] if is_face else ''}  ({bt:.1f}s)")
        bm["method"] = "baseline"; bm["image"] = img_name; bm["time_s"] = bt
        all_results.append(bm)
        save_recon_img(recon_base, scene_dir, img_name, STEPS, "baseline")

        # 2) Correction
        t0 = time.perf_counter()
        recon_corr = run_correction(pipe, lat, ten, prompt_embeds, layers, weights)
        cm = compute_metrics(ten, recon_corr, lpips_fn, compute_arcface=is_face)
        ct = time.perf_counter() - t0
        cm_delta = cm["PSNR"] - bm["PSNR"]
        print(f"    correction:    PSNR={cm['PSNR']:.2f} LPIPS={cm['LPIPS']:.3f}  "
              f"Δ={cm_delta:+.2f} dB  ({ct:.1f}s)")
        cm["method"] = "correction"; cm["image"] = img_name; cm["delta_psnr"] = cm_delta
        cm["time_s"] = ct
        all_results.append(cm)
        save_recon_img(recon_corr, scene_dir, img_name, STEPS, "correction")

        # 3) Style + Pin
        t0 = time.perf_counter()
        recon_pin, pin_log = run_style_pin(
            pipe, lat, ten, prompt_embeds, styled_embeds,
            layers, weights, extractor, v_content, v_style)
        sm = compute_metrics(ten, recon_pin, lpips_fn, compute_arcface=is_face)
        st = time.perf_counter() - t0
        sm_delta = sm["PSNR"] - bm["PSNR"]
        n_pin = sum(1 for d in pin_log if d[2] > 0.02)
        print(f"    style+pin:     PSNR={sm['PSNR']:.2f} LPIPS={sm['LPIPS']:.3f}  "
              f"Δ={sm_delta:+.2f} dB  pin={n_pin}/{len(pin_log)}  ({st:.1f}s)")
        sm["method"] = "style_pin"; sm["image"] = img_name; sm["delta_psnr"] = sm_delta
        sm["pinning_triggered"] = n_pin; sm["pinning_checks"] = len(pin_log)
        sm["time_s"] = st
        all_results.append(sm)
        save_recon_img(recon_pin, scene_dir, img_name, STEPS, "style_pin")

        # Store first 3 images for grid
        if len(images_for_grid) < 3:
            images_for_grid[img_name] = {
                "Original": ten, "Baseline": recon_base,
                "Correction": recon_corr, "Style+Pin": recon_pin,
            }

        del lat, ten, recon_base, recon_corr, recon_pin
        torch.cuda.empty_cache()

    # Save metrics
    with open(scene_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate comparison grid
    for img_name, imgs in list(images_for_grid.items())[:3]:
        images_01 = {k: (v + 1) / 2 for k, v in imgs.items()}
        grid_path = scene_dir / f"{img_name}_comparison.png"
        make_grid_image(images_01, str(grid_path), ncols=4,
                        reference_tensor=(imgs["Original"] + 1) / 2)
        print(f"  [Grid] {grid_path}")

    # Summary
    methods = ["baseline", "correction", "style_pin"]
    print(f"\n  Summary ({scene_name}):")
    for m_name in methods:
        vals = [r["PSNR"] for r in all_results if r["method"] == m_name]
        lpips_vals = [r.get("LPIPS", 0) for r in all_results if r["method"] == m_name]
        avg_p = np.mean(vals) if vals else 0
        avg_l = np.mean(lpips_vals) if lpips_vals else 0
        if m_name == "baseline":
            print(f"    {m_name:<15s} PSNR={avg_p:.2f}  LPIPS={avg_l:.3f}")
        else:
            b_vals = [r["PSNR"] for r in all_results if r["method"] == "baseline"]
            delta = avg_p - np.mean(b_vals) if b_vals else 0
            print(f"    {m_name:<15s} PSNR={avg_p:.2f}  LPIPS={avg_l:.3f}  Δ={delta:+.2f} dB")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None,
                        help="Single scene name (portraits/architecture/typography)")
    args = parser.parse_args()

    scenes_to_run = [args.scene] if args.scene else list(SCENES.keys())

    print(f"[Scene Validation] {len(scenes_to_run)} scenes, {STEPS} steps")
    print(f"[Corr λ] {CORR_LAM}, [Style strength] {STYLE_STRENGTH}")

    print("[0] Loading models...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    layers = get_top_drift_layers(5)
    weights = get_drift_weights(layers)
    print(f"Layers ({len(layers)}): {[l.split('.')[-1] for l in layers]}")
    w_summary = {l.split('.')[-1]: f"{w:.2f}" for l, w in weights.items()}
    print(f"Weights: {w_summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scene_results = {}
    for scene_name in scenes_to_run:
        img_dir, _ = SCENES[scene_name]
        results = process_scene(scene_name, img_dir, pipe, layers, weights,
                                extractor, lpips_fn)
        all_scene_results[scene_name] = results

    # Cross-scene summary
    print(f"\n{'='*70}")
    print("CROSS-SCENE SUMMARY")
    print(f"{'Scene':<18s} {'Baseline':>10s} {'Correction':>12s} {'Style+Pin':>12s}")
    print("-" * 55)

    for scene_name in scenes_to_run:
        results = all_scene_results[scene_name]
        b = np.mean([r["PSNR"] for r in results if r["method"] == "baseline"])
        c = np.mean([r["PSNR"] for r in results if r["method"] == "correction"])
        s = np.mean([r["PSNR"] for r in results if r["method"] == "style_pin"])
        label = SCENES[scene_name][1]
        print(f"{label:<18s} {b:>8.2f} dB  {c:>8.2f} dB (Δ{c-b:+.1f})  "
              f"{s:>8.2f} dB (Δ{s-b:+.1f})")

    # Save cross-scene summary
    summary = {}
    for scene_name in scenes_to_run:
        results = all_scene_results[scene_name]
        summary[scene_name] = {
            "n_images": len(set(r["image"] for r in results)),
            "methods": {},
        }
        for m in ["baseline", "correction", "style_pin"]:
            vals = [r["PSNR"] for r in results if r["method"] == m]
            lpips_vals = [r.get("LPIPS", 0) for r in results if r["method"] == m]
            if vals:
                summary[scene_name]["methods"][m] = {
                    "PSNR_mean": float(np.mean(vals)),
                    "PSNR_std": float(np.std(vals)),
                    "LPIPS_mean": float(np.mean(lpips_vals)),
                }

    with open(OUT_DIR / "cross_scene_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
