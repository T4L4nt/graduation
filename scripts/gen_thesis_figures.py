"""
Generate paper-quality comparison figures and user study materials.
Content images from data/coco_val only. Style via text prompt.

Usage:
  python scripts/gen_thesis_figures.py                    # all figures + user study
  python scripts/gen_thesis_figures.py --mode figures     # only thesis figures
  python scripts/gen_thesis_figures.py --mode user_study  # only user study pairs
"""

import argparse, os, sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from phase2_common import (DEVICE, DTYPE, load_pipeline, load_image, decode_latent,
    ddim_inversion, ddim_reconstruction, ddim_inversion_with_features,
    ddim_reconstruction_with_correction, FeatureCorrector,
    compute_metrics, save_recon_img, make_grid_image, get_top_drift_layers)
from phase3_prep import (CLIPFeatureExtractor, build_style_cross_attn_tokens,
    run_baseline, run_correction_with_style, run_correction_with_style_and_pinning,
    run_correction_only, slerp, make_styled_prompt_embedding)

OUT_DIR = Path("outputs/thesis_figures")

# ── All images from data/coco_val ──────────────────────────────────────────
COCO_DIR = Path("data/coco_val")
ALL_COCO = sorted([str(p) for p in COCO_DIR.glob("*.jpg")])

# Phase 2: 6 diverse images (stratified by file size)
PHASE2_SELECTION = [
    "data/coco_val/coco_000000000285.jpg",
    "data/coco_val/coco_000000000872.jpg",
    "data/coco_val/coco_000000000139.jpg",
    "data/coco_val/coco_000000000776.jpg",
    "data/coco_val/coco_000000000802.jpg",
    "data/coco_val/coco_000000001675.jpg",
]

# Phase 3 content images
PHASE3_CONTENT = [
    "data/coco_val/coco_000000000285.jpg",
    "data/coco_val/coco_000000000872.jpg",
    "data/coco_val/coco_000000000139.jpg",
]

# Style prompts — restrained, natural
STYLE_PROMPT_WATERCOLOR = "a watercolor painting"

# User study content pairs
USER_STUDY_CONTENT = [p for p in ALL_COCO if os.path.exists(p)]


def _img_name(p):
    return Path(p).stem.replace("coco_0000000", "")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Phase 2 Content Correction Grid
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase2_figure(pipe, lpips_fn):
    print("\n" + "=" * 60)
    print("Figure 1: Phase 2 — DDIM Baseline vs Our Correction")
    print("=" * 60)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    steps = 50

    images = {}
    metrics_dict = {}

    for img_path in PHASE2_SELECTION:
        if not os.path.exists(img_path):
            continue
        name = _img_name(img_path)
        original_latent, original_tensor = load_image(pipe, img_path)

        key_orig = f"{name}\n(Original)"
        images[key_orig] = (original_tensor + 1) / 2
        metrics_dict[key_orig] = {}

        # Baseline DDIM
        noise = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
        recon_base = decode_latent(pipe, ddim_reconstruction(pipe, noise, prompt_embeds, steps))
        m_base = compute_metrics(original_tensor, recon_base, lpips_fn)
        key_base = f"{name}\nDDIM Baseline"
        images[key_base] = (recon_base + 1) / 2
        metrics_dict[key_base] = m_base

        # Corrected (ours)
        noise_c, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds, steps, layers)
        corrector = FeatureCorrector(pipe.unet, layers, lam=0.7)
        corrector.set_reference(saved, 0)
        recon_corr = decode_latent(pipe, ddim_reconstruction_with_correction(
            pipe, noise_c, prompt_embeds, steps, saved, corrector))
        corrector.remove()
        m_corr = compute_metrics(original_tensor, recon_corr, lpips_fn)
        key_corr = f"{name}\nOurs (λ=0.7)"
        images[key_corr] = (recon_corr + 1) / 2
        metrics_dict[key_corr] = m_corr

        delta = m_corr["PSNR"] - m_base["PSNR"]
        print(f"  {name}: baseline PSNR={m_base['PSNR']:.1f}, ours={m_corr['PSNR']:.1f}, Δ={delta:+.1f}")

    ncols = 3
    ordered = {}
    for img_path in PHASE2_SELECTION:
        if not os.path.exists(img_path):
            continue
        name = _img_name(img_path)
        ordered[f"{name}\n(Original)"] = images[f"{name}\n(Original)"]
        ordered[f"{name}\nDDIM Baseline"] = images[f"{name}\nDDIM Baseline"]
        ordered[f"{name}\nOurs (λ=0.7)"] = images[f"{name}\nOurs (λ=0.7)"]

    ref_tensor = (load_image(pipe, PHASE2_SELECTION[0])[1] + 1) / 2
    make_grid_image(ordered, OUT_DIR / "phase2_correction.png", ncols=ncols,
                    reference_tensor=ref_tensor, metrics_dict=metrics_dict)
    print(f"  -> {OUT_DIR / 'phase2_correction.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Phase 3 Framework — Prompt-based Style with Content Protection
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase3_figure(pipe, lpips_fn):
    print("\n" + "=" * 60)
    print("Figure 2: Phase 3 — Prompt-based Style with Content Protection")
    print("=" * 60)
    extractor = CLIPFeatureExtractor()
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    steps = 50

    content_path = PHASE3_CONTENT[0]
    if not os.path.exists(content_path):
        print("  [SKIP] missing content image")
        return

    content_name = _img_name(content_path)
    original_latent, original_tensor = load_image(pipe, content_path)

    # Style via prompt embedding interpolation
    style_strength = 0.5
    styled_emb = make_styled_prompt_embedding(pipe, "", STYLE_PROMPT_WATERCOLOR, strength=style_strength)

    v_content = extractor.encode_text("a photo")

    images = {}
    metrics_dict = {}

    # Original
    key_orig = f"Content\n{content_name}"
    images[key_orig] = (original_tensor + 1) / 2
    metrics_dict[key_orig] = {}

    # Baseline DDIM
    bm, brecon, _ = run_baseline(pipe, original_latent, original_tensor, prompt_embeds, steps, lpips_fn)
    key_base = "DDIM Baseline"
    images[key_base] = (brecon + 1) / 2
    metrics_dict[key_base] = bm

    # Style injection WITHOUT protection (prompt only, no correction)
    sm, srecon, _ = run_correction_with_style(
        pipe, original_latent, original_tensor, prompt_embeds,
        steps, corr_lam=0.0, corr_layers=[],
        styled_prompt_embeds=styled_emb, lpips_fn=lpips_fn)
    key_style_only = f"Style Only\n\"{STYLE_PROMPT_WATERCOLOR}\""
    images[key_style_only] = (srecon + 1) / 2
    metrics_dict[key_style_only] = sm

    # Our full framework: corr + style + pinning
    pm, precon, _, pin_log = run_correction_with_style_and_pinning(
        pipe, original_latent, original_tensor, prompt_embeds,
        steps, corr_lam=0.5, corr_layers=layers,
        styled_prompt_embeds=styled_emb,
        extractor=extractor, v_content=v_content,
        lpips_fn=lpips_fn,
        pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5)
    triggered = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
    key_ours = f"Ours\n(corr+style+pin)"
    images[key_ours] = (precon + 1) / 2
    metrics_dict[key_ours] = pm

    print(f"  Baseline:      PSNR={bm['PSNR']:.1f}  LPIPS={bm.get('LPIPS',0):.3f}")
    print(f"  Style Only:    PSNR={sm['PSNR']:.1f}  LPIPS={sm.get('LPIPS',0):.3f}")
    print(f"  Ours:          PSNR={pm['PSNR']:.1f}  LPIPS={pm.get('LPIPS',0):.3f}  pin={triggered}/{len(pin_log)}")

    make_grid_image(images, OUT_DIR / "phase3_framework.png", ncols=4,
                    reference_tensor=(original_tensor + 1) / 2, metrics_dict=metrics_dict)
    print(f"  -> {OUT_DIR / 'phase3_framework.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Direction Control — SLERP between two text-based style directions
# ═══════════════════════════════════════════════════════════════════════════

def generate_direction_figure(pipe, lpips_fn):
    print("\n" + "=" * 60)
    print("Figure 4: Direction Control — SLERP Style Interpolation")
    print("=" * 60)
    extractor = CLIPFeatureExtractor()
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    steps = 50

    content_path = PHASE3_CONTENT[0]
    if not os.path.exists(content_path):
        print("  [SKIP] missing content image")
        return

    content_name = _img_name(content_path)
    original_latent, original_tensor = load_image(pipe, content_path)

    v_content = extractor.encode_text("a photo")

    # Both style directions from CLIP text
    v_text_a = extractor.encode_text(STYLE_PROMPT_WATERCOLOR)
    _, v_style_a, _ = extractor.compute_orthogonal_decomposition(v_text_a, v_content)

    v_text_b = extractor.encode_text("a cyberpunk scene with neon lights")
    _, v_style_b, _ = extractor.compute_orthogonal_decomposition(v_text_b, v_content)

    print(f"  Style A (watercolor): |v_style|={v_style_a.norm().item():.3f}")
    print(f"  Style B (cyberpunk): |v_style|={v_style_b.norm().item():.3f}")

    images = {}
    metrics_dict = {}

    images[f"Content\n{content_name}"] = (original_tensor + 1) / 2
    metrics_dict[f"Content\n{content_name}"] = {}

    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v_interp = slerp(v_style_a, v_style_b, t_val)
        styled_emb = build_style_cross_attn_tokens(pipe, "", v_interp, strength=0.5, mode="extra_token")
        sm, srecon, _ = run_correction_with_style(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.5, corr_layers=layers,
            styled_prompt_embeds=styled_emb, lpips_fn=lpips_fn)
        key = f"t={t_val:.2f}\nwatercolor→cyberpunk"
        images[key] = (srecon + 1) / 2
        metrics_dict[key] = sm
        print(f"  t={t_val:.2f}: PSNR={sm['PSNR']:.1f}  LPIPS={sm.get('LPIPS',0):.3f}")

    make_grid_image(images, OUT_DIR / "direction_interpolation.png", ncols=6,
                    reference_tensor=(original_tensor + 1) / 2, metrics_dict=metrics_dict)
    print(f"  -> {OUT_DIR / 'direction_interpolation.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Phase 2 Ablation — Layer group comparison
# ═══════════════════════════════════════════════════════════════════════════

def generate_ablation_figure(pipe, lpips_fn):
    """Compare layer groups on 3 images, show visual from first + aggregated metrics."""
    print("\n" + "=" * 60)
    print("Figure 5: Phase 2 — Ablation: Layer Group Comparison (3 images)")
    print("=" * 60)
    from phase2_common import (ddim_inversion_with_latents, ddim_reconstruction_with_latent_correction,
                               ENCODER_TOP5, ATTENTION_TOP5, RANDOM_5_UP_RESNETS)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers_top5 = get_top_drift_layers(5)
    steps = 50
    lam = 0.7

    content_images = PHASE2_SELECTION[:3]  # 285, 872, 139
    display_img_path = content_images[0]

    if not os.path.exists(display_img_path):
        print("  [SKIP] missing image")
        return

    agg = {}  # label -> list of per-image delta PSNR

    for img_idx, img_path in enumerate(content_images):
        if not os.path.exists(img_path):
            continue
        name = _img_name(img_path)
        original_latent, original_tensor = load_image(pipe, img_path)

        noise_b = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
        recon_b = decode_latent(pipe, ddim_reconstruction(pipe, noise_b, prompt_embeds, steps))
        m_b = compute_metrics(original_tensor, recon_b, lpips_fn)
        b_psnr = m_b["PSNR"]

        if img_idx == 0:
            images_display = {f"Original\n{name}": (original_tensor + 1) / 2}
            images_display["DDIM Baseline"] = (recon_b + 1) / 2

        for label, layer_group in [
            ("top5", layers_top5),
            ("random5", RANDOM_5_UP_RESNETS),
            ("encoder5", ENCODER_TOP5),
            ("attention5", ATTENTION_TOP5),
        ]:
            noise, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds, steps, layer_group)
            corrector = FeatureCorrector(pipe.unet, layer_group, lam)
            corrector.set_reference(saved, 0)
            recon = decode_latent(pipe, ddim_reconstruction_with_correction(
                pipe, noise, prompt_embeds, steps, saved, corrector))
            corrector.remove()
            m = compute_metrics(original_tensor, recon, lpips_fn)
            delta = m["PSNR"] - b_psnr
            agg.setdefault(label, []).append(delta)
            if img_idx == 0:
                images_display[label] = (recon + 1) / 2

        # latent_interp
        noise_l, saved_latents = ddim_inversion_with_latents(pipe, original_latent, prompt_embeds, steps)
        recon_l = decode_latent(pipe, ddim_reconstruction_with_latent_correction(
            pipe, noise_l, prompt_embeds, steps, saved_latents, lam))
        m = compute_metrics(original_tensor, recon_l, lpips_fn)
        agg.setdefault("latent_interp", []).append(m["PSNR"] - b_psnr)
        if img_idx == 0:
            images_display["latent_interp"] = (recon_l + 1) / 2

        print(f"  [{img_idx+1}/3] {name}: baseline PSNR={b_psnr:.1f}")
        del original_latent, original_tensor
        torch.cuda.empty_cache()

    metrics_display = {}
    metrics_display[f"Original\n{_img_name(content_images[0])}"] = {}
    metrics_display["DDIM Baseline"] = {}
    for label in ["top5", "random5", "encoder5", "attention5", "latent_interp"]:
        vals = agg.get(label, [0])
        avg_d = sum(vals) / len(vals)
        metrics_display[label] = {"ΔPSNR_avg": round(avg_d, 2)}
        print(f"  {label}: avg Δ={avg_d:+.2f} (n={len(vals)})")

    ordered = {
        "Original": images_display[f"Original\n{_img_name(content_images[0])}"],
        "DDIM Baseline": images_display["DDIM Baseline"],
        "top5": images_display["top5"],
        "random5": images_display["random5"],
        "encoder5": images_display["encoder5"],
        "attention5": images_display["attention5"],
        "latent_interp": images_display["latent_interp"],
    }
    ordered_metrics = {k: metrics_display.get(k, {}) for k in ordered}
    ref_tensor = (load_image(pipe, display_img_path)[1] + 1) / 2
    make_grid_image(ordered, OUT_DIR / "phase2_ablation.png", ncols=4,
                    reference_tensor=ref_tensor, metrics_dict=ordered_metrics)
    print(f"  -> {OUT_DIR / 'phase2_ablation.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# DCSC Figure: Pareto Frontier + Control Trajectory + Stability
# ═══════════════════════════════════════════════════════════════════════════

def generate_dcsc_figure(pipe, lpips_fn):
    """Thesis Figure: DCSC closed-loop style controller.

    Generates 3-panel figure:
      1. Pareto frontier (CLIP_style vs CLIP_content) for all methods
      2. Control trajectory (lambda, d_content, basis over steps)
      3. Stability bound verification
    """
    from dcsc_experiment import (
        compare_across_methods, compute_pareto_frontier,
        plot_pareto_frontier, generate_comparison_table,
    )
    from dcsc_stability import (
        estimate_lipschitz_constant, verify_empirical_stability,
        plot_stability_grid, plot_stability_pass_rate,
    )

    print("\n" + "=" * 60)
    print("DCSC Figure: Pareto + Control + Stability")
    print("=" * 60)

    extractor = CLIPFeatureExtractor()
    corr_layers = get_top_drift_layers(5)
    v_content = extractor.encode_text("a photo")
    style_text = "an oil painting in impressionist style"

    # Use 5 coco_val images for Pareto scan
    pareto_images = [
        "data/coco_val/coco_000000000139.jpg",
        "data/coco_val/coco_000000000285.jpg",
        "data/coco_val/coco_000000000632.jpg",
        "data/coco_val/coco_000000000724.jpg",
        "data/coco_val/coco_000000000776.jpg",
    ]
    pareto_images = [p for p in pareto_images if os.path.exists(p)]

    out_dir = OUT_DIR / "dcsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Panel 1: Fair Pareto Frontier Comparison ----
    print(f"\n[Panel 1] Pareto frontier comparison  style='{style_text}'")
    compare_results = compare_across_methods(
        pipe, pareto_images, extractor, style_text, v_content,
        num_steps=50, corr_lam=0.5, corr_layers=corr_layers,
        lpips_fn=lpips_fn,
        dcsc_Kp=1.0, dcsc_lambda_0=0.5,
    )
    plot_pareto_frontier(
        compare_results,
        str(out_dir / "dcsc_pareto.png"),
        title=f"Style-Content Pareto Frontier — '{style_text[:40]}...'")
    generate_comparison_table(
        compare_results, str(out_dir / "dcsc_comparison.tex"))

    # Print per-method averages
    print("  Method averages:")
    for method in ["DDIM", "Correction", "DCSC"]:
        entries = compare_results.get(method, [])
        if entries:
            import numpy as np
            psnr = np.mean([e["PSNR"] for e in entries])
            cs = np.mean([e.get("CLIP_style", 0) for e in entries])
            cc = np.mean([e.get("CLIP_content", 0) for e in entries])
            print(f"    {method:15s}: PSNR={psnr:.2f} CLIP_s={cs:.3f} CLIP_c={cc:.3f}")

    # ---- Panel 2: Empirical Stability Analysis ----
    print(f"\n[Panel 2] Empirical stability analysis...")
    L = estimate_lipschitz_constant(extractor, pipe, pareto_images[:3], n_pairs=30)
    stability_results = verify_empirical_stability(
        pipe, pareto_images[:3], extractor, style_text, v_content,
        num_steps=50, corr_lam=0.5,
        Kp_values=[0.5, 1.0, 2.0, 5.0],
        lambda_0_values=[0.3, 0.5, 0.7],
        control_freq=5, lpips_fn=lpips_fn,
    )
    stability_out = out_dir / "stability"
    stability_out.mkdir(exist_ok=True)
    plot_stability_grid(stability_results, str(stability_out / "stability_grid.png"))
    plot_stability_pass_rate(stability_results, str(stability_out / "stability_pass_rate.png"))

    n_stable = stability_results["n_stable"]
    n_total = stability_results["n_total"]
    print(f"\n  L_estimate={L:.4f}")
    print(f"  Stability: {n_stable}/{n_total} stable ({100*n_stable/max(n_total,1):.1f}%)")
    print(f"  Output: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# User Study Materials
# ═══════════════════════════════════════════════════════════════════════════

def generate_user_study_pairs(pipe, lpips_fn):
    print("\n" + "=" * 60)
    print("User Study: Pairwise Comparison Pairs")
    print("=" * 60)
    extractor = CLIPFeatureExtractor()
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    steps = 50
    study_dir = OUT_DIR / "comparison_pairs"
    os.makedirs(study_dir, exist_ok=True)

    # ── Type 1: Content Preservation (Baseline vs Ours) ──
    print("\n--- Type 1: Content Preservation (10 pairs) ---")
    for img_path in USER_STUDY_CONTENT[:10]:
        if not os.path.exists(img_path):
            continue
        name = _img_name(img_path)
        original_latent, original_tensor = load_image(pipe, img_path)

        noise = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
        recon_a = decode_latent(pipe, ddim_reconstruction(pipe, noise, prompt_embeds, steps))
        m_a = compute_metrics(original_tensor, recon_a, lpips_fn)

        noise_c, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds, steps, layers)
        corrector = FeatureCorrector(pipe.unet, layers, lam=0.7)
        corrector.set_reference(saved, 0)
        recon_b = decode_latent(pipe, ddim_reconstruction_with_correction(
            pipe, noise_c, prompt_embeds, steps, saved, corrector))
        corrector.remove()
        m_b = compute_metrics(original_tensor, recon_b, lpips_fn)

        pair_dir = study_dir / f"content_{name}"
        os.makedirs(pair_dir, exist_ok=True)
        save_recon_img(recon_a, str(pair_dir), "A_baseline", steps, "")
        save_recon_img(recon_b, str(pair_dir), "B_corrected", steps, "")
        save_recon_img(original_tensor, str(pair_dir), "reference", steps, "")
        with open(pair_dir / "info.json", "w") as f:
            json.dump({
                "task": "content_preservation", "image": name,
                "A": {"method": "DDIM baseline", "PSNR": round(m_a["PSNR"], 2),
                      "SSIM": round(m_a["SSIM"], 4), "LPIPS": round(m_a.get("LPIPS", 0), 4)},
                "B": {"method": "DDIM+Corr (ours)", "PSNR": round(m_b["PSNR"], 2),
                      "SSIM": round(m_b["SSIM"], 4), "LPIPS": round(m_b.get("LPIPS", 0), 4)},
            }, f, indent=2)
        print(f"  {name}: baseline={m_a['PSNR']:.1f} -> ours={m_b['PSNR']:.1f} (Δ={m_b['PSNR']-m_a['PSNR']:+.1f})")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    # ── Type 2: Style Safety (prompt style_only vs our framework) ──
    print("\n--- Type 2: Style Safety (5 pairs) ---")
    v_content = extractor.encode_text("a photo")
    style_strength = 0.5
    style_prompt = STYLE_PROMPT_WATERCOLOR

    for content_path in PHASE3_CONTENT:
        if not os.path.exists(content_path):
            continue
        c_name = _img_name(content_path)
        original_latent, original_tensor = load_image(pipe, content_path)
        styled_emb = make_styled_prompt_embedding(pipe, "", style_prompt, strength=style_strength)

        sm_a, recon_a, _ = run_correction_with_style(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.0, corr_layers=[],
            styled_prompt_embeds=styled_emb, lpips_fn=lpips_fn)

        sm_b, recon_b, _, pin_log = run_correction_with_style_and_pinning(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.5, corr_layers=layers,
            styled_prompt_embeds=styled_emb,
            extractor=extractor, v_content=v_content,
            lpips_fn=lpips_fn,
            pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5)

        triggered = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
        pair_dir = study_dir / f"style_safety_{c_name}"
        os.makedirs(pair_dir, exist_ok=True)
        save_recon_img(recon_a, str(pair_dir), "A_style_only", steps, "")
        save_recon_img(recon_b, str(pair_dir), "B_framework", steps, "")
        save_recon_img(original_tensor, str(pair_dir), "reference", steps, "")
        with open(pair_dir / "info.json", "w") as f:
            json.dump({
                "task": "style_safety", "content_image": c_name,
                "style_prompt": style_prompt,
                "A": {"method": "Style injection w/o protection", "PSNR": round(sm_a["PSNR"], 2),
                      "LPIPS": round(sm_a.get("LPIPS", 0), 4)},
                "B": {"method": "Our framework (corr+style+pin)", "PSNR": round(sm_b["PSNR"], 2),
                      "LPIPS": round(sm_b.get("LPIPS", 0), 4)},
                "pinning_triggered": f"{triggered}/{len(pin_log)}" if pin_log else "N/A",
            }, f, indent=2)
        print(f"  {c_name}: style_only={sm_a['PSNR']:.1f} -> ours={sm_b['PSNR']:.1f} (Δ={sm_b['PSNR']-sm_a['PSNR']:+.1f})")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    # Extra pairs for content images not in PHASE3_CONTENT
    extra_content = ["data/coco_val/coco_000000000776.jpg", "data/coco_val/coco_000000001000.jpg"]
    for content_path in extra_content:
        if not os.path.exists(content_path):
            continue
        c_name = _img_name(content_path)
        original_latent, original_tensor = load_image(pipe, content_path)
        styled_emb = make_styled_prompt_embedding(pipe, "", style_prompt, strength=style_strength)

        sm_a, recon_a, _ = run_correction_with_style(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.0, corr_layers=[],
            styled_prompt_embeds=styled_emb, lpips_fn=lpips_fn)

        sm_b, recon_b, _, pin_log = run_correction_with_style_and_pinning(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.5, corr_layers=layers,
            styled_prompt_embeds=styled_emb,
            extractor=extractor, v_content=v_content,
            lpips_fn=lpips_fn,
            pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5)

        triggered = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
        pair_dir = study_dir / f"style_safety_{c_name}"
        os.makedirs(pair_dir, exist_ok=True)
        save_recon_img(recon_a, str(pair_dir), "A_style_only", steps, "")
        save_recon_img(recon_b, str(pair_dir), "B_framework", steps, "")
        save_recon_img(original_tensor, str(pair_dir), "reference", steps, "")
        with open(pair_dir / "info.json", "w") as f:
            json.dump({
                "task": "style_safety", "content_image": c_name,
                "style_prompt": style_prompt,
                "A": {"method": "Style injection w/o protection", "PSNR": round(sm_a["PSNR"], 2),
                      "LPIPS": round(sm_a.get("LPIPS", 0), 4)},
                "B": {"method": "Our framework (corr+style+pin)", "PSNR": round(sm_b["PSNR"], 2),
                      "LPIPS": round(sm_b.get("LPIPS", 0), 4)},
                "pinning_triggered": f"{triggered}/{len(pin_log)}" if pin_log else "N/A",
            }, f, indent=2)
        print(f"  {c_name}: style_only={sm_a['PSNR']:.1f} -> ours={sm_b['PSNR']:.1f} (Δ={sm_b['PSNR']-sm_a['PSNR']:+.1f})")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    print(f"\n  User study materials saved to: {study_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(pipe, lpips_fn):
    print("\n" + "=" * 60)
    print("Summary: Full coco_val evaluation (all 19 images)")
    print("=" * 60)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    steps = 50

    results = []
    for img_path in ALL_COCO:
        if not os.path.exists(img_path):
            continue
        name = _img_name(img_path)
        original_latent, original_tensor = load_image(pipe, img_path)

        noise = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
        recon_b = decode_latent(pipe, ddim_reconstruction(pipe, noise, prompt_embeds, steps))
        m_b = compute_metrics(original_tensor, recon_b, lpips_fn)

        noise_c, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds, steps, layers)
        corrector = FeatureCorrector(pipe.unet, layers, lam=0.7)
        corrector.set_reference(saved, 0)
        recon_c = decode_latent(pipe, ddim_reconstruction_with_correction(
            pipe, noise_c, prompt_embeds, steps, saved, corrector))
        corrector.remove()
        m_c = compute_metrics(original_tensor, recon_c, lpips_fn)

        results.append({
            "image": name,
            "baseline_PSNR": round(m_b["PSNR"], 2),
            "baseline_SSIM": round(m_b["SSIM"], 4),
            "baseline_LPIPS": round(m_b.get("LPIPS", 0), 4),
            "ours_PSNR": round(m_c["PSNR"], 2),
            "ours_SSIM": round(m_c["SSIM"], 4),
            "ours_LPIPS": round(m_c.get("LPIPS", 0), 4),
            "delta_PSNR": round(m_c["PSNR"] - m_b["PSNR"], 2),
        })
        print(f"  {name}: baseline={m_b['PSNR']:.1f} ours={m_c['PSNR']:.1f} Δ={m_c['PSNR']-m_b['PSNR']:+.2f}")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    deltas = [r["delta_PSNR"] for r in results]
    avg_b = sum(r["baseline_PSNR"] for r in results) / len(results)
    avg_c = sum(r["ours_PSNR"] for r in results) / len(results)
    avg_d = sum(deltas) / len(deltas)
    avg_lpips_b = sum(r["baseline_LPIPS"] for r in results) / len(results)
    avg_lpips_c = sum(r["ours_LPIPS"] for r in results) / len(results)

    summary = {
        "n_images": len(results),
        "dataset": "coco_val (19 images)",
        "steps": steps, "lambda": 0.7,
        "avg_baseline_PSNR": round(avg_b, 2),
        "avg_ours_PSNR": round(avg_c, 2),
        "avg_delta_PSNR": round(avg_d, 2),
        "min_delta": round(min(deltas), 2),
        "max_delta": round(max(deltas), 2),
        "avg_baseline_LPIPS": round(avg_lpips_b, 4),
        "avg_ours_LPIPS": round(avg_lpips_c, 4),
        "results": results,
    }

    with open(OUT_DIR / "coco_val_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Aggregate: avg baseline PSNR={avg_b:.2f}, avg ours={avg_c:.2f}, avg Δ={avg_d:+.2f}")
    print(f"  LPIPS: {avg_lpips_b:.4f} -> {avg_lpips_c:.4f}")
    print(f"  Δ range: [{min(deltas):+.2f}, {max(deltas):+.2f}]")
    print(f"  -> {OUT_DIR / 'coco_val_summary.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures and user study materials")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "figures", "user_study", "summary", "cleanup"])
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.mode == "cleanup":
        print("Cleanup mode: verifying output structure...")
        for d in [OUT_DIR, OUT_DIR / "comparison_pairs"]:
            if d.exists():
                count = len(list(d.rglob("*.png"))) + len(list(d.rglob("*.json")))
                print(f"  {d}: {count} files")
        return

    print("[0] Loading model...")
    pipe = load_pipeline()

    lpips_fn = None
    if not args.skip_lpips:
        import lpips
        print("[1] Loading LPIPS...")
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    if args.mode in ("all", "figures"):
        generate_phase2_figure(pipe, lpips_fn)
        generate_phase3_figure(pipe, lpips_fn)
        generate_direction_figure(pipe, lpips_fn)
        generate_ablation_figure(pipe, lpips_fn)
        if args.mode == "all":
            generate_dcsc_figure(pipe, lpips_fn)  # DCSC figures (Pareto + control + stability)

    if args.mode in ("all", "user_study"):
        generate_user_study_pairs(pipe, lpips_fn)

    if args.mode in ("all", "summary"):
        generate_summary(pipe, lpips_fn)

    print(f"\nDone. Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
