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
from clip_utils import (CLIPFeatureExtractor, build_style_cross_attn_tokens,
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
# Phase 5: Data-driven summary figures (no model inference, from existing JSON)
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase5_data_figures():
    """Generate thesis-ready summary figures from Phase 5 experimental data.
    No model inference — reads existing JSON outputs."""
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("Phase 5 Data Figures: SOTA table, step robustness, P2P parity")
    print("=" * 60)

    # --- Load Phase 5 summary ---
    p5_path = Path("outputs/phase5_final/final_summary.json")
    if not p5_path.exists():
        print("  [SKIP] Phase 5 summary not found — run phase5_final_comparison.py first")
        return

    with open(p5_path) as f:
        p5_data = json.load(f)

    summaries = p5_data["summaries"]

    # --- Figure A: Step count robustness curve ---
    # Data from Phase 2 full (step sweep)
    step_data = {
        4:  {"ddim": 17.02, "corr": 18.73, "delta": 1.72},
        10: {"ddim": 17.19, "corr": 21.66, "delta": 4.47},
        20: {"ddim": 19.24, "corr": 23.89, "delta": 4.65},
        50: {"ddim": 22.45, "corr": 25.20, "delta": 2.75},
        100: {"ddim": 23.63, "corr": 25.44, "delta": 1.81},
    }
    steps = sorted(step_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: PSNR curves
    ax = axes[0]
    ax.plot(steps, [step_data[s]["ddim"] for s in steps], "o-", color="#888888",
            linewidth=2, markersize=8, label="DDIM Baseline")
    ax.plot(steps, [step_data[s]["corr"] for s in steps], "s-", color="#2ecc71",
            linewidth=2, markersize=8, label="DDIM + ResCorr (Ours)")
    ax.fill_between(steps, [step_data[s]["ddim"] for s in steps],
                    [step_data[s]["corr"] for s in steps],
                    alpha=0.15, color="#2ecc71")
    ax.set_xlabel("DDIM Steps", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("PSNR vs DDIM Steps (19 coco_val)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Right: Delta curve
    ax = axes[1]
    deltas = [step_data[s]["delta"] for s in steps]
    ax.fill_between(steps, deltas, alpha=0.3, color="#3498db")
    ax.plot(steps, deltas, "o-", color="#2980b9", linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2)
    ax.axhline(y=0, color="black", linewidth=0.5)
    # Mark optimal
    best_step = max(steps, key=lambda s: step_data[s]["delta"])
    ax.annotate(f"Best: {best_step} steps\n(Δ={step_data[best_step]['delta']:+.1f} dB)",
                xy=(best_step, step_data[best_step]["delta"]),
                xytext=(best_step + 15, step_data[best_step]["delta"] + 0.3),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=11, fontweight="bold", color="#e74c3c")
    ax.set_xlabel("DDIM Steps", fontsize=13)
    ax.set_ylabel("ΔPSNR (Corr − Baseline, dB)", fontsize=13)
    ax.set_title("Correction Benefit vs Steps", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "step_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR / 'step_robustness.png'}")

    # --- Figure B: SOTA comparison bar chart (stylized for thesis) ---
    methods = ["DDIM", "NTI", "EDICT", "P2P", "Ours_Corr"]
    labels = ["DDIM\n(baseline)", "NTI\n(BLIP)", "EDICT", "P2P\n(cross-attn)", "Ours\nResCorr"]
    colors = ["#bdc3c7", "#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    means = [summaries[m]["PSNR"]["mean"] for m in methods]
    stds = [summaries[m]["PSNR"]["std"] for m in methods]
    ddim_mean = means[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=6,
                  alpha=0.92, edgecolor="white", linewidth=0.8, width=0.6)

    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title("Content Preservation on COCO val2017 (50 DDIM steps, n=19)",
                 fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i == 0:
            continue
        delta = mean - ddim_mean
        ax.annotate(f"$\\Delta$={delta:+.1f} dB",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold")

    # P2P vs Ours annotation
    p2p_m = means[3]
    ours_m = means[4]
    ax.annotate(f"P2P − Ours = {p2p_m - ours_m:+.2f} dB\n(Cohen's d < 0.05)",
                xy=(3.5, max(p2p_m, ours_m)),
                fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sota_barchart_thesis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR / 'sota_barchart_thesis.png'}")

    # --- Figure C: Per-image P2P vs Ours correlation ---
    per_img = p5_data["per_image"]
    p2p_vals = [per_img[img]["P2P"] for img in sorted(per_img.keys())]
    ours_vals = [per_img[img]["Ours_Corr"] for img in sorted(per_img.keys())]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(p2p_vals, ours_vals, c="#2ecc71", s=80, alpha=0.85,
               edgecolors="white", linewidth=0.8, zorder=5)

    lim_min = min(min(p2p_vals), min(ours_vals)) - 1
    lim_max = max(max(p2p_vals), max(ours_vals)) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.25,
            linewidth=1, label="y = x (equal performance)", zorder=1)

    r_val = np.corrcoef(p2p_vals, ours_vals)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {r_val:.4f}\nMean diff = {np.mean(p2p_vals)-np.mean(ours_vals):+.2f} dB",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("P2P Cross-Attention PSNR (dB)", fontsize=13)
    ax.set_ylabel("Ours ResCorr PSNR (dB)", fontsize=13)
    ax.set_title("Per-Image: P2P vs Ours (n=19)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "p2p_vs_ours_thesis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUT_DIR / 'p2p_vs_ours_thesis.png'} (r={r_val:.4f})")

    # --- Save updated summary JSON ---
    # Merge Phase 5 stats with summary format
    thesis_summary = {
        "dataset": "coco_val",
        "n_images": 19,
        "steps": 50,
        "sota_comparison": {
            m: {
                "PSNR": f"{summaries[m]['PSNR']['mean']:.2f} ± {summaries[m]['PSNR']['std']:.2f}",
                "SSIM": f"{summaries[m]['SSIM']['mean']:.3f} ± {summaries[m]['SSIM']['std']:.3f}",
                "LPIPS": f"{summaries[m]['LPIPS']['mean']:.3f} ± {summaries[m]['LPIPS']['std']:.3f}",
            }
            for m in methods if m in summaries
        },
        "step_robustness": step_data,
        "statistical_tests": {
            "p2p_vs_ours_ttest_p": 0.0015,
            "p2p_vs_ours_cohens_d": 0.033,
            "interpretation": "P2P has statistically detectable but practically negligible advantage (0.13 dB, Cohen's d=0.033 << 0.2 threshold for small effect)",
            "p2p_ours_pearson_r": round(float(r_val), 4),
        },
    }
    with open(OUT_DIR / "thesis_summary.json", "w") as f:
        json.dump(thesis_summary, f, indent=2)
    print(f"  -> {OUT_DIR / 'thesis_summary.json'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures and user study materials")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "figures", "user_study", "summary", "phase5", "cleanup"])
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

    # Phase 5: data-only figures, no model needed
    if args.mode == "phase5":
        generate_phase5_data_figures()
        print(f"\nDone. Output: {OUT_DIR.resolve()}")
        return

    # All other modes need the model
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

    if args.mode in ("all", "user_study"):
        generate_user_study_pairs(pipe, lpips_fn)

    if args.mode in ("all", "summary"):
        generate_summary(pipe, lpips_fn)

    if args.mode in ("all", "phase5"):
        generate_phase5_data_figures()

    print(f"\nDone. Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
