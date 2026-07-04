"""
DiT Phase 3: CLIP 正交投影 + 风格注入 + 钉扎约束 (HunyuanDiT)

在 HunyuanDiT 上验证风格解耦框架：
1. CLIP 正交投影（架构无关，直接复用）
2. 风格注入：
   - 跨注意力方式：修改 CLIP embedding（1024-dim，dual encoder 适配）
   - 特征偏置方式：DiTStyleFeatureInjector（768-dim v_style → 1408 hidden）
3. 正交钉扎约束（像素空间操作，架构无关，直接复用）

用法:
  python scripts/dit_phase3_prep.py --mode clip
  python scripts/dit_phase3_prep.py --mode inject --image data/coco_val/coco_xxx.jpg
  python scripts/dit_phase3_prep.py --mode full --image data/coco_val/coco_xxx.jpg
"""
import argparse, json, sys, time
from pathlib import Path

import torch, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase3_common import (
    CLIPFeatureExtractor, STYLE_CANDIDATES, find_closest_style, slerp,
)
from dit_phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_and_encode, decode_latent, encode_prompt_dit,
    FeatureCollector, FeatureCorrector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion_baseline, ddim_reconstruction_baseline,
    compute_metrics, save_recon_img,
    TOP_5_LAYERS, TOP_10_LAYERS, TRANSITION_ZONE,
)

OUT_DIR = Path("outputs/dit_phase3")
NUM_STEPS = 50


# ═══════════════════════════════════════════════════════════════
# DiT 风格注入：跨注意力方式
# ═══════════════════════════════════════════════════════════════

def build_dit_style_prompt_embeds(pipe, cond, v_style, strength=0.5):
    """Inject v_style into HunyuanDiT CLIP prompt embeddings.

    HunyuanDiT cross-attention: [B, 333, 1024] = CLIP(77) + T5(256→1024).
    Modifies only the CLIP part (first 77 tokens).

    Args:
        pipe: HunyuanDiTPipeline
        cond: base conditioning dict from encode_prompt_dit
        v_style: [1, 768] CLIP style vector (from CLIPFeatureExtractor)
        strength: [0, 1] injection strength

    Returns: modified encoder_hidden_states [1, 77, 1024]
    """
    clip_emb = cond["encoder_hidden_states"]  # [1, 77, 1024]
    v = v_style.to(dtype=clip_emb.dtype, device=clip_emb.device)  # [1, 768]

    # v_style is 768-dim, CLIP embeddings are 1024-dim
    # Option A: cyclic pad to 1024
    if v.shape[-1] < clip_emb.shape[-1]:
        repeats = (clip_emb.shape[-1] + v.shape[-1] - 1) // v.shape[-1]
        v_pad = v.repeat(1, repeats)[:, :clip_emb.shape[-1]]  # [1, 1024]
    else:
        v_pad = v

    # Interpolate v_pad into all CLIP tokens
    v_clip = v_pad.unsqueeze(1).repeat(1, clip_emb.shape[1], 1)  # [1, 77, 1024]
    styled_clip = (1.0 - strength) * clip_emb + strength * v_clip

    return styled_clip


# ═══════════════════════════════════════════════════════════════
# DiT 风格注入：特征偏置方式
# ═══════════════════════════════════════════════════════════════

class DiTStyleFeatureInjector:
    """Inject CLIP v_style as per-token feature bias in HunyuanDiT blocks.

    v_style (768-dim CLIP) → cyclic pad → 1408-dim → per-token addition bias.
    Training-free: only cyclic padding, no learned projection.
    """

    def __init__(self, transformer, layers, v_style, strength=0.5):
        self.transformer = transformer
        self.v_style = v_style.to(device=DEVICE, dtype=torch.float32)
        self.strength = strength
        self.handles = []

        for name in layers:
            mod = self._find(name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook(out)
                )
                self.handles.append(h)

    def _find(self, name):
        mod = self.transformer
        for t in name.split("."):
            try:
                mod = getattr(mod, t)
            except AttributeError:
                return None
        return mod

    def _hook(self, output):
        """Apply v_style as per-token bias.

        output: [B, N, D] = [1, 4096, 1408]  (for 1024x1024, patch_size=2)
        v_style: [1, 768] CLIP multimodal embedding
        """
        # Cyclic pad v_style from 768 to hidden_dim (1408)
        D = output.shape[-1]
        v = self.v_style.to(dtype=output.dtype, device=output.device)
        repeats = (D + v.shape[-1] - 1) // v.shape[-1]
        v_pad = v.repeat(1, repeats)[:, :D]  # [1, 1408]
        bias = v_pad.unsqueeze(1)  # [1, 1, 1408]

        corrected = output + self.strength * bias
        return (corrected,) + output[1:] if isinstance(output, tuple) else corrected

    def set_strength(self, strength):
        self.strength = max(0.0, strength)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ═══════════════════════════════════════════════════════════════
# 完整 Pipeline: 校正 + 风格注入 + 钉扎
# ═══════════════════════════════════════════════════════════════

def run_correction_with_style_and_pinning_dit(
    pipe, original_latent, original_tensor, cond,
    num_steps, corr_lam, corr_layers,
    styled_clip_emb, t5_cond,  # styled CLIP + original T5
    extractor, v_content,
    style_injector=None,
    pinning_freq=10, pinning_threshold=0.02, pinning_strength=0.5,
):
    """Phase 2 correction + style injection + orthogonal pinning for DiT.

    Args:
        pipe: HunyuanDiTPipeline
        cond: original (baseline) conditioning dict
        styled_clip_emb: [1, 77, 1024] modified CLIP embeddings
        t5_cond: dict with T5-specific keys (unchanged)
        extractor: CLIPFeatureExtractor for pinning monitor
        v_content: [1, 768] reference content vector
    """
    from dit_phase2_common import dit_forward

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # DDIM inversion with feature collection
    noise, saved_features = ddim_inversion_with_features(
        pipe, original_latent, cond, num_steps, corr_layers,
    )

    # Compute reference content projection from the original image
    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(original_tensor), v_content)

    # Setup corrector and optional style injector
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.transformer, corr_layers, sched)

    pinning_log = []
    effective_strength = style_injector.strength if style_injector else 0.5
    base_strength = effective_strength

    z = noise.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            # Step-matched correction reference
            if t_int in saved_features:
                corrector.set_reference(saved_features[t_int], step_idx)
            else:
                corrector.set_reference({}, step_idx)

            # Build full conditioning: styled CLIP + original T5
            full_cond = {
                "encoder_hidden_states": styled_clip_emb,
                "text_embedding_mask": cond["text_embedding_mask"],
                "encoder_hidden_states_t5": cond["encoder_hidden_states_t5"],
                "text_embedding_mask_t5": cond["text_embedding_mask_t5"],
            }

            noise_pred = dit_forward(pipe.transformer, z, t, full_cond)
            z = scheduler.step(noise_pred, t, z).prev_sample

            # Pinning check every pinning_freq steps
            if (step_idx + 1) % pinning_freq == 0 and step_idx < len(timesteps) - 1:
                decoded = decode_latent(pipe, z)
                v_current = extractor.encode_image_from_tensor(decoded)
                cur_proj = extractor.compute_content_projection(v_current, v_content)
                deviation = abs(cur_proj - ref_proj)

                triggered = deviation > pinning_threshold
                if triggered and style_injector is not None:
                    scale = max(0.0, 1.0 - pinning_strength * deviation / max(ref_proj, 1e-8))
                    effective_strength = base_strength * scale
                    style_injector.set_strength(effective_strength)

                pinning_log.append({
                    "step": step_idx, "t": t_int,
                    "cur_proj": float(cur_proj), "ref_proj": float(ref_proj),
                    "deviation": float(deviation), "triggered": triggered,
                    "effective_strength": effective_strength,
                })

    corrector.remove()
    if style_injector is not None:
        style_injector.remove()

    recon = decode_latent(pipe, z)
    metrics = compute_metrics(original_tensor, recon)

    return metrics, recon, pinning_log


# ═══════════════════════════════════════════════════════════════
# Modes
# ═══════════════════════════════════════════════════════════════

def mode_clip():
    """验证 CLIP 正交投影数学（架构无关，直接复用）。"""
    print("=" * 60)
    print("DiT Phase 3: CLIP Orthogonal Projection")
    print("=" * 60)

    extractor = CLIPFeatureExtractor()
    print(f"\nCLIP model: openai/clip-vit-large-patch14 (dim={extractor.hidden_dim})")

    # Load test image
    img_path = "data/coco_val/coco_000000000285.jpg"
    v_img = extractor.encode_image(img_path)
    v_content = extractor.encode_text("a photo of natural scene")
    print(f"\nImage: {Path(img_path).name}")
    print(f"  |v_img| = {v_img.norm().item():.4f}")
    print(f"  |v_content| = {v_content.norm().item():.4f}")
    print(f"  cos(v_img, v_content) = {(v_img * v_content).sum().item():.4f}")

    # Orthogonal decomposition
    proj, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
    print(f"\nOrthogonal Decomposition (StyleTex):")
    print(f"  |proj| = {proj.norm().item():.4f}")
    print(f"  |v_style| = {v_style.norm().item():.4f}")
    print(f"  cos(v_style, v_content) = {cos:.6f} (should be ≈ 0)")

    # Find closest style
    style_text, style_key, sim = find_closest_style(v_style, extractor)
    print(f"\nClosest style candidate: '{style_text}' (sim={sim:.4f})")
    print(f"Note: Style injection into HunyuanDiT will use v_style (768-dim CLIP)")
    print(f"      projected to match DiT's hidden dimension (1408) via cyclic padding.")

    return extractor, v_content


def mode_inject(pipe, img_path, extractor, v_content):
    """风格注入：cross-attention + feature bias 两种方式。"""
    print(f"\n{'='*60}")
    print("DiT Style Injection")
    print(f"{'='*60}")

    cond = encode_prompt_dit(pipe, "")
    latents, tensor, _ = load_and_encode(pipe, img_path)
    img_name = Path(img_path).stem

    # Get style direction
    v_img = extractor.encode_image(img_path)
    _, v_style, _ = extractor.compute_orthogonal_decomposition(v_img, v_content)

    # Find style target text
    style_text, style_key, _ = find_closest_style(v_style, extractor)
    print(f"\nImage: {img_name}")
    print(f"Style candidate: {style_key} ('{style_text}')")

    # Method 1: Cross-attention style injection
    print(f"\n[1] Cross-attention style injection...")
    t0 = time.perf_counter()
    styled_clip = build_dit_style_prompt_embeds(pipe, cond, v_style, strength=0.5)

    noise = ddim_inversion_baseline(pipe, latents, cond, NUM_STEPS)
    # Reconstruction with styled CLIP (T5 unchanged)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    from dit_phase2_common import dit_forward
    z = noise.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            full_cond = {
                "encoder_hidden_states": styled_clip,
                "text_embedding_mask": cond["text_embedding_mask"],
                "encoder_hidden_states_t5": cond["encoder_hidden_states_t5"],
                "text_embedding_mask_t5": cond["text_embedding_mask_t5"],
            }
            noise_pred = dit_forward(pipe.transformer, z, t, full_cond)
            z = scheduler.step(noise_pred, t, z).prev_sample

    recon_cross = decode_latent(pipe, z)
    metrics_cross = compute_metrics(tensor, recon_cross)
    elapsed = time.perf_counter() - t0
    print(f"  PSNR={metrics_cross['PSNR']:.2f}  LPIPS={metrics_cross['LPIPS']:.4f}  ({elapsed:.1f}s)")
    save_recon_img(recon_cross, OUT_DIR / f"{img_name}_style_cross_attn.png")

    # Method 2: Feature bias style injection
    print(f"\n[2] Feature bias style injection...")
    t0 = time.perf_counter()
    injector = DiTStyleFeatureInjector(
        pipe.transformer, TOP_5_LAYERS, v_style, strength=0.5,
    )

    noise2 = ddim_inversion_baseline(pipe, latents, cond, NUM_STEPS)
    z2 = noise2.clone()
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = dit_forward(pipe.transformer, z2, t, cond)
            z2 = scheduler.step(noise_pred, t, z2).prev_sample

    injector.remove()
    recon_bias = decode_latent(pipe, z2)
    metrics_bias = compute_metrics(tensor, recon_bias)
    elapsed = time.perf_counter() - t0
    print(f"  PSNR={metrics_bias['PSNR']:.2f}  LPIPS={metrics_bias['LPIPS']:.4f}  ({elapsed:.1f}s)")
    save_recon_img(recon_bias, OUT_DIR / f"{img_name}_style_feat_bias.png")

    # Style transfer metrics
    v_cross = extractor.encode_image_from_tensor(recon_cross)
    v_bias = extractor.encode_image_from_tensor(recon_bias)
    sim_cross = float((v_cross * v_style).sum())
    sim_bias = float((v_bias * v_style).sum())
    sim_orig = float((v_img * v_style).sum())

    print(f"\nStyle alignment (cos with v_style):")
    print(f"  Original:     {sim_orig:.4f}")
    print(f"  Cross-attn:   {sim_cross:.4f} {'(+style)' if sim_cross > sim_orig else ''}")
    print(f"  Feature bias: {sim_bias:.4f} {'(+style)' if sim_bias > sim_orig else ''}")

    return metrics_cross, metrics_bias


def mode_full(pipe, img_path, extractor, v_content):
    """完整 pipeline: Phase 2 校正 + 风格注入 + 钉扎约束。"""
    print(f"\n{'='*60}")
    print("DiT Full Pipeline: Correction + Style + Pinning")
    print(f"{'='*60}")

    cond = encode_prompt_dit(pipe, "")
    latents, tensor, _ = load_and_encode(pipe, img_path)
    img_name = Path(img_path).stem

    # Get style direction
    v_img = extractor.encode_image(img_path)
    _, v_style, _ = extractor.compute_orthogonal_decomposition(v_img, v_content)
    style_text, style_key, _ = find_closest_style(v_style, extractor)
    print(f"\nImage: {img_name}")
    print(f"Style: {style_key}")

    # 1. Baseline (no correction, no style)
    print(f"\n[1/4] Baseline (DDIM inversion + reconstruction)...")
    t0 = time.perf_counter()
    noise_bl = ddim_inversion_baseline(pipe, latents, cond, NUM_STEPS)
    recon_bl = ddim_reconstruction_baseline(pipe, noise_bl, cond, NUM_STEPS)
    recon_bl_img = decode_latent(pipe, recon_bl)
    metrics_bl = compute_metrics(tensor, recon_bl_img)
    print(f"  PSNR={metrics_bl['PSNR']:.2f}  LPIPS={metrics_bl['LPIPS']:.4f}")
    save_recon_img(recon_bl_img, OUT_DIR / f"{img_name}_baseline.png")

    # 2. Correction only (Phase 2)
    print(f"\n[2/4] Correction only (Phase 2, top-5, λ=0.7)...")
    t0 = time.perf_counter()
    noise_corr, saved = ddim_inversion_with_features(
        pipe, latents, cond, NUM_STEPS, TOP_5_LAYERS,
    )
    sched = LambdaScheduler(0.7, NUM_STEPS, "constant")
    corrector = FeatureCorrector(pipe.transformer, TOP_5_LAYERS, sched)
    recon_corr = ddim_reconstruction_with_correction(
        pipe, noise_corr, cond, NUM_STEPS, corrector, saved,
    )
    corrector.remove()
    recon_corr_img = decode_latent(pipe, recon_corr)
    metrics_corr = compute_metrics(tensor, recon_corr_img)
    print(f"  PSNR={metrics_corr['PSNR']:.2f}  LPIPS={metrics_corr['LPIPS']:.4f}")
    save_recon_img(recon_corr_img, OUT_DIR / f"{img_name}_correction.png")

    # 3. Correction + Style (no pinning)
    print(f"\n[3/4] Correction + Style (no pinning)...")
    styled_clip = build_dit_style_prompt_embeds(pipe, cond, v_style, strength=0.5)
    injector = DiTStyleFeatureInjector(pipe.transformer, TOP_5_LAYERS, v_style, strength=0.3)

    noise_s, saved_s = ddim_inversion_with_features(
        pipe, latents, cond, NUM_STEPS, TOP_5_LAYERS,
    )
    sched_s = LambdaScheduler(0.7, NUM_STEPS, "constant")
    corrector_s = FeatureCorrector(pipe.transformer, TOP_5_LAYERS, sched_s)

    from dit_phase2_common import dit_forward
    z_s = noise_s.clone()
    scheduler = pipe.scheduler; scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    with torch.no_grad():
        for step_idx, t in enumerate(scheduler.timesteps):
            t_int = int(t)
            if t_int in saved_s:
                corrector_s.set_reference(saved_s[t_int], step_idx)
            full_cond = {
                "encoder_hidden_states": styled_clip,
                "text_embedding_mask": cond["text_embedding_mask"],
                "encoder_hidden_states_t5": cond["encoder_hidden_states_t5"],
                "text_embedding_mask_t5": cond["text_embedding_mask_t5"],
            }
            noise_pred = dit_forward(pipe.transformer, z_s, t, full_cond)
            z_s = scheduler.step(noise_pred, t, z_s).prev_sample

    corrector_s.remove()
    injector.remove()
    recon_style_img = decode_latent(pipe, z_s)
    metrics_style = compute_metrics(tensor, recon_style_img)
    print(f"  PSNR={metrics_style['PSNR']:.2f}  LPIPS={metrics_style['LPIPS']:.4f}")
    save_recon_img(recon_style_img, OUT_DIR / f"{img_name}_style_no_pin.png")

    # 4. Correction + Style + Pinning
    print(f"\n[4/4] Correction + Style + Pinning...")
    injector_pin = DiTStyleFeatureInjector(pipe.transformer, TOP_5_LAYERS, v_style, strength=0.3)

    metrics_pin, recon_pin, pin_log = run_correction_with_style_and_pinning_dit(
        pipe, latents, tensor, cond, NUM_STEPS, 0.7, TOP_5_LAYERS,
        styled_clip, cond, extractor, v_content,
        style_injector=injector_pin,
        pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5,
    )
    print(f"  PSNR={metrics_pin['PSNR']:.2f}  LPIPS={metrics_pin['LPIPS']:.4f}")
    n_triggers = sum(1 for p in pin_log if p["triggered"])
    print(f"  Pinning: {n_triggers}/{len(pin_log)} checks triggered")
    save_recon_img(recon_pin, OUT_DIR / f"{img_name}_style_pin.png")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Method':<30}{'PSNR':>8}{'LPIPS':>8}")
    print(f"{'-'*46}")
    for name, m in [("Baseline", metrics_bl), ("Correction", metrics_corr),
                     ("Style (no pin)", metrics_style), ("Style + Pin", metrics_pin)]:
        print(f"{name:<30}{m['PSNR']:>8.2f}{m['LPIPS']:>8.4f}")

    # Save results
    results = [
        {"method": "baseline", **metrics_bl},
        {"method": "correction", **metrics_corr},
        {"method": "style_no_pin", **metrics_style},
        {"method": "style_pin", **metrics_pin},
    ]
    with open(OUT_DIR / f"{img_name}_results.json", "w") as f:
        json.dump({"image": img_name, "results": results, "pinning_log": pin_log},
                  f, indent=2, default=str)

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full",
                        choices=["clip", "inject", "full"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--n-images", type=int, default=None,
                        help="Number of coco_val images (overrides --image)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "clip":
        mode_clip()
        return

    # CLIP init
    print("=" * 60)
    print("DiT Phase 3: Style Disentanglement & Pinning")
    print("=" * 60)
    extractor, v_content = mode_clip()

    # Determine images
    if args.n_images is not None:
        coco_images = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco_images[:args.n_images]]
    elif args.image:
        images = [args.image]
    else:
        images = ["data/coco_val/coco_000000000285.jpg"]

    pipe = load_pipeline()

    print(f"\nProcessing {len(images)} image(s):")
    for p in images:
        print(f"  {Path(p).name}")

    if args.mode == "inject":
        for img_path in images:
            mode_inject(pipe, img_path, extractor, v_content)
    elif args.mode == "full":
        all_results = []
        for img_path in images:
            results = mode_full(pipe, img_path, extractor, v_content)
            all_results.append({"image": Path(img_path).stem, "results": results})

        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("Final Summary")
        print(f"{'='*60}")
        for entry in all_results:
            print(f"\n{entry['image']}:")
            print(f"{'Method':<30}{'PSNR':>8}{'LPIPS':>8}")
            print(f"{'-'*46}")
            for r in entry["results"]:
                m = r
                print(f"{r['method']:<30}{m['PSNR']:>8.2f}{m['LPIPS']:>8.4f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
