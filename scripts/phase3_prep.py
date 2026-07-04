"""
Phase 3 准备工作：CLIP 正交投影风格解耦原型

四个模式:
  --mode clip        正交投影数学验证
  --mode injection   纯风格 prompt 注入（无内容保持）
  --mode full        完整 pipeline：Phase 2 残差校正 + 风格 prompt
  --mode ablation    风格强度扫描

CLIP: openai/clip-vit-large-patch14 (ViT-L/14) — 真正的图像+文本多模态编码器。
StyleTex 公式: v_style = v_img - proj_{v_content}(v_img)

用法:
  python scripts/phase3_prep.py --mode clip
  python scripts/phase3_prep.py --mode injection --image data/basetest/face2.jpg
  python scripts/phase3_prep.py --mode full --image data/basetest/face2.jpg
"""

import argparse, json, os, csv, time, sys
from pathlib import Path

import torch, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

# ---- Import Phase 2 core functions from shared module ----
from phase2_common import (
    load_pipeline, load_image, decode_latent,
    FeatureCollector, FeatureCorrector, StyleFeatureInjector, AdaINStyleInjector, LambdaScheduler,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    ddim_inversion, ddim_reconstruction,
    compute_metrics, compute_arcface_similarity,
    make_grid_image, histogram_match, save_recon_img,
    DEVICE, DTYPE,
    get_top_drift_layers,
)

OUT_DIR = Path("outputs/phase3_prep")

TEST_IMAGES = [
    "data/basetest/face1.jpg", "data/basetest/face2.jpg",
    "data/basetest/nature.jpg", "data/content.jpg", "data/watercolor.jpeg",
]


# =========================================================================
# CLIP Feature Extraction (openai/clip-vit-large-patch14)
# =========================================================================

_clip_model_cache = {}


def _get_clip_model(device):
    """Lazy-load CLIP ViT-L/14, cached per device."""
    if device not in _clip_model_cache:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        _clip_model_cache[device] = (model, processor)
    return _clip_model_cache[device]


class CLIPFeatureExtractor:
    """CLIP ViT-L/14 多模态编码器：图像 → shared embedding space ← 文本。

    v_img  = CLIP.get_image_features(image)        [1, 768]  视觉嵌入
    v_text = CLIP.get_text_features(tokenize(text)) [1, 768]  文本嵌入

    两者在同一 768-dim 归一化 multimodal 空间中，可做正交投影。
    """

    def __init__(self, device=None):
        self._device = device or DEVICE
        self.model, self.processor = _get_clip_model(self._device)
        self.hidden_dim = self.model.config.projection_dim  # 768

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """CLIP 文本编码 → 768-dim 归一化 multimodal embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs).pooler_output  # [1, 768]
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """CLIP 视觉编码 → 768-dim 归一化 multimodal embedding.

        image: PIL.Image 或 str (file path)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs).pooler_output  # [1, 768]
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode_image_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """CLIP visual encoding from [-1,1] or [0,1] tensor → 768-dim embedding.

        tensor: [1, 3, H, W] in [-1,1] (VAE output) or [0,1]
        """
        # Convert to PIL for CLIP processor
        img_np = (tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2
        img_np = (img_np.clip(0, 1) * 255).astype('uint8')
        pil = Image.fromarray(img_np)
        return self.encode_image(pil)

    def compute_content_projection(self, v: torch.Tensor, v_content: torch.Tensor) -> float:
        """Compute scalar projection |proj_{v_content}(v)| as content preservation score.

        Higher = more content preserved. Changes indicate content drift.
        """
        vc_norm = v_content / (v_content.norm(dim=-1, keepdim=True) + 1e-8)
        return float((v * vc_norm).sum())

    def clip_direction_similarity(self, edited_image, target_image_or_vector):
        """CLIP 方向相似度: cos(CLIP(edited), CLIP(target))。

        衡量编辑后图像的风格方向是否与目标一致。
        若 target 为 PIL/tensor → 编码后计算 cos；若为 Tensor[1,768] → 直接计算。
        返回: (sim, v_edited, v_target)
        """
        v_edited = self.encode_image_from_tensor(edited_image) \
            if isinstance(edited_image, torch.Tensor) else self.encode_image(edited_image)

        if isinstance(target_image_or_vector, torch.Tensor) and target_image_or_vector.dim() == 2:
            v_target = target_image_or_vector  # 已经是 CLIP embedding
        elif isinstance(target_image_or_vector, torch.Tensor):
            v_target = self.encode_image_from_tensor(target_image_or_vector)
        else:
            v_target = self.encode_image(target_image_or_vector)

        sim = float((v_edited * v_target).sum())
        return sim, v_edited, v_target

    def compute_content_preservation(self, edited_tensor, original_v):
        """内容保持度: cos(CLIP(edited), v_original)。"""
        v_edited = self.encode_image_from_tensor(edited_tensor)
        sim = float((v_edited * original_v).sum())
        return sim

    def compute_orthogonal_decomposition(self, v_img, v_content):
        """StyleTex: v_style = v_img - proj_{v_content}(v_img).

        将 v_img 分解为沿内容方向的分量 (proj) 和与内容正交的风格分量 (v_style)。
        验证: cos(v_style, v_content) ≈ 0 → 风格与内容正交。
        """
        vc_norm = v_content.detach() / (v_content.detach().norm(dim=-1, keepdim=True) + 1e-8)
        proj = (v_img.detach() * vc_norm).sum(dim=-1, keepdim=True) * vc_norm
        v_style = v_img.detach() - proj
        cos = float((v_style * vc_norm).sum())
        return proj, v_style, cos


# =========================================================================
# Style candidate matching
# =========================================================================

STYLE_CANDIDATES = [
    ("an oil painting in impressionist style", "oil_painting"),
    ("a watercolor sketch with soft brushstrokes", "watercolor"),
    ("a pencil drawing with fine linework", "pencil_drawing"),
    ("a vibrant digital illustration", "digital_art"),
    ("a black and white photograph", "bw_photo"),
    ("a cartoon with bold outlines", "cartoon"),
    ("an anime-style illustration", "anime"),
    ("a vintage sepia-toned photograph", "vintage"),
    ("a neon-lit cyberpunk scene", "cyberpunk"),
    ("a minimalist flat design graphic", "minimalist"),
]


def find_closest_style(v_style, extractor):
    """Find closest style candidate to v_style (cosine similarity)."""
    best = ("", "", -1.0)
    for text, key in STYLE_CANDIDATES:
        emb = extractor.encode_text(text)
        sim = float((v_style * emb).sum())
        if sim > best[2]:
            best = (text, key, sim)
    return best


def make_styled_prompt_embedding(pipe, base_prompt, style_text, strength=0.5):
    """Linear interpolation of prompt embeddings: (1-λ)*base + λ*style."""
    base_emb = pipe.encode_prompt(base_prompt, DEVICE, 1, False)[0]
    style_emb = pipe.encode_prompt(style_text, DEVICE, 1, False)[0]
    return (1.0 - strength) * base_emb + strength * style_emb


def slerp(v1, v2, t):
    """Spherical linear interpolation between two normalized vectors.

    v1, v2: [1, D] normalized CLIP embeddings. t ∈ [0, 1].
    Returns: interpolated normalized vector [1, D].
    """
    v1_n = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2_n = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (v1_n * v2_n).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta) + 1e-8
    w1 = torch.sin((1 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    return w1 * v1_n + w2 * v2_n


def build_style_cross_attn_tokens(pipe, base_prompt, v_style, strength=0.5, mode="extra_token"):
    """Build cross-attention input with v_style directly injected.

    Args:
        pipe: SD pipeline
        base_prompt: text prompt for inversion (typically "")
        v_style: [1, 768] CLIP style vector (normalized)
        strength: style injection strength
        mode: "extra_token" — v_style as extra token (78 tokens)
              "interpolate"   — weighted mix of base + v_style (77 tokens)

    Returns: encoder_hidden_states [1, N, 768]
    """
    base_emb = pipe.encode_prompt(base_prompt, DEVICE, 1, False)[0]  # [1, 77, 768]
    v = v_style.to(dtype=base_emb.dtype, device=base_emb.device)

    if mode == "extra_token":
        style_token = strength * v.unsqueeze(1)  # [1, 1, 768]
        return torch.cat([base_emb, style_token], dim=1)

    elif mode == "interpolate":
        v_repeated = v.unsqueeze(1).repeat(1, base_emb.shape[1], 1)
        return (1.0 - strength) * base_emb + strength * v_repeated

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================================
# Pipeline
# =========================================================================

def run_baseline(pipe, original_latent, original_tensor, prompt_embeds, num_steps,
                 lpips_fn=None, compute_arcface=False):
    t0 = time.perf_counter()
    noise = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    recon = decode_latent(pipe, recon_latent)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)
    return m, recon, elapsed


def run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                         num_steps, lam, layers, lpips_fn=None, compute_arcface=False):
    """Phase 2 correction (exact same code as phase2_full)."""
    t0 = time.perf_counter()
    noise, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds,
                                                  num_steps, layers)
    sched = LambdaScheduler(lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, layers, sched)
    corrector.set_reference(saved, 0)
    recon_latent = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, num_steps, saved, corrector)
    corrector.remove()
    recon = decode_latent(pipe, recon_latent)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)
    return m, recon, elapsed


def run_correction_with_style(pipe, original_latent, original_tensor, prompt_embeds,
                                num_steps, corr_lam, corr_layers,
                                styled_prompt_embeds,
                                lpips_fn=None, compute_arcface=False,
                                style_injector=None):
    """Phase 2 correction + style injection (cross-attention and/or feature bias).

    Args:
        styled_prompt_embeds: cross-attention input (can be built by
            build_style_cross_attn_tokens for direct v_style injection)
        style_injector: optional StyleFeatureInjector for per-layer feature bias
    """
    t0 = time.perf_counter()
    noise, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds,
                                                  num_steps, corr_layers)
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    # Note: style_injector hooks are already registered in __init__

    scheduler = pipe.scheduler; scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        if t_int in saved:
            corrector.set_reference(saved[t_int], step_idx)
        else:
            corrector.set_reference({}, step_idx)
        inp = scheduler.scale_model_input(z, t)
        with torch.no_grad():
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=styled_prompt_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    corrector.remove()
    if style_injector is not None:
        style_injector.remove()

    recon = decode_latent(pipe, z)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)
    return m, recon, elapsed


def run_correction_with_style_and_pinning(
    pipe, original_latent, original_tensor, prompt_embeds,
    num_steps, corr_lam, corr_layers,
    styled_prompt_embeds,
    extractor, v_content,
    lpips_fn=None, compute_arcface=False,
    style_injector=None,
    pinning_freq=10, pinning_threshold=0.02, pinning_strength=0.5,
):
    """Phase 2 correction + style injection + orthogonal pinning constraint.

    At each ``pinning_freq`` denoising steps, decode the current latent,
    CLIP-encode it, and check the content projection |proj_{v_content}(v_current)|.
    If it deviates from the reference by more than ``pinning_threshold``,
    the effective style strength is scaled down for subsequent steps.

    This implements the "内容子空间正交钉扎机制" from the thesis proposal:
    style editing is constrained to stay orthogonal to the content manifold.

    Returns: (metrics, recon_tensor, elapsed, pinning_log)
      pinning_log: list of (step_idx, content_proj, deviation, effective_strength)
    """
    t0 = time.perf_counter()

    # Compute reference content projection from the original image
    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(original_tensor), v_content)
    print(f"  [Pin] Reference content projection: {ref_proj:.4f}")

    # DDIM inversion with feature collection
    noise, saved = ddim_inversion_with_features(pipe, original_latent, prompt_embeds,
                                                  num_steps, corr_layers)
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    # Reconstruct original style_strength for adaptation
    if "strength" in getattr(style_injector, "__dict__", {}):
        base_strength = style_injector.strength
    else:
        base_strength = 0.5

    scheduler = pipe.scheduler; scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    pinning_log = []
    effective_strength = base_strength

    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        if t_int in saved:
            corrector.set_reference(saved[t_int], step_idx)
        else:
            corrector.set_reference({}, step_idx)

        # Orthogonal pinning check (skip step 0 — too noisy to decode)
        if step_idx > 0 and pinning_freq > 0 and step_idx % pinning_freq == 0:
            with torch.no_grad():
                # Decode current latent to image
                current_img = decode_latent(pipe, z.clone())
                # CLIP encode and compute content projection
                v_current = extractor.encode_image_from_tensor(current_img)
                cur_proj = extractor.compute_content_projection(v_current, v_content)
                deviation = abs(cur_proj - ref_proj)
                pinning_log.append((step_idx, cur_proj, deviation, effective_strength))

                if deviation > pinning_threshold:
                    # Scale back style strength proportionally to deviation
                    scale = max(0.0, 1.0 - pinning_strength * (deviation / max(ref_proj, 0.01)))
                    effective_strength = base_strength * scale
                    if style_injector is not None:
                        style_injector.set_strength(effective_strength)
                    print(f"  [Pin] step={step_idx:3d}  proj={cur_proj:.4f}  "
                          f"dev={deviation:.4f}  →  style={effective_strength:.3f}")

        inp = scheduler.scale_model_input(z, t)
        with torch.no_grad():
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=styled_prompt_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    corrector.remove()
    if style_injector is not None:
        style_injector.remove()

    recon = decode_latent(pipe, z)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)

    # Print pinning summary
    if pinning_log:
        max_dev = max(d[2] for d in pinning_log)
        triggered = sum(1 for d in pinning_log if d[2] > pinning_threshold)
        print(f"  [Pin] checks={len(pinning_log)}  max_dev={max_dev:.4f}  "
              f"triggered={triggered}/{len(pinning_log)}")

    return m, recon, elapsed, pinning_log




def _save_csv(rows, filename):
    if not rows: return
    keys = sorted(set(k for r in rows for k in r))
    path = OUT_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"[CSV] {path}")


# =========================================================================
# MODES
# =========================================================================

def mode_clip():
    print("=" * 60)
    print("MODE: clip  —  CLIP 正交投影数学验证")
    print("=" * 60)
    extractor = CLIPFeatureExtractor()

    # ---- Part 1: text-only sanity check ----
    content_texts = ["", "a photo", "a photograph", "a realistic image"]
    style_texts = [
        "an oil painting", "a watercolor sketch", "a cartoon drawing",
        "an impressionist painting",
    ]

    print("\n--- Part 1: Text-only orthogonal decomposition ---")
    print(f"{'Content':<22s} {'Style':<30s} {'|v_style|':>8s} {'|proj|':>8s} {'cos':>10s}  Check")
    print("-" * 95)
    for ct in content_texts:
        v_content = extractor.encode_text(ct)
        for st in style_texts:
            v_img = extractor.encode_text(st)
            proj, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
            ok = abs(cos) < 1e-3
            print(f"{ct:<22s} {st:<30s} {v_style.norm().item():8.4f} {proj.norm().item():8.4f} {cos:10.2e}  {'PASS' if ok else 'FAIL'}")

    # Reconstruction check
    v_img = extractor.encode_text("an oil painting")
    v_content = extractor.encode_text("a photo")
    proj, v_style, _ = extractor.compute_orthogonal_decomposition(v_img, v_content)
    err = (v_img.detach() - (proj + v_style)).norm().item()
    print(f"\nReconstruction error |v_img - (proj+v_style)|: {err:.2e} (≈0 → decomposition exact)")

    # ---- Part 2: real image embeddings ----
    print("\n--- Part 2: Real image → orthogonal decomposition ---")
    import os as _os
    for img_path in TEST_IMAGES:
        if not _os.path.exists(img_path):
            print(f"  [SKIP] {img_path} not found")
            continue
        v_img = extractor.encode_image(img_path)            # ← CLIP vision encoder
        v_content = extractor.encode_text("a photo")        # ← CLIP text encoder
        proj, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
        name = Path(img_path).stem
        print(f"  {name:<18s} |v_style|={v_style.norm().item():.4f}  |proj|={proj.norm().item():.4f}  "
              f"cos(v_style,v_content)={cos:.2e}  {'PASS' if abs(cos) < 1e-3 else 'NON-ORTHOGONAL'}")

    print("\n→ orthogonal projection verified in CLIP multimodal space (image+text).")


def mode_injection(args):
    print("=" * 60)
    print("MODE: injection  —  风格 prompt 注入（无 Phase 2 校正）")
    print("=" * 60)
    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    img_path = args.image or TEST_IMAGES[1]
    img_name = Path(img_path).stem
    is_face = "face" in img_name.lower()
    steps = args.steps

    original_latent, original_tensor = load_image(pipe, img_path)

    # CLIP orthogonal projection with real image embedding
    v_content = extractor.encode_text("a photo")
    v_img = extractor.encode_image(img_path)
    _, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
    style_text, style_key, style_sim = find_closest_style(v_style, extractor)
    print(f"图片: {img_name}  cos(v_style,content)={cos:.2e}  风格匹配: {style_key}  sim={style_sim:.3f}")

    # Baseline
    print(f"\nBaseline (DDIM {steps}步)...")
    bm, brecon, bt = run_baseline(pipe, original_latent, original_tensor, prompt_embeds, steps, lpips_fn, is_face)
    print(f"  Baseline: PSNR={bm['PSNR']:.2f} SSIM={bm['SSIM']:.3f} LPIPS={bm.get('LPIPS',0):.3f} ArcFace={bm.get('ArcFace',0):.3f}")

    results = []
    images_grid = {"Original": original_tensor, "Baseline": brecon}

    for lam in args.strength:
        styled_embeds = make_styled_prompt_embedding(pipe, "", style_text, strength=lam)
        label = f"λ={lam:.1f}" if lam > 0.01 else "λ=0 (style off)"

        m, recon, elapsed = run_correction_with_style(
            pipe, original_latent, original_tensor, prompt_embeds,
            steps, corr_lam=0.0, corr_layers=[],
            styled_prompt_embeds=styled_embeds,
            lpips_fn=lpips_fn, compute_arcface=is_face)

        print(f"  {label}: PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f} LPIPS={m.get('LPIPS',0):.3f} ArcFace={m.get('ArcFace',0):.3f}")
        results.append({"image": img_name, "strength": lam, "style": style_key, **m, "time_s": elapsed})
        images_grid[label] = recon

    os.makedirs(OUT_DIR / "recons", exist_ok=True)
    make_grid_image(images_grid, OUT_DIR / "recons" / f"{img_name}_style_injection.png",
                    ncols=5, reference_tensor=original_tensor)
    _save_csv(results, "metrics_injection.csv")
    print(f"\n输出: {OUT_DIR}")


def mode_compare(args):
    """Ablation: style injection WITHOUT content protection vs our full framework."""
    print("=" * 60)
    print("MODE: compare  —  风格注入 baseline vs 我们的框架")
    print("=" * 60)
    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    corr_layers = get_top_drift_layers(5)
    corr_lam = 0.5
    steps = args.steps
    v_content = extractor.encode_text("a photo")

    all_results = []

    for img_path in (TEST_IMAGES if not args.image else [args.image]):
        if not os.path.exists(img_path): continue
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        original_latent, original_tensor = load_image(pipe, img_path)

        v_img = extractor.encode_image(img_path)
        _, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
        print(f"\n{'='*50}")
        print(f"{img_name}  face={is_face}  cos(v_style,content)={cos:.2e}")

        # 1) DDIM baseline
        bm, _, bt = run_baseline(pipe, original_latent, original_tensor, prompt_embeds, steps, lpips_fn, is_face)
        print(f"  baseline:         PSNR={bm['PSNR']:.2f} ArcFace={bm.get('ArcFace',0):.3f}")
        bm["lambda"] = "baseline"; bm["image"] = img_name; bm["time_s"] = bt
        all_results.append(bm)

        for lam in args.strength:
            styled_emb = build_style_cross_attn_tokens(pipe, "", v_style, strength=lam, mode="extra_token")

            # 2) Style injection WITHOUT correction (baseline for comparison)
            sm_nocorr, _, st_nocorr = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam=0.0, corr_layers=[],  # NO correction
                styled_prompt_embeds=styled_emb,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            print(f"  style_only λ={lam:.1f}:    PSNR={sm_nocorr['PSNR']:.2f} ArcFace={sm_nocorr.get('ArcFace',0):.3f}  Δ={sm_nocorr['PSNR']-bm['PSNR']:+.2f}")
            sm_nocorr["lambda"] = f"style_only_{lam:.1f}"; sm_nocorr["image"] = img_name; sm_nocorr["time_s"] = st_nocorr
            all_results.append(sm_nocorr)

            # 3) Correction only (no style)
            if lam == args.strength[0]:  # only once
                cm, _, ct = run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                                                  steps, corr_lam, corr_layers, lpips_fn, is_face)
                print(f"  corr_only:        PSNR={cm['PSNR']:.2f} ArcFace={cm.get('ArcFace',0):.3f}  Δ={cm['PSNR']-bm['PSNR']:+.2f}")
                cm["lambda"] = "corr_only"; cm["image"] = img_name; cm["time_s"] = ct
                all_results.append(cm)

            # 4) Our full framework: correction + style
            sm_full, _, st_full = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,  # WITH correction
                styled_prompt_embeds=styled_emb,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            print(f"  corr+style λ={lam:.1f}:   PSNR={sm_full['PSNR']:.2f} ArcFace={sm_full.get('ArcFace',0):.3f}  Δ={sm_full['PSNR']-bm['PSNR']:+.2f}")
            sm_full["lambda"] = f"corr+style_{lam:.1f}"; sm_full["image"] = img_name; sm_full["time_s"] = st_full
            all_results.append(sm_full)

            # 5) Full framework + pinning
            sm_pin, _, st_pin, pin_log = run_correction_with_style_and_pinning(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_emb,
                extractor=extractor, v_content=v_content,
                lpips_fn=lpips_fn, compute_arcface=is_face,
                pinning_freq=max(5, steps // 10),
                pinning_threshold=0.02, pinning_strength=0.5)
            pin_trig = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
            print(f"  corr+style+pin λ={lam:.1f}: PSNR={sm_pin['PSNR']:.2f} ArcFace={sm_pin.get('ArcFace',0):.3f}  Δ={sm_pin['PSNR']-bm['PSNR']:+.2f}  pin={pin_trig}/{len(pin_log)}")
            sm_pin["lambda"] = f"corr+style+pin_{lam:.1f}"; sm_pin["image"] = img_name; sm_pin["time_s"] = st_pin
            all_results.append(sm_pin)

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    out_dir = OUT_DIR / "compare"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        keys = sorted(set(k for r in all_results for k in r))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_results)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    _print_summary(all_results)
    print(f"\n输出: {out_dir}")


def mode_full(args):
    print("=" * 60)
    print("MODE: full  —  Phase 2 残差校正 + Phase 3 风格 prompt")
    print("=" * 60)
    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    corr_layers = get_top_drift_layers(5)
    corr_lam = 0.5
    steps = args.steps

    all_results = []

    for img_path in (TEST_IMAGES if not args.image else [args.image]):
        if not os.path.exists(img_path): continue
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        original_latent, original_tensor = load_image(pipe, img_path)

        v_content = extractor.encode_text("a photo")
        v_img = extractor.encode_image(img_path)
        _, v_style, cos = extractor.compute_orthogonal_decomposition(v_img, v_content)
        style_text, style_key, style_sim = find_closest_style(v_style, extractor)

        print(f"\n{'='*50}")
        print(f"{img_name}  face={is_face}  style={style_key} sim={style_sim:.3f}")

        # Baseline
        bm, _, bt = run_baseline(pipe, original_latent, original_tensor, prompt_embeds, steps, lpips_fn, is_face)
        print(f"  baseline:       PSNR={bm['PSNR']:.2f} LPIPS={bm.get('LPIPS',0):.3f} ArcFace={bm.get('ArcFace',0):.3f}")
        bm["lambda"] = "baseline"; bm["image"] = img_name; bm["time_s"] = bt
        all_results.append(bm)

        # Phase 2 correction only
        cm, _, ct = run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                                          steps, corr_lam, corr_layers, lpips_fn, is_face)
        print(f"  correction only: PSNR={cm['PSNR']:.2f} LPIPS={cm.get('LPIPS',0):.3f} ArcFace={cm.get('ArcFace',0):.3f}  Δ={cm['PSNR']-bm['PSNR']:+.2f}")
        cm["lambda"] = "corr_only"; cm["image"] = img_name; cm["time_s"] = ct
        all_results.append(cm)

        # Correction + Style: multiple injection methods
        for lam in args.strength:
            # --- Method 1: Old prompt text interpolation (baseline) ---
            styled_embeds_old = make_styled_prompt_embedding(pipe, "", style_text, strength=lam)
            sm_old, _, st_old = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds_old,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            print(f"  corr+prompt λ={lam:.1f}: PSNR={sm_old['PSNR']:.2f} LPIPS={sm_old.get('LPIPS',0):.3f} ArcFace={sm_old.get('ArcFace',0):.3f}  Δ={sm_old['PSNR']-bm['PSNR']:+.2f}")
            sm_old["lambda"] = f"corr+prompt_{lam:.1f}"; sm_old["image"] = img_name; sm_old["time_s"] = st_old
            all_results.append(sm_old)

            # --- Method 2: v_style as extra cross-attention token ---
            styled_embeds_token = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=lam, mode="extra_token")
            sm_token, _, st_token = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds_token,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            print(f"  corr+xattn_tok λ={lam:.1f}: PSNR={sm_token['PSNR']:.2f} LPIPS={sm_token.get('LPIPS',0):.3f} ArcFace={sm_token.get('ArcFace',0):.3f}  Δ={sm_token['PSNR']-bm['PSNR']:+.2f}")
            sm_token["lambda"] = f"corr+xattn_tok_{lam:.1f}"; sm_token["image"] = img_name; sm_token["time_s"] = st_token
            all_results.append(sm_token)

            # --- Method 3: v_style interpolated into all tokens ---
            styled_embeds_interp = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=lam, mode="interpolate")
            sm_interp, _, st_interp = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds_interp,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            print(f"  corr+xattn_int λ={lam:.1f}: PSNR={sm_interp['PSNR']:.2f} LPIPS={sm_interp.get('LPIPS',0):.3f} ArcFace={sm_interp.get('ArcFace',0):.3f}  Δ={sm_interp['PSNR']-bm['PSNR']:+.2f}")
            sm_interp["lambda"] = f"corr+xattn_int_{lam:.1f}"; sm_interp["image"] = img_name; sm_interp["time_s"] = st_interp
            all_results.append(sm_interp)

            # --- Method 4: feature bias (StyleFeatureInjector) ---
            style_inj = StyleFeatureInjector(pipe.unet, corr_layers, v_style, strength=lam)
            sm_bias, _, st_bias = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=prompt_embeds,  # use base prompt, bias handles style
                lpips_fn=lpips_fn, compute_arcface=is_face,
                style_injector=style_inj)
            style_inj.remove()
            print(f"  corr+feat_bias λ={lam:.1f}: PSNR={sm_bias['PSNR']:.2f} LPIPS={sm_bias.get('LPIPS',0):.3f} ArcFace={sm_bias.get('ArcFace',0):.3f}  Δ={sm_bias['PSNR']-bm['PSNR']:+.2f}")
            sm_bias["lambda"] = f"corr+feat_bias_{lam:.1f}"; sm_bias["image"] = img_name; sm_bias["time_s"] = st_bias
            all_results.append(sm_bias)

            # --- Method 5: cross-attn token + feature bias (both paths) ---
            style_inj2 = StyleFeatureInjector(pipe.unet, corr_layers, v_style, strength=lam)
            sm_both, _, st_both = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds_token,  # cross-attn tokens
                lpips_fn=lpips_fn, compute_arcface=is_face,
                style_injector=style_inj2)
            style_inj2.remove()
            print(f"  corr+both λ={lam:.1f}: PSNR={sm_both['PSNR']:.2f} LPIPS={sm_both.get('LPIPS',0):.3f} ArcFace={sm_both.get('ArcFace',0):.3f}  Δ={sm_both['PSNR']-bm['PSNR']:+.2f}")
            sm_both["lambda"] = f"corr+both_{lam:.1f}"; sm_both["image"] = img_name; sm_both["time_s"] = st_both
            all_results.append(sm_both)

            # --- Method 6: xattn_tok + orthogonal pinning constraint ---
            sm_pin, _, st_pin, pin_log = run_correction_with_style_and_pinning(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds_token,
                extractor=extractor, v_content=v_content,
                lpips_fn=lpips_fn, compute_arcface=is_face,
                pinning_freq=max(5, steps // 10), pinning_threshold=0.02,
                pinning_strength=0.5)
            print(f"  corr+xattn_tok+pin λ={lam:.1f}: PSNR={sm_pin['PSNR']:.2f} LPIPS={sm_pin.get('LPIPS',0):.3f} ArcFace={sm_pin.get('ArcFace',0):.3f}  Δ={sm_pin['PSNR']-bm['PSNR']:+.2f}")
            sm_pin["lambda"] = f"corr+xattn_tok+pin_{lam:.1f}"; sm_pin["image"] = img_name; sm_pin["time_s"] = st_pin
            # Log pinning stats if available
            if pin_log:
                max_dev = max(d[2] for d in pin_log)
                triggered = sum(1 for d in pin_log if d[2] > 0.02)
                sm_pin["pinning_max_dev"] = max_dev
                sm_pin["pinning_triggered"] = triggered
            all_results.append(sm_pin)

    os.makedirs(OUT_DIR, exist_ok=True)
    _save_csv(all_results, "metrics_full.csv")
    with open(OUT_DIR / "metrics_full.json", "w") as f:
        json.dump(all_results, f, indent=2)
    _print_summary(all_results)
    print(f"\n输出: {OUT_DIR}")


def mode_direction(args):
    """方向维度控制：两个风格参考图间的球面插值网格。"""
    import glob as globmod

    if not args.style_ref:
        print("[错误] 需指定 --style-ref（两个风格参考图路径，用逗号分隔）")
        return
    refs = [p.strip() for p in args.style_ref.split(",")]
    if len(refs) != 2:
        print(f"[错误] 需指定两个风格参考图，当前: {refs}")
        return
    if not all(os.path.exists(p) for p in refs):
        print(f"[错误] 参考图不存在: {refs}")
        return

    print("=" * 60)
    print("MODE: direction  —  风格方向球面插值")
    print(f"  风格 A: {refs[0]}")
    print(f"  风格 B: {refs[1]}")
    print("=" * 60)

    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    corr_layers = get_top_drift_layers(5)
    corr_lam = 0.5
    steps = args.steps

    # Encode both style references, decompose to pure style
    v_content = extractor.encode_text("a photo")
    v_a = extractor.encode_image(refs[0])
    _, v_style_a, _ = extractor.compute_orthogonal_decomposition(v_a, v_content)
    v_b = extractor.encode_image(refs[1])
    _, v_style_b, _ = extractor.compute_orthogonal_decomposition(v_b, v_content)
    print(f"|v_style_A|={v_style_a.norm().item():.3f}  |v_style_B|={v_style_b.norm().item():.3f}")

    # Content images
    img_path = args.image or TEST_IMAGES[0]
    if "*" in img_path: content_imgs = sorted(globmod.glob(img_path))[:3]  # limit to 3
    else: content_imgs = [img_path]
    content_imgs = [p for p in content_imgs if os.path.exists(p)]
    if not content_imgs: return

    T_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = []

    for img_path in content_imgs:
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        print(f"\n{'='*40}\n{img_name}")

        for t_val in T_values:
            # SLERP interpolate style vectors
            v_interp = slerp(v_style_a, v_style_b, t_val)
            # Build cross-attention embedding
            styled_emb = build_style_cross_attn_tokens(pipe, "", v_interp, strength=0.5, mode="extra_token")
            sm, srecon, st = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_emb,
                lpips_fn=lpips_fn, compute_arcface=("face" in img_name.lower()))
            print(f"  t={t_val:.2f}: PSNR={sm['PSNR']:.2f} LPIPS={sm.get('LPIPS',0):.3f}")
            sm["lambda"] = f"dir_t={t_val:.2f}"; sm["image"] = img_name; sm["time_s"] = st
            all_results.append(sm)
            save_recon_img(srecon, OUT_DIR / "direction", img_name, steps, f"dir_t={t_val:.2f}")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    out_dir = OUT_DIR / "direction"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        keys = sorted(set(k for r in all_results for k in r))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_results)
    with open(out_dir / "metrics.json", "w") as f: json.dump(all_results, f, indent=2)
    print(f"\n输出: {out_dir}")


def mode_ablation(args):
    print("=" * 60)
    print("MODE: ablation  —  风格强度扫描")
    print("=" * 60)
    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    corr_layers = get_top_drift_layers(5)
    corr_lam = 0.5
    steps = args.steps

    img_path = args.image or TEST_IMAGES[1]
    if not os.path.exists(img_path): return
    img_name = Path(img_path).stem
    is_face = "face" in img_name.lower()

    v_content = extractor.encode_text("a photo")
    v_img = extractor.encode_image(img_path)
    _, v_style, _ = extractor.compute_orthogonal_decomposition(v_img, v_content)
    style_text, style_key, style_sim = find_closest_style(v_style, extractor)
    print(f"风格匹配: {style_key} (sim={style_sim:.3f})")

    original_latent, original_tensor = load_image(pipe, img_path)

    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    results, psnrs, lpipss, arcfaces, xlabels = [], [], [], [], []

    bm, _, _ = run_baseline(pipe, original_latent, original_tensor, prompt_embeds, steps, lpips_fn, is_face)
    print(f"Baseline: PSNR={bm['PSNR']:.2f}\n")

    for lam in strengths:
        if lam == 0.0:
            m, _, t = run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                                            steps, corr_lam, corr_layers, lpips_fn, is_face)
            label = "corr_only"
        else:
            styled_embeds = make_styled_prompt_embedding(pipe, "", style_text, strength=lam)
            m, _, t = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_embeds,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            label = f"λ={lam:.1f}"

        print(f"  {label}: PSNR={m['PSNR']:.2f} LPIPS={m.get('LPIPS',0):.3f} ArcFace={m.get('ArcFace',0):.3f}")
        results.append({"image": img_name, "strength": lam, "config": label, **m, "time_s": t})
        psnrs.append(m['PSNR']); lpipss.append(m.get('LPIPS', 0))
        arcfaces.append(m.get('ArcFace', 0)); xlabels.append(label)

    _save_ablation_plot(xlabels, psnrs, lpipss, arcfaces, img_name)
    _save_csv(results, "metrics_ablation.csv")
    print(f"\n输出: {OUT_DIR}")


def _save_ablation_plot(labels, psnrs, lpipss, arcfaces, img_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    xs = range(len(labels))
    axes[0].bar(xs, psnrs, color="steelblue"); axes[0].set_title("PSNR (dB)")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].bar(xs, lpipss, color="darkorange"); axes[1].set_title("LPIPS")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].bar(xs, arcfaces, color="forestgreen"); axes[2].set_title("ArcFace")
    axes[2].axhline(y=0.7, color="red", linestyle="--", label="ArcFace≥0.7")
    axes[2].set_xticks(xs); axes[2].set_xticklabels(labels, rotation=45, ha="right"); axes[2].legend()
    plt.suptitle(f"Style Strength Ablation — {img_name}", fontsize=13)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUT_DIR / f"ablation_{img_name}.png", dpi=150); plt.close()


def _print_summary(all_results):
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'Image':<15s} {'Config':<22s} {'PSNR':>8s} {'LPIPS':>8s} {'ArcFace':>8s} {'ΔPSNR':>8s}")
    print("-" * 70)
    imgs = sorted(set(r["image"] for r in all_results))
    for img in imgs:
        img_results = [r for r in all_results if r["image"] == img]
        baseline_psnr = next(r["PSNR"] for r in img_results if r["lambda"] == "baseline")
        for r in img_results:
            delta = r["PSNR"] - baseline_psnr
            clip_s = f"CLIP_s={r.get('CLIP_style',0):.3f}" if 'CLIP_style' in r else ""
            print(f"{r['image']:<15s} {r['lambda']:<22s} {r['PSNR']:8.2f} {r.get('LPIPS',0):8.3f} {r.get('ArcFace',0):8.3f} {delta:+8.2f}  {clip_s}")


# =========================================================================
# Main
# =========================================================================

def mode_style_transfer(args):
    """风格参考图 → 内容图迁移。使用独立的风格图片作为 v_style 来源。"""
    import glob as globmod

    if not args.style_ref or not os.path.exists(args.style_ref):
        print(f"[错误] 风格参考图不存在: {args.style_ref}")
        return
    if not args.image:
        print("[错误] 需指定 --image（内容图路径或 glob）")
        return

    print("=" * 60)
    print("MODE: style_transfer  —  风格参考图注入")
    print(f"  风格: {args.style_ref}")
    print(f"  内容: {args.image}")
    print("=" * 60)

    import lpips
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    corr_layers = get_top_drift_layers(5)
    corr_lam = 0.5
    steps = args.steps

    # --- Load style reference image as tensor for AdaIN ---
    style_tensor = load_image(pipe, args.style_ref)[1]
    print(f"  style_tensor shape: {style_tensor.shape}")

    # --- Encode style reference with orthogonal decomposition ---
    print(f"\n[风格参考] {args.style_ref}")
    v_style_img = extractor.encode_image(args.style_ref)
    # Use the same content direction as mode_full for consistency
    v_content_dir = extractor.encode_text("a photo")
    proj_ref, v_style_pure, cos_ref = extractor.compute_orthogonal_decomposition(
        v_style_img, v_content_dir)
    print(f"  |v_style_img|={v_style_img.norm().item():.3f}  "
          f"|v_style_pure|={v_style_pure.norm().item():.3f}  "
          f"cos(v_style_pure, v_content)={cos_ref:.2e}")
    # Use the pure style component (orthogonal to content) for injection
    v_style_ref = v_style_pure

    # --- Find content images ---
    if "*" in args.image or "?" in args.image:
        content_images = sorted(globmod.glob(args.image))
    elif os.path.isdir(args.image):
        content_images = sorted(globmod.glob(os.path.join(args.image, "*.jpg"))
                                + globmod.glob(os.path.join(args.image, "*.jpeg"))
                                + globmod.glob(os.path.join(args.image, "*.png")))
    else:
        content_images = [args.image]

    content_images = [p for p in content_images if os.path.exists(p)]
    if not content_images:
        print(f"[错误] 未找到匹配的内容图: {args.image}")
        return
    print(f"[内容图] {len(content_images)} 张")

    all_results = []

    for img_path in content_images:
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        original_latent, original_tensor = load_image(pipe, img_path)

        print(f"\n{'='*50}")
        print(f"{img_name}")

        # Pre-encode content for CLIP preservation metric
        v_content_orig = extractor.encode_image_from_tensor(original_tensor)

        # Helper: compute CLIP metrics, save recon, add to results
        out_dir = OUT_DIR / "style_transfer" / "recons" / img_name
        os.makedirs(out_dir, exist_ok=True)

        def _add_clip_metrics(r, recon, label, elapsed):
            clip_style, _, _ = extractor.clip_direction_similarity(recon, v_style_ref)
            clip_content = extractor.compute_content_preservation(recon, v_content_orig)
            r["CLIP_style"] = clip_style
            r["CLIP_content"] = clip_content
            r["lambda"] = label; r["image"] = img_name; r["time_s"] = elapsed
            all_results.append(r)
            save_recon_img(recon, OUT_DIR / "style_transfer", img_name, steps, label.replace(".", "_"))
            return r

        # Baseline
        bm, brecon, bt = run_baseline(pipe, original_latent, original_tensor, prompt_embeds,
                                        steps, lpips_fn, is_face)
        _add_clip_metrics(bm, brecon, "baseline", bt)
        base_clip_style = bm["CLIP_style"]
        print(f"  baseline:       PSNR={bm['PSNR']:.2f} CLIP_style={bm['CLIP_style']:.3f} CLIP_content={bm['CLIP_content']:.3f}")

        # Correction only
        cm, crecon, ct = run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                                               steps, corr_lam, corr_layers, lpips_fn, is_face)
        _add_clip_metrics(cm, crecon, "corr_only", ct)
        print(f"  corr_only:      PSNR={cm['PSNR']:.2f} CLIP_style={cm['CLIP_style']:.3f} CLIP_content={cm['CLIP_content']:.3f}  ΔPSNR={cm['PSNR']-bm['PSNR']:+.2f}")

        for lam in args.strength:
            # --- xattn_tok with style reference ---
            styled_emb = build_style_cross_attn_tokens(
                pipe, "", v_style_ref, strength=lam, mode="extra_token")
            sm, srecon, st = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_emb,
                lpips_fn=lpips_fn, compute_arcface=is_face)
            _add_clip_metrics(sm, srecon, f"xattn_tok_{lam:.1f}", st)
            print(f"  xattn_tok λ={lam:.1f}: PSNR={sm['PSNR']:.2f} CLIP_style={sm['CLIP_style']:.3f} CLIP_content={sm['CLIP_content']:.3f}  ΔPSNR={sm['PSNR']-bm['PSNR']:+.2f}")

            # --- feat_bias with style reference ---
            style_inj = StyleFeatureInjector(pipe.unet, corr_layers, v_style_ref, strength=lam)
            sm_b, srecon_b, st_b = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=prompt_embeds,
                lpips_fn=lpips_fn, compute_arcface=is_face,
                style_injector=style_inj)
            style_inj.remove()
            _add_clip_metrics(sm_b, srecon_b, f"feat_bias_{lam:.1f}", st_b)
            print(f"  feat_bias λ={lam:.1f}: PSNR={sm_b['PSNR']:.2f} CLIP_style={sm_b['CLIP_style']:.3f} CLIP_content={sm_b['CLIP_content']:.3f}  ΔPSNR={sm_b['PSNR']-bm['PSNR']:+.2f}")

            # --- AdaIN: feature-space style statistics matching ---
            style_inj_adain = AdaINStyleInjector(pipe, corr_layers, style_tensor, strength=lam)
            sm_a, srecon_a, st_a = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=prompt_embeds,  # no cross-attn, AdaIN handles style
                lpips_fn=lpips_fn, compute_arcface=is_face,
                style_injector=style_inj_adain)
            style_inj_adain.remove()
            _add_clip_metrics(sm_a, srecon_a, f"adain_{lam:.1f}", st_a)
            print(f"  adain λ={lam:.1f}: PSNR={sm_a['PSNR']:.2f} CLIP_style={sm_a['CLIP_style']:.3f} CLIP_content={sm_a['CLIP_content']:.3f}  ΔPSNR={sm_a['PSNR']-bm['PSNR']:+.2f}")

            # --- xattn_tok + pinning ---
            sm_p, srecon_p, st_p, pin_log = run_correction_with_style_and_pinning(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lam, corr_layers,
                styled_prompt_embeds=styled_emb,
                extractor=extractor, v_content=extractor.encode_text("a photo"),
                lpips_fn=lpips_fn, compute_arcface=is_face,
                pinning_freq=max(5, steps // 10),
                pinning_threshold=0.02, pinning_strength=0.5)
            _add_clip_metrics(sm_p, srecon_p, f"xattn_tok+pin_{lam:.1f}", st_p)
            print(f"  xattn+pin λ={lam:.1f}: PSNR={sm_p['PSNR']:.2f} CLIP_style={sm_p['CLIP_style']:.3f} CLIP_content={sm_p['CLIP_content']:.3f}  ΔPSNR={sm_p['PSNR']-bm['PSNR']:+.2f}")

        del original_latent, original_tensor
        torch.cuda.empty_cache()

    # --- Save ---
    out_dir = OUT_DIR / "style_transfer"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        keys = sorted(set(k for r in all_results for k in r))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_results)
    print(f"\n[CSV] {csv_path}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[JSON] {out_dir / 'metrics.json'}")
    _print_summary(all_results)
    print(f"\n输出: {out_dir}")


# =========================================================================
# DCSC mode
# =========================================================================

def mode_dcsc(args):
    """DCSC: Drift-Aware Closed-Loop Style Controller.

    Runs Pareto scan over (Kp, lambda_0) and compares with Phase 3 pinning.
    """
    from dcsc_experiment import (
        pareto_scan, compare_across_methods, compute_pareto_frontier,
        plot_pareto_frontier, generate_comparison_table,
    )
    import lpips

    print("=" * 60)
    print("DCSC: Drift-Aware Closed-Loop Style Controller")
    print("=" * 60)

    # Determine images
    if args.image:
        import glob as _glob
        images = sorted(_glob.glob(args.image))
        if not images:
            print(f"[ERROR] No images found: {args.image}"); return
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:5]]
    print(f"Images: {len(images)}")

    # Load
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE) if not args.skip_lpips else None
    corr_layers = get_top_drift_layers(5)

    # Get Kp and lambda_0 values from CLI or defaults
    Kp_values = list(getattr(args, "Kp", [0.5, 1.0, 2.0, 5.0]))
    lambda_0_values = list(getattr(args, "lambda_0", [0.3, 0.5, 0.7]))
    control_freq = getattr(args, "control_freq", 5)
    dcsc_Kp_opt = getattr(args, "dcsc_Kp_opt", 1.0)
    dcsc_lam_opt = getattr(args, "dcsc_lam_opt", 0.5)

    print(f"Kp values: {Kp_values}")
    print(f"lambda_0 values: {lambda_0_values}")
    print(f"control_freq: {control_freq}")

    out_dir = OUT_DIR / "dcsc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pareto scan ----
    print(f"\n[1] Pareto scan...")
    results = pareto_scan(
        pipe, images, extractor,
        num_steps=args.steps, corr_lam=0.5, corr_layers=corr_layers,
        Kp_values=Kp_values, lambda_0_values=lambda_0_values,
        control_freq=control_freq, lpips_fn=lpips_fn,
    )

    # Find Pareto-optimal points
    pareto = compute_pareto_frontier(results)
    print(f"\n[Pareto] {len(pareto)} non-dominated points:")
    for p in pareto:
        print(f"  Kp={p['Kp']:.1f} λ_0={p['lambda_0']:.2f}  "
              f"PSNR={p['PSNR']:.2f}  CLIP_s={p['CLIP_style']:.3f}  "
              f"CLIP_c={p['CLIP_content']:.3f}")

    # Plot Pareto
    plot_pareto_frontier(
        {"DCSC": results},
        str(out_dir / "pareto_frontier.png"),
        title=f"DCSC Pareto Frontier ({len(images)} images)")

    # ---- Cross-method comparison ----
    print(f"\n[2] Cross-method comparison at optimal DCSC params...")
    compare_results = compare_across_methods(
        pipe, images, extractor,
        num_steps=args.steps, corr_lam=0.5, corr_layers=corr_layers,
        lpips_fn=lpips_fn,
        dcsc_Kp=dcsc_Kp_opt, dcsc_lambda_0=dcsc_lam_opt,
    )

    # Plot comparison
    plot_pareto_frontier(
        compare_results,
        str(out_dir / "comparison_pareto.png"),
        title=f"Style-Content Pareto Frontier ({len(images)} images)")
    generate_comparison_table(
        compare_results, str(out_dir / "comparison_table.tex"))

    # Print summary
    print("\n" + "=" * 60)
    print("DCSC Summary")
    print("=" * 60)
    for method in ["DDIM", "Correction", "StyleOnly", "Phase3_Pinning", "DCSC"]:
        entries = compare_results.get(method, [])
        if not entries:
            continue
        psnr = np.mean([e["PSNR"] for e in entries])
        lpips_v = np.mean([e.get("LPIPS", 0) for e in entries])
        cs = np.mean([e.get("CLIP_style", 0) for e in entries])
        cc = np.mean([e.get("CLIP_content", 0) for e in entries])
        print(f"  {method:20s}: PSNR={psnr:.2f} LPIPS={lpips_v:.3f}  "
              f"CLIP_s={cs:.3f} CLIP_c={cc:.3f}")

    print(f"\nOutput: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Prep: Style Decoupling Prototype")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["clip", "injection", "full", "ablation", "style_transfer", "compare", "direction", "dcsc"])
    parser.add_argument("--image", type=str, default=None,
                        help="单张内容图路径或 glob 模式（如 data/coco_val/*.jpg）")
    parser.add_argument("--style-ref", type=str, default=None,
                        help="风格参考图路径（如 data/watercolor.jpeg）")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--Kp", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
                        help="DCSC: proportional gain values for Pareto scan")
    parser.add_argument("--lambda-0", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="DCSC: base style strength values")
    parser.add_argument("--control-freq", type=int, default=5,
                        help="DCSC: control frequency (steps between updates)")
    parser.add_argument("--dcsc-Kp-opt", type=float, default=1.0,
                        help="DCSC: optimal Kp for cross-method comparison")
    parser.add_argument("--dcsc-lam-opt", type=float, default=0.5,
                        help="DCSC: optimal lambda_0 for cross-method comparison")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    if args.mode == "clip": mode_clip()
    elif args.mode == "injection": mode_injection(args)
    elif args.mode == "full": mode_full(args)
    elif args.mode == "ablation": mode_ablation(args)
    elif args.mode == "style_transfer": mode_style_transfer(args)
    elif args.mode == "compare": mode_compare(args)
    elif args.mode == "direction": mode_direction(args)
    elif args.mode == "dcsc": mode_dcsc(args)


if __name__ == "__main__":
    main()
