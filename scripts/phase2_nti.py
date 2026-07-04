"""
第二阶段第四步：Null-Text Inversion (NTI) 基线对比
NTI (Mokady et al., CVPR 2023): 优化每步空文本嵌入以精确反演

对比:
  1. DDIM (基线)
  2. DDIM + 残差校正 (我们的方法)
  3. NTI (优化基线)

NTI 需要 per-image caption 作为 CFG 条件 prompt。按照原论文，使用 BLIP
自动生成每张图片的描述。Caption 缓存到 outputs/phase2_nti/captions.json，
首次运行后无需重新生成。

用法:
  python scripts/phase2_nti.py --steps 20              # 论文标准 NTI
  python scripts/phase2_nti.py --steps 20 --legacy-nti  # 旧版参数复现
"""

import argparse, os, time, csv, json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import lpips
from PIL import Image

from phase2_common import (
    DEVICE, DTYPE, MODEL_ID,
    load_pipeline, load_image, decode_latent,
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    FeatureCorrector,
    compute_metrics, save_recon_img, save_results_csv, make_grid_image,
    DEFAULT_TEST_IMAGES, get_top_drift_layers,
)

OUT_DIR = Path("outputs/phase2_nti")
CAPTIONS_CACHE = OUT_DIR / "captions.json"

# Paper-standard NTI parameters (Mokady et al., CVPR 2023)
NTI_N_ITER = 10           # paper: 10 inner steps per timestep
NTI_GUIDANCE = 7.5        # paper: default CFG scale
NTI_INITIAL_LR = 0.01     # paper: 1e-2 * (1 - i/100) linear decay

# Legacy NTI parameters (old buggy configuration for reproducibility)
LEGACY_N_ITER = 8
LEGACY_GUIDANCE = 3.0
LEGACY_LR = 0.02

# Correction lambda: global value (from val-set tuning) instead of per-image
NTI_CORR_LAMBDA = 0.5


# ---------------------------------------------------------------------------
# BLIP caption generation (per-image, as in the original NTI paper)
# ---------------------------------------------------------------------------

_blip_model = None
_blip_processor = None

# Natural-language captions for CLIP zero-shot fallback.
# Structured as (prompt_template, objects) to generate diverse, natural captions.
# CLIP will pick the best-matching template + object combination.
COCO_OBJECTS = [
    "people walking on a street", "a person sitting at a table",
    "a cat sleeping on a couch", "a dog running in a park",
    "birds flying in the sky", "a horse standing in a field",
    "cars parked on a street", "a bicycle leaning against a wall",
    "a motorcycle on a road", "a bus driving down a street",
    "a truck on a highway", "a traffic light at an intersection",
    "a stop sign at a corner", "a bench in a park",
    "a sheep in a field", "a cow in a pasture",
    "food on a plate on a table", "a dining room with a table and chairs",
    "a kitchen with cabinets and appliances", "a living room with a sofa and TV",
    "a bedroom with a bed and pillows", "a bathroom with a sink and mirror",
    "a tall building in a city", "a house with a garden",
    "a city street with buildings", "a mountain landscape with trees",
    "snow-covered mountains under a blue sky", "a beach with waves and sand",
    "a river flowing through a forest", "a dense forest with tall trees",
    "colorful flowers in a garden", "fresh fruit on a wooden table",
    "vegetables displayed at a market", "a birthday cake with candles",
    "a pizza on a wooden board", "a sandwich on a plate",
    "people playing sports on a field", "a tennis player on a court",
    "a baseball player swinging a bat", "a boat sailing on water",
    "an airplane flying in the sky", "a train on railroad tracks",
    "a street with signs and cars", "a clock on a wall",
    "a vase with flowers on a table", "a wooden chair in a room",
    "a comfortable couch in a living room", "a bed with white sheets",
    "a flat screen television on a wall", "a laptop computer on a desk",
    "a person holding a cell phone", "books on a shelf",
    "a backpack on the floor", "an umbrella in the rain",
    "a skateboarder doing a trick", "a surfer riding a wave",
    "a tennis racket and balls", "a bottle of water on a table",
    "a wine glass on a table", "a coffee cup on a saucer",
    "silverware on a dining table", "a bowl of soup",
    "fresh bananas on a counter", "red apples in a basket",
    "oranges stacked in a pile", "broccoli on a cutting board",
    "carrots on a plate", "a hot dog with mustard",
    "a donut with sprinkles", "a refrigerator with magnets",
    "an oven in a kitchen", "a sink with dishes",
    "a teddy bear on a bed", "a toothbrush and toothpaste",
    "a person skiing on snow", "a snowboarder on a slope",
    "a giraffe in a zoo", "an elephant at a watering hole",
    "a zebra grazing on grass", "a lion resting in the shade",
    "a colorful parrot on a branch", "a fish swimming in an aquarium",
    "a baby in a high chair", "children playing in a playground",
    "a man in a suit and tie", "a woman in a dress",
    "a crowd at a concert", "a street market with stalls",
    "a bridge over water", "a lighthouse on a cliff",
    "a sunset over the ocean", "a rainbow after rain",
    "a snowy winter landscape", "autumn leaves on trees",
]


def _load_blip():
    """Lazy-load BLIP image captioning model (Salesforce/blip-image-captioning-base)."""
    global _blip_model, _blip_processor
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-base"
        _blip_processor = BlipProcessor.from_pretrained(model_id)
        _blip_model = BlipForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    return _blip_processor, _blip_model


def _generate_caption_blip(img_path: str) -> str:
    """Generate a caption using BLIP."""
    processor, model = _load_blip()
    img = Image.open(img_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)


def _generate_caption_clip(img_path: str) -> str:
    """Zero-shot caption via CLIP image-text matching (offline fallback).

    Uses the cached openai/clip-vit-large-patch14 to find the best-matching
    content description from a predefined set of COCO-relevant captions.
    """
    from phase3_common import CLIPFeatureExtractor
    extractor = CLIPFeatureExtractor()
    v_img = extractor.encode_image(img_path)  # [1, 768] normalized

    best_text, best_sim = "a photo", -1.0
    for text in COCO_CONTENT_DESCRIPTIONS:
        v_text = extractor.encode_text(text)
        sim = float((v_img * v_text).sum())
        if sim > best_sim:
            best_sim = sim
            best_text = text

    # Add a natural-language prefix for better CFG guidance
    return best_text


def generate_caption(img_path: str) -> str:
    """Generate a caption, preferring BLIP if available, falling back to CLIP."""
    try:
        return _generate_caption_blip(img_path)
    except (OSError, ImportError, RuntimeError) as e:
        print(f"  [BLIP unavailable ({type(e).__name__}), using CLIP fallback]")
        return _generate_caption_clip(img_path)


def load_or_generate_captions(image_paths: list) -> dict:
    """Load cached captions or generate with BLIP, keyed by image stem name.

    Preserves existing captions across runs — only generates for new images.
    """
    captions = {}
    if CAPTIONS_CACHE.exists():
        with open(CAPTIONS_CACHE) as f:
            captions = json.load(f)

    for p in image_paths:
        name = Path(p).stem
        if name in captions and captions[name]:
            continue  # already cached
        print(f"  [BLIP] Generating caption for {Path(p).name}...")
        captions[name] = generate_caption(p)

    # Save back (preserving old entries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_CACHE, "w") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f"  [Captions] {len(captions)} cached → {CAPTIONS_CACHE}")
    return captions


# ---------------------------------------------------------------------------
# DDIM inversion with trajectory (save intermediate latents for NTI)
# ---------------------------------------------------------------------------

def ddim_inversion_trajectory(pipe, latents, prompt_embeds, num_steps):
    """DDIM inversion, returning the full trajectory of latents at each timestep.

    Returns: (z_T, trajectory) where trajectory[t] = z_t (t from T down to 0).
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    trajectory = {0: z.clone()}  # z_0

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
            trajectory[int(t_next)] = z.clone()

    return z, trajectory


# ---------------------------------------------------------------------------
# NTI: optimize per-timestep null-text embeddings
# ---------------------------------------------------------------------------

def nti_optimize_and_reconstruct(pipe, z_T, trajectory, prompt_embeds, num_steps,
                                  n_iter=NTI_N_ITER, guidance_scale=NTI_GUIDANCE,
                                  initial_lr=NTI_INITIAL_LR, legacy=False):
    """Optimize per-timestep null-text embeddings and reconstruct.

    Paper-standard behavior (legacy=False):
    - All T timesteps optimized (no skip)
    - Null embedding carry-over from previous timestep
    - Decaying LR: lr_i = initial_lr * (1 - i / total_iters)

    Legacy mode (legacy=True): old buggy parameters for reproducibility.

    Returns: reconstructed latent (z_0).
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    total_iters = num_steps * n_iter

    z = z_T.clone()
    empty_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]  # [1, 77, 768]

    # Paper: initialize from previous timestep's optimized result
    prev_optimized = empty_embeds.clone()
    global_iter = [0]  # mutable counter for global LR decay

    for step_idx, t in enumerate(timesteps):
        t_int = int(t)

        if legacy and step_idx == 0:
            # Legacy: skip first timestep optimization
            null_emb = empty_embeds.clone()
        else:
            null_emb = _optimize_null_text(
                pipe, z, t, t_int, trajectory, prompt_embeds,
                prev_optimized, n_iter, guidance_scale, initial_lr,
                global_iter, total_iters, legacy
            )

        # Carry-over: save optimized embedding for next timestep
        prev_optimized = null_emb.detach().clone()

        # Take DDIM step with optimized null-text
        with torch.no_grad():
            eps_cond = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
            eps_uncond = pipe.unet(z, t, encoder_hidden_states=null_emb).sample
            noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                noise_pred = eps_cond
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


def _optimize_null_text(pipe, z_cur, t, t_int, trajectory, prompt_embeds,
                         init_null, n_iter, guidance_scale,
                         initial_lr=NTI_INITIAL_LR, global_iter=None, total_iters=None,
                         legacy=False):
    """Optimize a single null-text embedding (Adam, paper-standard or legacy)."""
    scheduler = pipe.scheduler

    timesteps = scheduler.timesteps.tolist()
    idx = timesteps.index(t)
    t_prev = 0 if idx == len(timesteps) - 1 else int(timesteps[idx + 1])
    z_target = trajectory[t_prev].detach()

    null_emb = init_null.clone().detach().requires_grad_(True)

    with torch.no_grad():
        eps_cond = pipe.unet(z_cur, t, encoder_hidden_states=prompt_embeds).sample

    if legacy:
        opt = torch.optim.Adam([null_emb], lr=0.02)
    else:
        opt = torch.optim.Adam([null_emb], lr=initial_lr)

    for _ in range(n_iter):
        opt.zero_grad()

        if not legacy and global_iter is not None and total_iters is not None:
            # Paper: lr_i = 1e-2 * (1 - i / total_iters)
            current_lr = initial_lr * (1.0 - global_iter[0] / total_iters)
            for pg in opt.param_groups:
                pg['lr'] = max(current_lr, 1e-6)

        eps_uncond = pipe.unet(z_cur, t, encoder_hidden_states=null_emb).sample
        noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        z_pred = scheduler.step(noise_pred, t, z_cur).prev_sample
        loss = torch.nn.functional.mse_loss(z_pred, z_target)
        if torch.isnan(loss):
            break
        loss.backward()
        if legacy:
            torch.nn.utils.clip_grad_norm_([null_emb], 1.0)
        opt.step()
        if not legacy and global_iter is not None:
            global_iter[0] += 1

    return null_emb.detach()


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_ddim_baseline(pipe, original_latent, original_tensor, prompt_embeds,
                       num_steps, lpips_fn=None, compute_dists=False):
    t0 = time.perf_counter()
    noise = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    elapsed = time.perf_counter() - t0
    recon = decode_latent(pipe, recon_latent)
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_dists=compute_dists)
    del noise, recon_latent
    torch.cuda.empty_cache()
    return m, recon, elapsed


def run_ddim_corr(pipe, original_latent, original_tensor, prompt_embeds,
                   num_steps, lam, layers, lpips_fn=None, compute_dists=False):
    t0 = time.perf_counter()
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, layers)
    corrector = FeatureCorrector(pipe.unet, layers, lam=lam)
    recon_latent = ddim_reconstruction_with_correction(
        pipe, noise, prompt_embeds, num_steps, saved, corrector)
    corrector.remove()
    elapsed = time.perf_counter() - t0
    recon = decode_latent(pipe, recon_latent)
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_dists=compute_dists)
    del noise, saved, recon_latent
    torch.cuda.empty_cache()
    return m, recon, elapsed


def run_nti(pipe, original_latent, original_tensor,
             nti_prompt_embeds, num_steps, lpips_fn=None, compute_dists=False,
             n_iter=NTI_N_ITER, guidance=NTI_GUIDANCE, initial_lr=NTI_INITIAL_LR,
             legacy=False):
    """Run NTI. Uses empty prompt for inversion, conditional prompt for CFG."""
    # Inversion with empty prompt (same as DDIM baseline)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    t0 = time.perf_counter()
    z_T, trajectory = ddim_inversion_trajectory(
        pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = nti_optimize_and_reconstruct(
        pipe, z_T, trajectory, nti_prompt_embeds, num_steps,
        n_iter=n_iter, guidance_scale=guidance, initial_lr=initial_lr, legacy=legacy)
    elapsed = time.perf_counter() - t0
    recon = decode_latent(pipe, recon_latent)
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_dists=compute_dists,
                         compute_arcface=False)
    del z_T, trajectory, recon_latent
    torch.cuda.empty_cache()
    return m, recon, elapsed


def run_nti_legacy(pipe, original_latent, original_tensor,
                    nti_prompt_embeds, num_steps, lpips_fn=None, compute_dists=False):
    """Legacy NTI for backward compatibility."""
    return run_nti(pipe, original_latent, original_tensor,
                   nti_prompt_embeds, num_steps, lpips_fn, compute_dists,
                   n_iter=LEGACY_N_ITER, guidance=LEGACY_GUIDANCE,
                   initial_lr=LEGACY_LR, legacy=True)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NTI 基线对比")
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--image", type=str, default=None, help="单图测试路径")
    parser.add_argument("--caption", type=str, default=None,
                        help="手动指定 caption（覆盖 BLIP 生成）")
    parser.add_argument("--no-blip", action="store_true",
                        help="禁用 BLIP，使用通用 prompt 'a photo'（旧行为）")
    parser.add_argument("--legacy-nti", action="store_true",
                        help="使用旧版NTI参数复现 (guidance=3.0, lr=0.02, 8iter, 跳过首步)")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dists", action="store_true", help="计算 DISTS 指标")
    args = parser.parse_args()

    steps_list = args.steps if isinstance(args.steps, list) else [args.steps]
    test_images = [args.image] if args.image else DEFAULT_TEST_IMAGES
    if isinstance(steps_list, int):
        steps_list = [steps_list]

    # NTI configuration
    if args.legacy_nti:
        n_iter, guidance, init_lr, legacy = LEGACY_N_ITER, LEGACY_GUIDANCE, LEGACY_LR, True
        mode_label = "LEGACY"
    else:
        n_iter, guidance, init_lr, legacy = NTI_N_ITER, NTI_GUIDANCE, NTI_INITIAL_LR, False
        mode_label = "PAPER"

    # Load global lambda from Phase 2 tuning if available
    corr_lambda = NTI_CORR_LAMBDA
    tuning_path = Path("outputs/phase2_full/selected_lambda.json")
    if tuning_path.exists():
        with open(tuning_path) as f:
            data = json.load(f)
        corr_lambda = data.get("selected_lambda", NTI_CORR_LAMBDA)
        print(f"[λ] Loaded from tuning: λ={corr_lambda}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-image captions via BLIP (paper-standard) or fallback
    if args.caption:
        captions = {Path(p).stem: args.caption for p in test_images}
        print(f"[Prompt] Manual: \"{args.caption}\"")
    elif args.no_blip:
        captions = {Path(p).stem: "a photo" for p in test_images}
        print("[Prompt] BLIP disabled, using \"a photo\" for all images")
    else:
        print("[0] Generating per-image BLIP captions (NTI paper-standard)...")
        captions = load_or_generate_captions(test_images)
        for name, cap in captions.items():
            print(f"  {name}: \"{cap}\"")

    print(f"[设备] {DEVICE}")
    print(f"[步数] {steps_list}")
    print(f"[NTI] {mode_label} (gs={guidance}, iters={n_iter}, lr={init_lr}, "
          f"carry_over={not legacy}, skip_first={legacy})")
    print(f"[Corr λ] {corr_lambda}")
    print(f"[DISTS] {'ON' if args.dists else 'OFF'}")

    print("[1] 加载 SD pipeline...")
    pipe = load_pipeline()

    lpips_fn = None
    if not args.skip_lpips:
        print("[2] 加载 LPIPS...")
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    layers = get_top_drift_layers(5)
    all_results = []

    total = len(test_images) * len(steps_list) * 3
    count = 0

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"[跳过] {img_path}")
            continue
        img_name = Path(img_path).stem
        is_face = "face" in img_name.lower()
        nti_caption = captions.get(img_name, "a photo")
        nti_prompt_embeds = pipe.encode_prompt(nti_caption, DEVICE, 1, False)[0]
        original_latent, original_tensor = load_image(pipe, img_path)

        for steps in steps_list:
            # --- DDIM baseline ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 DDIM...", end="", flush=True)
            m_ddim, r_ddim, t_ddim = run_ddim_baseline(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, lpips_fn, compute_dists=args.dists)
            all_results.append({"image": img_name, "steps": steps, "method": "DDIM",
                                "caption": "", **m_ddim, "time_s": t_ddim})
            save_recon_img(r_ddim, OUT_DIR, img_name, steps, "ddim")

            # --- DDIM+Corr ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 DDIM+Corr(λ={corr_lambda:.1f})...",
                  end="", flush=True)
            m_corr, r_corr, t_corr = run_ddim_corr(
                pipe, original_latent, original_tensor, prompt_embeds,
                steps, corr_lambda, layers, lpips_fn, compute_dists=args.dists)
            all_results.append({"image": img_name, "steps": steps,
                                "method": f"DDIM+Corr(λ={corr_lambda:.1f})",
                                "caption": "", **m_corr, "time_s": t_corr})
            save_recon_img(r_corr, OUT_DIR, img_name, steps, f"ddim_corr_lam{corr_lambda:.1f}")

            # --- NTI (with per-image BLIP caption) ---
            count += 1
            print(f"\r[{count}/{total}] {img_name} {steps}步 NTI \"{nti_caption[:40]}...\" ...",
                  end="", flush=True)
            m_nti, r_nti, t_nti = run_nti(
                pipe, original_latent, original_tensor,
                nti_prompt_embeds, steps, lpips_fn, compute_dists=args.dists,
                n_iter=n_iter, guidance=guidance, initial_lr=init_lr, legacy=legacy)
            all_results.append({"image": img_name, "steps": steps, "method": "NTI",
                                "caption": nti_caption, **m_nti, "time_s": t_nti})
            save_recon_img(r_nti, OUT_DIR, img_name, steps, "nti")

            # Generate comparison grid
            grid_dir = OUT_DIR / "grids"
            os.makedirs(grid_dir, exist_ok=True)
            images_dict = {
                "Original": (original_tensor + 1) / 2,
                "DDIM": (r_ddim + 1) / 2,
                "DDIM+Corr": (r_corr + 1) / 2,
                "NTI": (r_nti + 1) / 2,
            }
            metrics_dict = {"DDIM": m_ddim, "DDIM+Corr": m_corr, "NTI": m_nti}
            make_grid_image(images_dict, grid_dir / f"{img_name}_s{steps}_comparison.png",
                            ncols=4, reference_tensor=(original_tensor + 1) / 2,
                            metrics_dict=metrics_dict)
            print(f"\n  [Grid] {grid_dir / f'{img_name}_s{steps}_comparison.png'}")

            del r_ddim, r_corr, r_nti
            torch.cuda.empty_cache()

        del original_latent, original_tensor

    print()
    save_results_csv(all_results, OUT_DIR, "metrics.csv")

    # Summary
    print_summary(all_results)
    print(f"\n完成。输出: {OUT_DIR.resolve()}")


def print_summary(results):
    print(f"\n{'='*90}")
    print("三方对比总结 (50步)")
    if not any(r["steps"] == 50 for r in results):
        print("  (无 50 步数据)")
        return

    print(f"{'Image':15s} {'DDIM':>8s} {'DDIM+Corr':>10s} {'NTI':>8s}  "
          f"{'CorrΔ':>6s} {'NTIΔ':>6s}  {'NTI time':>8s}  Caption")
    print("-" * 90)

    images = sorted(set(r["image"] for r in results))
    for img in images:
        ddim_r = next((r for r in results if r["image"] == img and r["steps"] == 50
                        and r["method"] == "DDIM"), None)
        corr_r = next((r for r in results if r["image"] == img and r["steps"] == 50
                        and r["method"].startswith("DDIM+Corr")), None)
        nti_r = next((r for r in results if r["image"] == img and r["steps"] == 50
                       and r["method"] == "NTI"), None)
        if not all([ddim_r, corr_r, nti_r]):
            continue

        corr_delta = corr_r["PSNR"] - ddim_r["PSNR"]
        nti_delta = nti_r["PSNR"] - ddim_r["PSNR"]
        caption = nti_r.get("caption", "")[:50]
        print(f"{img:15s} {ddim_r['PSNR']:8.2f} {corr_r['PSNR']:10.2f} {nti_r['PSNR']:8.2f}  "
              f"{corr_delta:+6.2f} {nti_delta:+6.2f}  {nti_r['time_s']:7.1f}s  {caption}")

    ddim_avg = np.mean([r["PSNR"] for r in results
                        if r["steps"] == 50 and r["method"] == "DDIM"])
    corr_avg = np.mean([r["PSNR"] for r in results
                        if r["steps"] == 50 and r["method"].startswith("DDIM+Corr")])
    nti_avg = np.mean([r["PSNR"] for r in results
                       if r["steps"] == 50 and r["method"] == "NTI"])
    print("-" * 90)
    print(f"{'AVERAGE':15s} {ddim_avg:8.2f} {corr_avg:10.2f} {nti_avg:8.2f}  "
          f"{corr_avg-ddim_avg:+6.2f} {nti_avg-ddim_avg:+6.2f}")


if __name__ == "__main__":
    main()
