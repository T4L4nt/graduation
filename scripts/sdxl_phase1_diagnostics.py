"""
SDXL Phase 1: DDIM 反演-重建漂移动态诊断

复现 SD 1.5 Phase 1 诊断流程，在 SDXL 上分析：
- 单图 DDIM 反演 → 重建 PSNR/SSIM/LPIPS
- UNet 各层特征漂移（inv 与 recon 特征 L2 距离）
- 跨步数（4/10/20/50）的漂移动态

与 SD 1.5 输出的对比用于论文 4.X 节 SDXL 泛化章。
"""
import argparse, json, csv
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
import lpips
from skimage.metrics import structural_similarity as ssim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

OUT_DIR = Path("outputs/sdxl_phase1")
TEST_IMAGES = [
    "data/basetest/face1.jpg",
    "data/basetest/face2.jpg",
    "data/basetest/nature.jpg",
    "data/basetest/architecture.jpg",
    "data/basetest/still_life.jpg",
]
STEP_LIST = [4, 10, 20, 50]


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, local_files_only=True,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # SDXL VAE produces NaN in fp16; keep VAE in fp32
    pipe.vae.to(torch.float32)
    return pipe


# ═══════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════

def load_and_encode(pipe, path, size=1024):
    """Load image → resize → [−1,1] tensor + VAE latent (VAE runs fp32)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    # VAE is fp32; encode in fp32 then convert to fp16 for UNet
    tensor_fp32 = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    tensor_fp32 = 2 * tensor_fp32 - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor_fp32).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent.to(dtype=DTYPE), tensor_fp32, img


def decode_latent(pipe, latent):
    """Decode latent with VAE (fp32). Returns fp32 tensor."""
    with torch.no_grad():
        tensor = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
    return tensor


# ═══════════════════════════════════════════════════════════════
# SDXL text encoding
# ═══════════════════════════════════════════════════════════════

def encode_prompt_sdxl(pipe, prompt, device=DEVICE):
    """SDXL dual-encoder: returns (prompt_embeds, pooled_prompt_embeds, time_ids)."""
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    # time_ids encodes: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=DTYPE)
    return prompt_embeds, pooled_prompt_embeds, time_ids


# ═══════════════════════════════════════════════════════════════
# DDIM inversion / reconstruction (SDXL)
# ═══════════════════════════════════════════════════════════════

def _get_added_cond_kwargs(pooled_prompt_embeds, time_ids):
    """Pack SDXL conditioning kwargs for UNet."""
    return {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}


def ddim_inversion_sdxl(pipe, latents, prompt_embeds,
                        pooled_prompt_embeds, time_ids,
                        num_steps, guidance_scale=1.0):
    """DDIM inversion without feature collection (features collected separately)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    added_cond = _get_added_cond_kwargs(pooled_prompt_embeds, time_ids)
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
    return z


def ddim_inversion_final_features(pipe, latents, prompt_embeds,
                                  pooled_prompt_embeds, time_ids,
                                  num_steps, hooker):
    """Run DDIM inversion and collect features from the FINAL step only (t≈0).
    The last UNet forward during inversion (closest to clean latent) is the
    most informative for drift analysis — it's where the VAE decoding is
    most sensitive to feature perturbations.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    added_cond = _get_added_cond_kwargs(pooled_prompt_embeds, time_ids)
    extended_ts = timesteps.tolist() + [0]

    last_features = {}
    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
        # Collect features after the last inversion UNet forward
        last_features = {
            k: v.detach().cpu().clone() for k, v in hooker.features.items()
        }
    return z, last_features


def ddim_reconstruction_final_features(pipe, noise, prompt_embeds,
                                       pooled_prompt_embeds, time_ids,
                                       num_steps, hooker):
    """Reconstruct and collect features from the FINAL step only.
    Compare final-step features from inversion vs reconstruction — this is
    the key comparison: do the same UNet layers produce the same features
    when running in opposite directions?
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    added_cond = _get_added_cond_kwargs(pooled_prompt_embeds, time_ids)

    last_features = {}
    with torch.no_grad():
        for t in timesteps:
            noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds,
                                   added_cond_kwargs=added_cond).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
            # Collect features at the LAST step (closest to clean image)
            last_features = {
                k: v.detach().cpu().clone() for k, v in hooker.features.items()
            }
    return z, last_features


# ═══════════════════════════════════════════════════════════════
# Feature hooking
# ═══════════════════════════════════════════════════════════════

def discover_hook_targets(unet):
    """Dynamically discover ResNet blocks and Attention transformer blocks."""
    targets = []
    for name, _ in unet.named_modules():
        parts = name.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(name)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(name)
    return sorted(targets)


class UNetFeatureHooker:
    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.handles = []

    def _find_module(self, name):
        tokens = name.split(".")
        mod = self.unet
        for t in tokens:
            try:
                mod = getattr(mod, t)
            except AttributeError:
                return None
        return mod

    def register(self):
        for name in discover_hook_targets(self.unet):
            mod = self._find_module(name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self.features.__setitem__(n, out)
                )
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()




# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(original_tensor, recon_tensor, lpips_fn):
    orig = original_tensor.detach()
    recon = recon_tensor.detach()
    orig_01 = (orig + 1) / 2
    recon_01 = (recon + 1) / 2

    mse_val = torch.nn.functional.mse_loss(orig_01, recon_01).item()
    psnr_val = 20 * np.log10(1.0) - 10 * np.log10(mse_val) if mse_val > 0 else 100.0

    # SSIM uses CPU numpy
    orig_np = orig_01.cpu().squeeze(0).permute(1, 2, 0).numpy()
    recon_np = recon_01.cpu().squeeze(0).permute(1, 2, 0).numpy()
    ssim_val = ssim(orig_np, recon_np, channel_axis=2, data_range=1.0)

    # LPIPS needs tensors on the same device as the LPIPS model
    lpips_device = next(lpips_fn.parameters()).device
    lpips_val = lpips_fn(orig_01.to(lpips_device), recon_01.to(lpips_device)).item()

    return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def make_heatmap(drift_matrix, layer_names, img_names, out_path):
    """Heatmap: rows=layers, cols=images, values=drift."""
    fig, ax = plt.subplots(figsize=(max(8, len(img_names)*1.5),
                                    max(6, len(layer_names)*0.35)))
    im = ax.imshow(drift_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(img_names)))
    ax.set_xticklabels(img_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_title("SDXL UNet Layer Drift (MSE inv vs recon)", fontsize=12)
    plt.colorbar(im, ax=ax, label="MSE drift")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_bar_chart(layer_names, mean_drifts, out_path, top_n=30):
    """Top-N mean drift bar chart."""
    idx = np.argsort(mean_drifts)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), [mean_drifts[i] for i in idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([layer_names[i] for i in idx], fontsize=7)
    ax.set_xlabel("Mean MSE drift")
    ax.set_title(f"SDXL: Top-{top_n} Layer Drift (averaged over images)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_step_decay(layer_data, step_list, out_path):
    """Drift-vs-steps line chart for top drift layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(layer_data)))
    for (layer_name, drift_by_step), c in zip(layer_data.items(), colors):
        ax.plot(step_list, drift_by_step, "o-", label=layer_name, color=c, ms=4)
    ax.set_xlabel("DDIM steps")
    ax.set_ylabel("MSE drift")
    ax.set_title("SDXL: Drift vs DDIM Steps (top layers)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-images", type=int, default=None,
                        help="Limit to first N images (default: all)")
    parser.add_argument("--step-scan", action="store_true",
                        help="Scan across step counts (4,10,20,50)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SDXL Phase 1: DDIM Inversion-Reconstruction Drift Diagnostics")
    print("=" * 60)

    # Load pipeline
    print("\n[1/4] Loading SDXL pipeline with DDIM scheduler...")
    pipe = load_pipeline()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    images = TEST_IMAGES[:args.n_images] if args.n_images else TEST_IMAGES
    print(f"\n[2/4] Test images ({len(images)}):")
    for p in images:
        print(f"  {p}")

    # Hook setup
    print("\n[3/4] Setting up UNet hooks...")
    hooker = UNetFeatureHooker(pipe.unet)
    all_layers = discover_hook_targets(pipe.unet)
    print(f"  Discovered {len(all_layers)} hook targets")

    if args.step_scan:
        step_list = STEP_LIST
    else:
        step_list = [args.steps]

    all_results = []
    # drift_matrix: [n_layers × n_images] for the primary step count
    drift_by_layer_img = defaultdict(dict)
    # drift_by_step: {layer: [mean_drift_at_step_4, ...]}
    drift_by_step = defaultdict(list)

    for step_idx, num_steps in enumerate(step_list):
        print(f"\n{'='*60}")
        print(f"  Steps = {num_steps}")
        print(f"{'='*60}")

        for img_idx, img_path in enumerate(images):
            img_name = Path(img_path).stem
            print(f"\n  [{img_idx+1}/{len(images)}] {img_name}")

            # Load & encode (VAE is fp32)
            latents, tensor, _ = load_and_encode(pipe, img_path)
            prompt_embeds, pooled_embeds, time_ids = encode_prompt_sdxl(pipe, "")

            # ---- Inversion with final-step features ----
            inv_hooker = UNetFeatureHooker(pipe.unet)
            inv_hooker.register()
            noise, inv_features = ddim_inversion_final_features(
                pipe, latents, prompt_embeds, pooled_embeds, time_ids,
                num_steps, inv_hooker,
            )
            inv_hooker.remove()

            # ---- Reconstruction with final-step features ----
            recon_hooker = UNetFeatureHooker(pipe.unet)
            recon_hooker.register()
            recon_latents, recon_features = ddim_reconstruction_final_features(
                pipe, noise, prompt_embeds, pooled_embeds, time_ids,
                num_steps, recon_hooker,
            )
            recon_hooker.remove()

            # ---- Image metrics ----
            recon_tensor = decode_latent(pipe, recon_latents)
            metrics = compute_metrics(tensor, recon_tensor, lpips_fn)
            print(f"    PSNR={metrics['PSNR']:.2f}  SSIM={metrics['SSIM']:.4f}  LPIPS={metrics['LPIPS']:.4f}")

            # ---- Per-layer drift (final-step comparison) ----
            avg_drifts = {}
            for layer in all_layers:
                if layer in inv_features and layer in recon_features:
                    drift = torch.nn.functional.mse_loss(
                        inv_features[layer].float(), recon_features[layer].float()
                    ).item()
                else:
                    drift = 0.0
                avg_drifts[layer] = drift
                drift_by_layer_img[layer][img_name] = drift

            # Log
            for layer, d in sorted(avg_drifts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {layer:50s}  drift={d:.6e}")

            all_results.append({
                "image": img_name,
                "steps": num_steps,
                "metrics": metrics,
                "drifts": avg_drifts,
            })

    # ── Summarize ──
    print(f"\n{'='*60}")
    print("Summary: Top-10 drift layers (averaged over images)")

    layer_mean = {}
    for layer in all_layers:
        vals = [drift_by_layer_img[layer][img] for img in [Path(p).stem for p in images]
                if img in drift_by_layer_img[layer]]
        if vals:
            layer_mean[layer] = np.mean(vals)

    ranked = sorted(layer_mean.items(), key=lambda x: -x[1])
    top10 = [l for l, _ in ranked[:10]]

    print(f"\n{'Rank':<6}{'Layer':<55}{'Mean Drift':<16}{'Category'}")
    print("-" * 85)
    for rank, (layer, drift) in enumerate(ranked, 1):
        if "resnets" in layer:
            cat = "ResNet"
        elif "attentions" in layer or "transformer_blocks" in layer:
            cat = "Attention"
        else:
            cat = "Other"
        marker = " ←" if rank <= 10 else ""
        print(f"{rank:<6}{layer:<55}{drift:<16.6e}{cat}{marker}")

    # ── Save JSON ──
    json_path = OUT_DIR / "layer_drift_summary.json"
    with open(json_path, "w") as f:
        json.dump({
        "model": MODEL_ID,
        "steps": step_list[-1],
        "n_images": len(images),
        "images": [Path(p).stem for p in images],
        "top_10": top10,
        "full_ranking": [{"rank": i, "layer": l, "mean_drift": float(d)}
                         for i, (l, d) in enumerate(ranked, 1)],
        "per_image": {img: {l: float(v) for l, v in drifts.items()}
                      for img, drifts in drift_by_layer_img.items()},
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Heatmap ──
    img_names = [Path(p).stem for p in images]
    drift_matrix = np.array([
        [drift_by_layer_img[layer].get(img, 0) for img in img_names]
        for layer in all_layers
    ])
    # Log-scale for better visibility
    drift_log = np.log10(np.clip(drift_matrix, 1e-12, None))
    make_heatmap(drift_log, all_layers, img_names,
                 OUT_DIR / "sdxl_drift_heatmap.png")

    # ── Bar chart ──
    mean_drifts = [layer_mean.get(l, 0) for l in all_layers]
    make_bar_chart(all_layers, mean_drifts, OUT_DIR / "sdxl_top30_drift.png")

    print("\nDone.")
    print(f"  Heatmap:       {OUT_DIR / 'sdxl_drift_heatmap.png'}")
    print(f"  Bar chart:     {OUT_DIR / 'sdxl_top30_drift.png'}")
    print(f"  JSON summary:  {json_path}")


if __name__ == "__main__":
    main()
