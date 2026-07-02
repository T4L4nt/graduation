"""
DiT Phase 1: DDIM 反演-重建漂移动态诊断 (HunyuanDiT)

在 HunyuanDiT (纯 Transformer 扩散模型) 上复现 Phase 1 诊断流程：
- 单图 DDIM 反演 → 重建 PSNR/SSIM/LPIPS
- Transformer 各层特征漂移（inv 与 recon 特征 L2 距离）
- 40 个 HunyuanDiTBlock 的漂移排序

与 SD 1.5 / SDXL 输出对比用于论文 DiT 泛化章节。
"""
import argparse, json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from diffusers import HunyuanDiTPipeline, DDIMScheduler
from torchvision import transforms
import lpips
from skimage.metrics import structural_similarity as ssim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"

OUT_DIR = Path("outputs/dit_phase1")
TEST_IMAGES = [
    "data/basetest/face1.jpg",
    "data/basetest/face2.jpg",
    "data/basetest/nature.jpg",
    "data/basetest/architecture.jpg",
    "data/basetest/still_life.jpg",
]
STEP_LIST = [10, 20, 50]
IMAGE_SIZE = 1024


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

def load_pipeline():
    """Load HunyuanDiT with DDIM scheduler (v_prediction aware)."""
    pipe = HunyuanDiTPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, local_files_only=True,
    ).to(DEVICE)
    # Replace DDPMScheduler with DDIMScheduler, keeping v_prediction config
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction",
    )
    # VAE runs fp32 to avoid NaN
    pipe.vae.to(torch.float32)
    return pipe


# ═══════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════

def load_and_encode(pipe, path, size=IMAGE_SIZE):
    """Load image → resize → [-1,1] tensor + VAE latent (VAE runs fp32)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
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
# DiT prompt encoding
# ═══════════════════════════════════════════════════════════════

def encode_prompt_dit(pipe, prompt="", device=DEVICE):
    """Encode prompt for HunyuanDiT dual-encoder (CLIP/BERT + T5).

    Returns dict with all conditioning tensors needed for transformer forward.
    """
    prompt_embeds, negative_prompt_embeds, \
    prompt_attention_mask, negative_prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt, device=device, dtype=DTYPE,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    # Encode T5 (text_encoder_2)
    prompt_embeds_2, negative_prompt_embeds_2, \
    prompt_attention_mask_2, negative_prompt_attention_mask_2 = pipe.encode_prompt(
        prompt=prompt, device=device, dtype=DTYPE,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
        text_encoder_index=1,
    )
    return {
        "encoder_hidden_states": prompt_embeds,
        "text_embedding_mask": prompt_attention_mask,
        "encoder_hidden_states_t5": prompt_embeds_2,
        "text_embedding_mask_t5": prompt_attention_mask_2,
    }


# ═══════════════════════════════════════════════════════════════
# DiT transformer forward wrapper
# ═══════════════════════════════════════════════════════════════

def dit_forward(transformer, z, t, cond):
    """Call HunyuanDiT transformer with all required conditioning.

    HunyuanDiT outputs 8 channels (learn_sigma=True):
    first 4 = noise/v_prediction, last 4 = variance. We only need the prediction.
    """
    t_tensor = torch.tensor([t], device=DEVICE, dtype=DTYPE)
    out = transformer(
        hidden_states=z,
        timestep=t_tensor,
        encoder_hidden_states=cond["encoder_hidden_states"],
        text_embedding_mask=cond["text_embedding_mask"],
        encoder_hidden_states_t5=cond["encoder_hidden_states_t5"],
        text_embedding_mask_t5=cond["text_embedding_mask_t5"],
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        return_dict=True,
    ).sample
    # Split: first half is noise/v_prediction, second half is variance
    noise_pred, _ = out.chunk(2, dim=1)
    return noise_pred


# ═══════════════════════════════════════════════════════════════
# DDIM inversion / reconstruction (v_prediction-aware)
# ═══════════════════════════════════════════════════════════════

def ddim_inversion_final_features(pipe, latents, cond, num_steps, hooker):
    """Run DDIM inversion (v_prediction-aware) and collect final-step features.

    v_prediction: v = sqrt(alpha)*eps - sqrt(1-alpha)*x_0
    Conversion: x_0 = sqrt(alpha)*z - sqrt(1-alpha)*v
                eps = sqrt(1-alpha)*z + sqrt(alpha)*v
    DDIM step: z_next = sqrt(alpha_next)*x_0 + sqrt(1-alpha_next)*eps
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            v_pred = dit_forward(pipe.transformer, z, t_cur, cond)

            # Convert v_prediction to x_0 and epsilon
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            sigma_cur = (1 - alpha_cur).sqrt()
            x0_pred = alpha_cur.sqrt() * z - sigma_cur * v_pred
            eps_pred = sigma_cur * z + alpha_cur.sqrt() * v_pred

            # DDIM step to noisier timestep
            alpha_next = scheduler.alphas_cumprod[t_next]
            sigma_next = (1 - alpha_next).sqrt()
            z = alpha_next.sqrt() * x0_pred + sigma_next * eps_pred

        # Collect features after the last inversion step
        hooker.features = {
            k: v.detach().cpu().clone() for k, v in hooker.features.items()
        }
    return z, hooker.features


def ddim_reconstruction_final_features(pipe, noise, cond, num_steps, hooker):
    """Reconstruct and collect features from the FINAL step only."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    hooker.features = {}

    with torch.no_grad():
        for t in timesteps:
            noise_pred = dit_forward(pipe.transformer, z, t, cond)
            z = scheduler.step(noise_pred, t, z).prev_sample
            # Capture features from the last (cleanest) step
            hooker.features = {
                k: v.detach().cpu().clone() for k, v in hooker.features.items()
            }
    return z, hooker.features


# ═══════════════════════════════════════════════════════════════
# Feature hooking for HunyuanDiT
# ═══════════════════════════════════════════════════════════════

def discover_dit_hook_targets(transformer):
    """Discover all HunyuanDiTBlock outputs (blocks.0 through blocks.39)."""
    targets = []
    for name, _ in transformer.named_modules():
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit():
            targets.append(name)
    return sorted(targets, key=lambda x: int(x.split(".")[1]))


class DiTFeatureHooker:
    """Hook into HunyuanDiT transformer blocks to capture intermediate features."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.features = {}
        self.handles = []

    def _find_module(self, name):
        tokens = name.split(".")
        mod = self.transformer
        for t in tokens:
            try:
                mod = getattr(mod, t)
            except AttributeError:
                return None
        return mod

    def register(self):
        for name in discover_dit_hook_targets(self.transformer):
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
    """Compute PSNR, SSIM, LPIPS between original and reconstructed images."""
    orig = original_tensor.detach()
    recon = recon_tensor.detach()
    orig_01 = (orig + 1) / 2
    recon_01 = (recon + 1) / 2

    mse_val = torch.nn.functional.mse_loss(orig_01, recon_01).item()
    psnr_val = 20 * np.log10(1.0) - 10 * np.log10(mse_val) if mse_val > 0 else 100.0

    orig_np = orig_01.cpu().squeeze(0).permute(1, 2, 0).numpy()
    recon_np = recon_01.cpu().squeeze(0).permute(1, 2, 0).numpy()
    ssim_val = ssim(orig_np, recon_np, channel_axis=2, data_range=1.0)

    lpips_device = next(lpips_fn.parameters()).device
    lpips_val = lpips_fn(orig_01.to(lpips_device), recon_01.to(lpips_device)).item()

    return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def make_heatmap(drift_matrix, layer_names, img_names, out_path):
    """Heatmap: rows=layers, cols=images, values=log10(drift)."""
    fig, ax = plt.subplots(figsize=(max(8, len(img_names) * 1.5),
                                    max(6, len(layer_names) * 0.35)))
    im = ax.imshow(drift_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(img_names)))
    ax.set_xticklabels(img_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_title("HunyuanDiT Block Drift (MSE inv vs recon)", fontsize=12)
    plt.colorbar(im, ax=ax, label="log10(MSE drift)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_bar_chart(layer_names, mean_drifts, out_path, top_n=40):
    """Bar chart of mean drift per block."""
    idx = np.argsort(mean_drifts)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(idx)), [mean_drifts[i] for i in idx], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([layer_names[i] for i in idx], fontsize=7)
    ax.set_xlabel("Mean MSE drift")
    title = f"HunyuanDiT: Block Drift (avg over images)"
    ax.set_title(title)
    # Color bottom vs top blocks differently
    for i, j in enumerate(idx):
        block_num = int(layer_names[j].split(".")[1])
        if block_num >= 20:
            ax.get_children()[i].set_color("darkorange")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--step-scan", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HunyuanDiT Phase 1: DDIM Inversion-Reconstruction Drift Diagnostics")
    print("=" * 60)

    print("\n[1/4] Loading HunyuanDiT pipeline with DDIM scheduler...")
    pipe = load_pipeline()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    # Discover hook targets (40 transformer blocks)
    all_layers = discover_dit_hook_targets(pipe.transformer)
    print(f"\n[2/4] Discovered {len(all_layers)} transformer blocks:")
    print(f"  Bottom blocks (0-19, no skip): {len([l for l in all_layers if int(l.split('.')[1]) < 20])}")
    print(f"  Top blocks (20-39, with skip):  {len([l for l in all_layers if int(l.split('.')[1]) >= 20])}")

    images = TEST_IMAGES[:args.n_images] if args.n_images else TEST_IMAGES
    print(f"\n[3/4] Test images ({len(images)}):")
    for p in images:
        print(f"  {p}")

    step_list = STEP_LIST if args.step_scan else [args.steps]

    all_results = []
    drift_by_layer_img = defaultdict(dict)

    for num_steps in step_list:
        print(f"\n{'='*60}")
        print(f"  Steps = {num_steps}")
        print(f"{'='*60}")

        for img_idx, img_path in enumerate(images):
            img_name = Path(img_path).stem
            print(f"\n  [{img_idx+1}/{len(images)}] {img_name}")

            # Load & encode
            latents, tensor, _ = load_and_encode(pipe, img_path)
            cond = encode_prompt_dit(pipe, "")

            # ---- Inversion with final-step features ----
            inv_hooker = DiTFeatureHooker(pipe.transformer)
            inv_hooker.register()
            noise, inv_features = ddim_inversion_final_features(
                pipe, latents, cond, num_steps, inv_hooker,
            )
            inv_hooker.remove()

            # ---- Reconstruction with final-step features ----
            recon_hooker = DiTFeatureHooker(pipe.transformer)
            recon_hooker.register()
            recon_latents, recon_features = ddim_reconstruction_final_features(
                pipe, noise, cond, num_steps, recon_hooker,
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
                    # For [B, N, D] tensors: flatten to 1D for MSE
                    drift = torch.nn.functional.mse_loss(
                        inv_features[layer].float(), recon_features[layer].float()
                    ).item()
                else:
                    drift = 0.0
                avg_drifts[layer] = drift
                drift_by_layer_img[layer][img_name] = drift

            # Log top-5 drift
            for layer, d in sorted(avg_drifts.items(), key=lambda x: -x[1])[:5]:
                blk = int(layer.split(".")[1])
                region = "top" if blk >= 20 else "bottom"
                print(f"      {layer:12s} ({region:6s})  drift={d:.6e}")

            all_results.append({
                "image": img_name, "steps": num_steps,
                "metrics": metrics, "drifts": avg_drifts,
            })

    # ── Summarize ──
    print(f"\n{'='*60}")
    print("Summary: Layer drift ranking (averaged over images)")

    layer_mean = {}
    for layer in all_layers:
        vals = [drift_by_layer_img[layer][img]
                for img in [Path(p).stem for p in images]
                if img in drift_by_layer_img[layer]]
        if vals:
            layer_mean[layer] = np.mean(vals)

    ranked = sorted(layer_mean.items(), key=lambda x: -x[1])
    top10 = [l for l, _ in ranked[:10]]

    print(f"\n{'Rank':<6}{'Block':<12}{'Mean Drift':<16}{'Region'}")
    print("-" * 50)
    for rank, (layer, drift) in enumerate(ranked, 1):
        blk = int(layer.split(".")[1])
        region = "top (skip)" if blk >= 20 else "bottom"
        marker = " <--" if rank <= 10 else ""
        print(f"{rank:<6}{layer:<12}{drift:<16.6e}{region}{marker}")

    # ── Statistics by region ──
    bottom_drifts = [d for l, d in layer_mean.items() if int(l.split(".")[1]) < 20]
    top_drifts = [d for l, d in layer_mean.items() if int(l.split(".")[1]) >= 20]
    print(f"\nBottom blocks (0-19): mean drift = {np.mean(bottom_drifts):.6e}")
    print(f"Top blocks (20-39):   mean drift = {np.mean(top_drifts):.6e}")

    # ── Save JSON ──
    json_path = OUT_DIR / "layer_drift_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": MODEL_ID, "steps": step_list[-1],
            "n_images": len(images),
            "images": [Path(p).stem for p in images],
            "architecture": "HunyuanDiT (40 transformer blocks, v_prediction)",
            "block_structure": "blocks.0-19 (bottom, no skip), blocks.20-39 (top, with skip)",
            "top_10": top10,
            "full_ranking": [{"rank": i, "layer": l, "mean_drift": float(d)}
                             for i, (l, d) in enumerate(ranked, 1)],
            "region_stats": {
                "bottom_mean": float(np.mean(bottom_drifts)),
                "top_mean": float(np.mean(top_drifts)),
            },
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
    drift_log = np.log10(np.clip(drift_matrix, 1e-12, None))
    make_heatmap(drift_log, all_layers, img_names,
                 OUT_DIR / "dit_drift_heatmap.png")

    # ── Bar chart ──
    mean_drifts = [layer_mean.get(l, 0) for l in all_layers]
    make_bar_chart(all_layers, mean_drifts, OUT_DIR / "dit_drift_barchart.png")

    print("\nDone.")
    print(f"  Heatmap:   {OUT_DIR / 'dit_drift_heatmap.png'}")
    print(f"  Bar chart: {OUT_DIR / 'dit_drift_barchart.png'}")
    print(f"  JSON:      {json_path}")


if __name__ == "__main__":
    main()
