"""
第一阶段完整诊断脚本：DDIM 反演 → 重建
- 多图 × 多步数 系统化实验
- 指标：PSNR / SSIM / LPIPS
- UNet 逐层特征 L2 距离（动态发现所有关键层）
- 漂移热力图 + 衰减曲线 + 层级柱状图
"""

import argparse
import json
import os
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
import lpips
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"

OUT_DIR = Path("outputs/phase1")
TEST_IMAGES = [
    "data/basetest/face1.jpg",
    "data/basetest/face2.jpg",
    "data/basetest/nature.jpg",
    "data/content.jpg",
    "data/watercolor.jpeg",
]
STEP_LIST = [4, 10, 20, 50, 100]


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_and_encode(pipe, path: str):
    img = Image.open(path).convert("RGB").resize((512, 512))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2 * tensor - 1  # [0,1] → [-1,1] (SD VAE training domain)
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent, tensor, img


def decode_latent(pipe, latent):
    with torch.no_grad():
        tensor = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    return tensor


# ---------------------------------------------------------------------------
# DDIM 反演 / 重建
# ---------------------------------------------------------------------------

def ddim_inversion(pipe, latents, prompt_embeds, num_steps, guidance_scale=1.0):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    uncond_embeds = None
    if guidance_scale > 1.0:
        uncond_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            latent_input = torch.cat([z] * 2) if guidance_scale > 1.0 else z
            emb = (
                torch.cat([uncond_embeds, prompt_embeds])
                if guidance_scale > 1.0
                else prompt_embeds
            )
            noise_pred = pipe.unet(latent_input, t_cur, encoder_hidden_states=emb).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    return z


def ddim_reconstruction(pipe, noise, prompt_embeds, num_steps, guidance_scale=1.0):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    uncond_embeds = None
    if guidance_scale > 1.0:
        uncond_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    z = noise.clone()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([z] * 2) if guidance_scale > 1.0 else z
            emb = (
                torch.cat([uncond_embeds, prompt_embeds])
                if guidance_scale > 1.0
                else prompt_embeds
            )
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=emb).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ---------------------------------------------------------------------------
# UNet 特征 Hook（动态发现所有关键层）
# ---------------------------------------------------------------------------

def discover_hook_targets(unet):
    """动态发现 UNet 中需要 hook 的关键层：每个 ResNet block 和 Attention block 的输出。"""
    targets = []

    for name, module in unet.named_modules():
        # Hook 每个 resnets.N（整个 ResNet block 的输出）
        parts = name.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            # name 如 "down_blocks.0.resnets.0" → 取 ResNet block
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(name)
        # Hook 每个 transformer_blocks.0（BasicTransformerBlock 的输出）
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(name)

    return sorted(targets)


class UNetFeatureHooker:
    """Hook UNet 中间层特征用于漂移分析。"""

    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.handles = []

        targets = discover_hook_targets(unet)
        print(f"  [Hook] 注册 {len(targets)} 个目标层...")
        for name in targets:
            mod = self._find_module(name)
            if mod is not None:
                handle = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook_fn(n, out)
                )
                self.handles.append(handle)
            else:
                print(f"    [警告] 未找到层: {name}")

    def _find_module(self, name):
        tokens = name.split(".")
        mod = self.unet
        for t in tokens:
            try:
                mod = getattr(mod, t)
            except AttributeError:
                return None
        return mod

    def _hook_fn(self, name, output):
        if isinstance(output, tuple):
            output = output[0]
        # 对 attention 输出取平均做降维（[B, N, C] → 取 [CLS] 或均值）
        if output.dim() == 3:
            output = output.mean(dim=1, keepdim=True)  # [B, 1, C] 用于后续比较
        self.features[name] = output.detach().cpu()

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_metrics(original_tensor, recon_tensor, lpips_fn=None):
    """计算 PSNR / SSIM / LPIPS。"""
    orig = original_tensor.float().clamp(-1, 1)
    recon = recon_tensor.float().clamp(-1, 1)

    mse = torch.nn.functional.mse_loss(orig, recon)
    psnr_val = (20 * torch.log10(2.0 / torch.sqrt(mse))).item()

    orig_np = (orig.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    recon_np = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    ssim_val = float(ssim(orig_np, recon_np, channel_axis=2, data_range=1.0))

    result = {"PSNR": float(psnr_val), "SSIM": ssim_val}
    if lpips_fn is not None:
        lpips_val = float(lpips_fn(orig, recon).item())
        result["LPIPS"] = lpips_val

    return result


# ---------------------------------------------------------------------------
# 漂移热力图
# ---------------------------------------------------------------------------

def save_heatmap(original_tensor, recon_tensor, out_path):
    diff = (recon_tensor.float() - original_tensor.float()).abs()
    diff_np = diff.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    diff_gray = diff_np.mean(axis=2)

    orig_np = ((original_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2)
    recon_np = ((recon_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_np)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    im = axes[2].imshow(diff_gray, cmap="hot")
    axes[2].set_title("|Recon - Original| (mean ch)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    diff_enhanced = np.clip(diff_gray * 5, 0, 1)
    axes[3].imshow(diff_enhanced, cmap="hot")
    axes[3].set_title("Drift x5 (enhanced)")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# UNet 层级漂移分析
# ---------------------------------------------------------------------------

def analyze_layer_drift(pipe, original_latent, prompt_embeds, num_steps, seeds=None):
    """在关键 timestep 比较原始路径与重建路径的 UNet 各层特征 L2 距离。

    方法：对选定 timestep t：
      - 参考: original_latent → DDPM 前向加噪至 t → UNet(z_ref, t) → 记录各层输出
      - 重建: 反演→重建，每步 UNet(z_recon, t) → 记录各层输出
    比较两组特征在各层的 L2 距离。多 seed 取平均和标准差。

    Returns: (avg_drifts, std_drifts) — 每个 dict 为 {layer_name: float}
    """
    if seeds is None:
        seeds = [42]

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    n = len(timesteps)

    # 选取早期/中期/晚期各 3 个关键步
    if n <= 6:
        key_indices = list(range(n))
    else:
        key_indices = [0, 1, 2, n//2 - 1, n//2, n//2 + 1, n-3, n-2, n-1]
    key_indices = sorted(set(max(0, min(n-1, i)) for i in key_indices))

    hooker = UNetFeatureHooker(pipe.unet)
    embs = prompt_embeds

    # 1) 反演（DDIM eta=0 是确定性的，与 seed 无关）
    print("    反演中...")
    inv_latent = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)

    # 2) 重建并记录每步的 latent
    print("    重建并收集特征...")
    recon_latents = [inv_latent.clone()]
    z = inv_latent.clone()
    with torch.no_grad():
        for _, t in enumerate(timesteps):
            noise_pred = pipe.unet(z, t, encoder_hidden_states=embs).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
            recon_latents.append(z.clone())

    # 3) 对每个 seed 独立做对比（DDPM 参考噪声依赖 seed）
    per_seed_results = defaultdict(list)

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        seed_drifts = defaultdict(list)
        with torch.no_grad():
            for idx in key_indices:
                t = timesteps[idx]
                alpha_t = scheduler.alphas_cumprod[t]
                noise_ref = torch.randn_like(original_latent)
                z_ref = alpha_t.sqrt() * original_latent + (1 - alpha_t).sqrt() * noise_ref
                z_recon = recon_latents[idx]

                hooker.clear()
                pipe.unet(z_ref, t.to(DEVICE), encoder_hidden_states=embs).sample
                ref_feats = hooker.features.copy()

                hooker.clear()
                pipe.unet(z_recon, t.to(DEVICE), encoder_hidden_states=embs).sample
                recon_feats = hooker.features.copy()

                for layer_name in ref_feats:
                    if layer_name not in recon_feats:
                        continue
                    l2 = torch.norm(
                        ref_feats[layer_name].float() - recon_feats[layer_name].float(),
                        p=2).item()
                    seed_drifts[layer_name].append(l2)

        # 此 seed 内跨 timestep 平均
        for layer_name, vals in seed_drifts.items():
            per_seed_results[layer_name].append(float(np.mean(vals)))

    hooker.remove()

    if not per_seed_results:
        return {}, {}

    avg_drifts = {k: float(np.mean(v)) for k, v in per_seed_results.items()}
    std_drifts = {k: float(np.std(v)) if len(v) > 1 else 0.0
                  for k, v in per_seed_results.items()}

    return avg_drifts, std_drifts


# ---------------------------------------------------------------------------
# 主实验循环
# ---------------------------------------------------------------------------

def run_experiments(pipe, lpips_fn, test_images, step_list):
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"[跳过] 文件不存在: {img_path}")
            continue

        img_name = Path(img_path).stem
        print(f"\n{'='*60}")
        print(f"[图片] {img_name}")
        print(f"{'='*60}")

        original_latent, original_tensor, pil_img = load_and_encode(pipe, img_path)

        for steps in step_list:
            print(f"  [{steps} 步] 反演 → 重建...", end=" ", flush=True)

            noise = ddim_inversion(pipe, original_latent, prompt_embeds, steps)
            recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, steps)
            recon_tensor = decode_latent(pipe, recon_latent)

            metrics = compute_metrics(original_tensor, recon_tensor, lpips_fn)
            results.append({"image": img_name, "steps": steps, **metrics})
            lpips_str = f" LPIPS={metrics['LPIPS']:.4f}" if "LPIPS" in metrics else ""
            print(f"PSNR={metrics['PSNR']:.2f} SSIM={metrics['SSIM']:.4f}{lpips_str}")

            recon_pil = transforms.ToPILImage()((recon_tensor.squeeze(0) / 2 + 0.5).clamp(0, 1))
            recon_dir = OUT_DIR / "reconstructions"
            os.makedirs(recon_dir, exist_ok=True)
            recon_pil.save(recon_dir / f"{img_name}_steps{steps}.png")

            heatmap_dir = OUT_DIR / "heatmaps"
            os.makedirs(heatmap_dir, exist_ok=True)
            save_heatmap(original_tensor, recon_tensor, heatmap_dir / f"{img_name}_steps{steps}.png")

            del noise, recon_latent, recon_tensor
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 结果可视化
# ---------------------------------------------------------------------------

def plot_decay_curves(results):
    if not results:
        return
    metric_keys = [k for k in results[0].keys() if k not in ("image", "steps")]
    df = {}
    for r in results:
        img = r["image"]
        if img not in df:
            df[img] = {"steps": []}
            for mk in metric_keys:
                df[img][mk] = []
        df[img]["steps"].append(r["steps"])
        for mk in metric_keys:
            df[img][mk].append(r[mk])

    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for (img_name, data), color in zip(df.items(), colors):
        for j, mk in enumerate(metric_keys):
            axes[j].plot(data["steps"], data[mk], "o-", color=color, label=img_name)

    for j, mk in enumerate(metric_keys):
        axes[j].set_xlabel("Steps"); axes[j].set_ylabel(mk)
        axes[j].set_title(f"{mk} vs Steps")
        if j == 0:
            axes[j].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "decay_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 衰减曲线 → {OUT_DIR / 'decay_curves.png'}")


def plot_layer_drift(layer_drifts, out_name="layer_drift.png", std_drifts=None):
    if not layer_drifts:
        print("[跳过] 无层级漂移数据")
        return

    # 按类型分组排序
    def sort_key(name):
        for prefix in ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
                       "mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]:
            if name.startswith(prefix):
                return prefix + name[len(prefix):]
        return name

    names = sorted(layer_drifts.keys(), key=sort_key)
    values = [layer_drifts[n] for n in names]
    errors = [std_drifts.get(n, 0) for n in names] if std_drifts else None

    # 缩短标签
    short_names = []
    for n in names:
        s = n.replace("down_blocks.", "D").replace("up_blocks.", "U") \
             .replace("mid_block.", "M").replace("resnets.", "R") \
             .replace("attentions.", "A").replace("transformer_blocks.", "T") \
             .replace(".", "")
        short_names.append(s)

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.6), 5))
    colors = []
    for n in names:
        if "down" in n:
            colors.append("#3498db")
        elif "mid" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    ax.bar(short_names, values, color=colors, yerr=errors, capsize=2,
           error_kw={"elinewidth": 0.5, "alpha": 0.5})
    ax.set_ylabel("Avg L2 Distance")
    title = "UNet Per-Layer Feature Drift"
    if std_drifts and len(std_drifts) > 0:
        n_seeds_est = sum(1 for v in std_drifts.values() if v > 0)
        if n_seeds_est > 0:
            title += f" (±1σ, multi-seed)"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Down blocks"),
        Patch(facecolor="#e74c3c", label="Mid block"),
        Patch(facecolor="#2ecc71", label="Up blocks"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 层级漂移 → {OUT_DIR / out_name}")


def save_aggregated_drift(all_layer_drifts, all_std_drifts=None, seeds=None):
    """跨图聚合层漂移：计算 mean/std，排序，保存 summary JSON。"""
    if not all_layer_drifts:
        return

    per_layer = defaultdict(list)
    for img_name, drifts in all_layer_drifts.items():
        for layer, val in drifts.items():
            per_layer[layer].append(val)

    aggregated = {}
    for layer, vals in per_layer.items():
        aggregated[layer] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_images": len(vals),
        }

    ranked = sorted(aggregated.items(), key=lambda x: x[1]["mean"], reverse=True)

    summary = {
        "n_images": len(all_layer_drifts),
        "images": sorted(all_layer_drifts.keys()),
        "seeds": seeds or [42],
        "per_image": {k: v for k, v in all_layer_drifts.items()},
        "aggregated": aggregated,
        "top_3": [layer for layer, _ in ranked[:3]],
        "top_5": [layer for layer, _ in ranked[:5]],
        "top_10": [layer for layer, _ in ranked[:10]],
        "ranking": [{"layer": layer, "mean": v["mean"], "std": v["std"]}
                    for layer, v in ranked],
    }

    if all_std_drifts:
        summary["per_image_std"] = {k: v for k, v in all_std_drifts.items()}

    path = OUT_DIR / "layer_drift_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 跨图聚合漂移 → {path}")

    # 打印 Top-10
    print("\n  Top-10 漂移层 (跨图平均):")
    print(f"  {'Rank':>4s} {'Layer':<45s} {'Mean L2':>10s} {'Std':>10s}")
    for i, (layer, v) in enumerate(ranked[:10]):
        print(f"  {i+1:4d} {layer:<45s} {v['mean']:10.1f} {v['std']:10.1f}")


def save_results_csv(results):
    csv_path = OUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fields = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] 指标汇总 → {csv_path}")

    json_path = OUT_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 指标汇总 → {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: 完整诊断实验")
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片测试（不指定则跑全部）")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="反演步数列表（默认 4 10 20 50 100）")
    parser.add_argument("--skip-layer", action="store_true",
                        help="跳过 UNet 层级漂移分析（加速）")
    parser.add_argument("--skip-lpips", action="store_true",
                        help="跳过 LPIPS（加速）")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="跳过主实验，仅做层级分析")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="层漂移分析随机种子列表 (默认 42)")
    args = parser.parse_args()

    test_images = [args.image] if args.image else TEST_IMAGES
    step_list = args.steps if args.steps else STEP_LIST

    print(f"[设备] {DEVICE}")
    print(f"[输出] {OUT_DIR.resolve()}")
    print(f"[图片] {test_images}")
    print(f"[步数] {step_list}")

    # 加载模型
    print("[0] 加载模型...")
    pipe = load_pipeline()

    results = []

    if not args.skip_experiments:
        lpips_fn = None
        if not args.skip_lpips:
            print("[1] 加载 LPIPS...")
            lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
        else:
            print("[1] 跳过 LPIPS")

        print("[2] 运行主实验...")
        results = run_experiments(pipe, lpips_fn, test_images, step_list)
        save_results_csv(results)
        plot_decay_curves(results)

    # UNet 层级漂移分析
    if not args.skip_layer:
        print(f"[3] UNet 层级漂移分析 (seeds={args.seeds}, {len(test_images)} 图)...")
        all_layer_drifts = {}
        all_std_drifts = {}
        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"  [跳过] 文件不存在: {img_path}")
                continue
            img_name = Path(img_path).stem
            print(f"\n  === {img_name} ===")
            latent, _, _ = load_and_encode(pipe, img_path)
            embs = pipe.encode_prompt("", DEVICE, 1, False)[0]
            for steps in [50]:
                print(f"  [{steps} 步] 层级漂移 ({len(args.seeds)} seeds)...")
                avg_drifts, std_drifts = analyze_layer_drift(
                    pipe, latent, embs, steps, seeds=args.seeds)
                if avg_drifts:
                    all_layer_drifts[img_name] = avg_drifts
                    all_std_drifts[img_name] = std_drifts
                    plot_layer_drift(avg_drifts,
                                     f"layer_drift_{img_name}_{steps}.png",
                                     std_drifts=std_drifts)
                    with open(OUT_DIR / f"layer_drift_{img_name}_{steps}.json", "w") as f:
                        json.dump({"avg": avg_drifts, "std": std_drifts,
                                   "seeds": args.seeds}, f, indent=2)
        if all_layer_drifts:
            save_aggregated_drift(all_layer_drifts, all_std_drifts, seeds=args.seeds)
        elif not any(os.path.exists(p) for p in test_images):
            print("  无可用图片")
    else:
        print("[3] 跳过层级分析")

    print(f"\n{'='*60}")
    print("第一阶段诊断完成。")
    print(f"输出目录: {OUT_DIR.resolve()}")
    print(f"  - metrics.csv / metrics.json  指标汇总")
    print(f"  - decay_curves.png            衰减曲线")
    print(f"  - reconstructions/            重建图")
    print(f"  - heatmaps/                   漂移热力图")
    print(f"  - layer_drift_*.png           层级漂移柱状图")


if __name__ == "__main__":
    main()
