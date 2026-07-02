"""
DiT Phase 2 共享模块：HunyuanDiT pipeline 加载、DDIM 操作、FeatureCorrector、指标

与 SD 版本的 phase2_common.py 对应，适配 HunyuanDiT 架构差异：
- 40 个 transformer blocks（非 UNet 结构）
- v_prediction（非 epsilon）
- 双文本编码器（CLIP + T5）
- 3D token 特征 [B, N, D]
"""
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import HunyuanDiTPipeline, DDIMScheduler
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
IMAGE_SIZE = 1024

# ═══════════════════════════════════════════════════════════════
# Phase 1 诊断结果驱动的层组（5 图平均，50 步）
# ═══════════════════════════════════════════════════════════════

# Top-5 漂移层（过渡区 + 首个 skip block）
TOP_5_LAYERS = [
    "blocks.20",  # 第一个 skip connection block，始终最高
    "blocks.19",  # 最后一个 bottom block
    "blocks.18",
    "blocks.17",
    "blocks.16",
]

TOP_10_LAYERS = TOP_5_LAYERS + [
    "blocks.15", "blocks.14", "blocks.13", "blocks.21", "blocks.12",
]

# 过渡区 (blocks.11-21)：bottom→top 转换区，总漂移最大
TRANSITION_ZONE = [f"blocks.{i}" for i in range(11, 22)]

# 深层 bottom blocks (0-10)：低漂移
BOTTOM_EARLY = [f"blocks.{i}" for i in range(0, 11)]

# 深层 top blocks (22-39)：最低漂移
TOP_LATE = [f"blocks.{i}" for i in range(22, 40)]

# Top-5 from different regions for ablation
REGION_TOP5 = {
    "transition": ["blocks.20", "blocks.19", "blocks.18", "blocks.17", "blocks.16"],
    "bottom": ["blocks.19", "blocks.18", "blocks.17", "blocks.16", "blocks.15"],
    "top": ["blocks.20", "blocks.21", "blocks.22", "blocks.23", "blocks.24"],
}

LAYER_GROUPS = {
    "top5": TOP_5_LAYERS,
    "top10": TOP_10_LAYERS,
    "transition": TRANSITION_ZONE,
    "bottom_early": BOTTOM_EARLY,
    "top_late": TOP_LATE,
    "region_transition": REGION_TOP5["transition"],
    "region_bottom": REGION_TOP5["bottom"],
    "region_top": REGION_TOP5["top"],
    "all": [f"blocks.{i}" for i in range(40)],
}


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════

def load_pipeline():
    pipe = HunyuanDiTPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, local_files_only=True,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, prediction_type="v_prediction",
    )
    pipe.vae.to(torch.float32)
    return pipe


# ═══════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════

def load_and_encode(pipe, path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    tensor = 2 * tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent.to(dtype=DTYPE), tensor, img


def decode_latent(pipe, latent):
    with torch.no_grad():
        tensor = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor).sample
    return tensor


# ═══════════════════════════════════════════════════════════════
# Prompt encoding
# ═══════════════════════════════════════════════════════════════

def encode_prompt_dit(pipe, prompt=""):
    embeds, _, mask, _ = pipe.encode_prompt(
        prompt=prompt, device=DEVICE, dtype=DTYPE,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
    )
    embeds_2, _, mask_2, _ = pipe.encode_prompt(
        prompt=prompt, device=DEVICE, dtype=DTYPE,
        num_images_per_prompt=1, do_classifier_free_guidance=False,
        text_encoder_index=1,
    )
    return {
        "encoder_hidden_states": embeds,
        "text_embedding_mask": mask,
        "encoder_hidden_states_t5": embeds_2,
        "text_embedding_mask_t5": mask_2,
    }


# ═══════════════════════════════════════════════════════════════
# Transformer forward
# ═══════════════════════════════════════════════════════════════

def dit_forward(transformer, z, t, cond):
    """HunyuanDiT forward, returns noise_pred only (splits 8-ch output)."""
    t_tensor = torch.tensor([t], device=DEVICE, dtype=DTYPE)
    out = transformer(
        hidden_states=z, timestep=t_tensor,
        encoder_hidden_states=cond["encoder_hidden_states"],
        text_embedding_mask=cond["text_embedding_mask"],
        encoder_hidden_states_t5=cond["encoder_hidden_states_t5"],
        text_embedding_mask_t5=cond["text_embedding_mask_t5"],
        image_meta_size=None, style=None, image_rotary_emb=None,
        return_dict=True,
    ).sample
    noise_pred, _ = out.chunk(2, dim=1)
    return noise_pred


# ═══════════════════════════════════════════════════════════════
# Feature Collector / Corrector（形状无关，直接复用 pattern）
# ═══════════════════════════════════════════════════════════════

class FeatureCollector:
    """Hook into transformer blocks and collect features during forward pass."""

    def __init__(self, transformer, layers):
        self.features = {}
        self.handles = []
        self.transformer = transformer
        for name in layers:
            mod = self._find(name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self.features.__setitem__(n, out)
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

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class FeatureCorrector:
    """Hook into transformer blocks and apply residual correction.

    f_corrected = f_output + lam * (f_reference - f_output)
    Works with both 4D [B,C,H,W] and 3D [B,N,D] tensors.
    """

    def __init__(self, transformer, layers, lam):
        self.transformer = transformer
        self.reference_features = {}
        self.lam = lam
        self.current_step = 0
        self.handles = []

        for name in layers:
            mod = self._find(name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook(n, out)
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

    def set_reference(self, ref_dict, step_idx=None):
        self.reference_features = {k: v.clone() for k, v in ref_dict.items()}
        if step_idx is not None:
            self.current_step = step_idx

    def _hook(self, name, output):
        if name not in self.reference_features:
            return output
        ref = self.reference_features[name].to(device=output.device, dtype=output.dtype)
        lam_val = self.lam.get(self.current_step) if hasattr(self.lam, "get") else self.lam
        corrected = output + lam_val * (ref - output)
        return (corrected,) + output[1:] if isinstance(output, tuple) else corrected

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ═══════════════════════════════════════════════════════════════
# LambdaScheduler
# ═══════════════════════════════════════════════════════════════

class LambdaScheduler:
    def __init__(self, lam, num_steps, mode="linear", alpha=3.0):
        self.lam = lam
        self.num_steps = num_steps
        self.mode = mode
        self.alpha = alpha

    def get(self, step_idx):
        if self.mode == "constant":
            return self.lam
        t = step_idx / max(self.num_steps - 1, 1)
        if self.mode == "linear":
            return self.lam * (1.0 - t)
        elif self.mode == "exp":
            return self.lam * np.exp(-self.alpha * t)
        return self.lam


# ═══════════════════════════════════════════════════════════════
# DDIM Inversion / Reconstruction (v_prediction-aware)
# ═══════════════════════════════════════════════════════════════

def ddim_inversion_with_features(pipe, latents, cond, num_steps, hook_layers):
    """DDIM inversion (v_prediction-aware), saving features at each step."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    collector = FeatureCollector(pipe.transformer, hook_layers)
    saved_features = {}
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            collector.clear()
            v_pred = dit_forward(pipe.transformer, z, t_cur, cond)

            # Save features from this step
            saved_features[int(t_cur)] = {
                k: v.detach().cpu().clone() for k, v in collector.features.items()
            }

            # v_pred → x0, eps → DDIM step to noisier timestep
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            sigma_cur = (1 - alpha_cur).sqrt()
            x0_pred = alpha_cur.sqrt() * z - sigma_cur * v_pred
            eps_pred = sigma_cur * z + alpha_cur.sqrt() * v_pred
            alpha_next = scheduler.alphas_cumprod[t_next]
            sigma_next = (1 - alpha_next).sqrt()
            z = alpha_next.sqrt() * x0_pred + sigma_next * eps_pred

    collector.remove()
    return z, saved_features


def ddim_reconstruction_with_correction(pipe, noise, cond, num_steps, corrector,
                                         saved_features):
    """DDIM reconstruction with FeatureCorrector hooks active.

    Uses step-matched features: denoising at timestep t uses inversion features
    from the same timestep t (collected during inversion).
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()

    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            if t_int in saved_features:
                corrector.set_reference(saved_features[t_int], step_idx)
            noise_pred = dit_forward(pipe.transformer, z, t, cond)
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


def ddim_inversion_baseline(pipe, latents, cond, num_steps):
    """Baseline DDIM inversion without feature collection."""
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
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            sigma_cur = (1 - alpha_cur).sqrt()
            x0_pred = alpha_cur.sqrt() * z - sigma_cur * v_pred
            eps_pred = sigma_cur * z + alpha_cur.sqrt() * v_pred
            alpha_next = scheduler.alphas_cumprod[t_next]
            sigma_next = (1 - alpha_next).sqrt()
            z = alpha_next.sqrt() * x0_pred + sigma_next * eps_pred

    return z


def ddim_reconstruction_baseline(pipe, noise, cond, num_steps):
    """Baseline DDIM reconstruction without correction."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            noise_pred = dit_forward(pipe.transformer, z, t, cond)
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(orig_tensor, recon_tensor):
    """PSNR, SSIM, LPIPS between two [-1,1] tensors."""
    from skimage.metrics import structural_similarity as ssim
    import lpips

    orig = orig_tensor.detach()
    recon = recon_tensor.detach()
    orig_01 = (orig + 1) / 2
    recon_01 = (recon + 1) / 2

    mse_val = torch.nn.functional.mse_loss(orig_01, recon_01).item()
    psnr_val = 20 * np.log10(1.0) - 10 * np.log10(mse_val) if mse_val > 0 else 100.0

    ssim_val = ssim(
        orig_01.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        recon_01.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        channel_axis=2, data_range=1.0,
    )

    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_val = lpips_fn(orig_01, recon_01).item()

    return {"PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val}


# ═══════════════════════════════════════════════════════════════
# Saving
# ═══════════════════════════════════════════════════════════════

def save_recon_img(tensor, path):
    """Save [-1,1] tensor as PNG."""
    from torchvision.utils import save_image
    img = (tensor.squeeze(0) + 1) / 2
    save_image(img.clamp(0, 1), path)


def save_results_csv(results, path):
    with open(path, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def save_results_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
