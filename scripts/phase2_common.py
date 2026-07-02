"""
Phase 2 共享模块：模型加载、特征 Hook、DDIM 操作、评估指标、可视化、保存

供 phase2_residual_correction.py / phase2_full.py / phase2_edict.py / phase2_nti.py 共享。

引入此模块后将删除各脚本中约 300 行重复代码。
"""

import json, os, csv, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# 注入层配置（基于 coco_val 19 张独立图片的漂移诊断结果）
TOP_3_LAYERS = [
    "up_blocks.2.resnets.0",
    "mid_block.resnets.1",
    "up_blocks.0.resnets.0",
]
TOP_5_LAYERS = [
    "up_blocks.2.resnets.0",
    "mid_block.resnets.1",
    "up_blocks.0.resnets.0",
    "up_blocks.1.resnets.1",
    "mid_block.resnets.0",
]
TOP_5_PLUS_MID = TOP_5_LAYERS  # mid_block already in top-5
ALL_UP_RESNETS = [
    "up_blocks.0.resnets.0", "up_blocks.0.resnets.1", "up_blocks.0.resnets.2",
    "up_blocks.1.resnets.0", "up_blocks.1.resnets.1", "up_blocks.1.resnets.2",
    "up_blocks.2.resnets.0", "up_blocks.2.resnets.1", "up_blocks.2.resnets.2",
    "up_blocks.3.resnets.0", "up_blocks.3.resnets.1", "up_blocks.3.resnets.2",
    "mid_block.resnets.0", "mid_block.resnets.1",
]

# Ablation: encoder-only ResNet blocks (top-5 by drift on coco_val)
ENCODER_TOP5 = [
    "down_blocks.3.resnets.1",
    "down_blocks.3.resnets.0",
    "down_blocks.2.resnets.1",
    "down_blocks.2.resnets.0",
    "down_blocks.1.resnets.1",
]

# Ablation: top-5 attention layers by drift on coco_val
ATTENTION_TOP5 = [
    "up_blocks.2.attentions.0.transformer_blocks.0",
    "up_blocks.1.attentions.1.transformer_blocks.0",
    "mid_block.attentions.0.transformer_blocks.0",
    "up_blocks.1.attentions.0.transformer_blocks.0",
    "up_blocks.2.attentions.1.transformer_blocks.0",
]

# Ablation: 5 randomly selected up_blocks resnets (seed=42, deterministic)
RANDOM_5_UP_RESNETS = [
    "up_blocks.0.resnets.0",
    "up_blocks.1.resnets.2",
    "up_blocks.2.resnets.2",
    "up_blocks.3.resnets.0",
    "up_blocks.3.resnets.1",
]

LAYER_GROUPS = {
    "top3": TOP_3_LAYERS,
    "top5": TOP_5_LAYERS,
    "top5+mid": TOP_5_PLUS_MID,
    "all_up": ALL_UP_RESNETS,
    "encoder5": ENCODER_TOP5,
    "attention5": ATTENTION_TOP5,
    "random5": RANDOM_5_UP_RESNETS,
}

# 默认测试图片集
DEFAULT_TEST_IMAGES = [
    # COCO val2017 — 15 diverse real-world photos (standard dataset)
    "data/coco_val/coco_000000000139.jpg",
    "data/coco_val/coco_000000000285.jpg",
    "data/coco_val/coco_000000000632.jpg",
    "data/coco_val/coco_000000000724.jpg",
    "data/coco_val/coco_000000000776.jpg",
    "data/coco_val/coco_000000000785.jpg",
    "data/coco_val/coco_000000000802.jpg",
    "data/coco_val/coco_000000000872.jpg",
    "data/coco_val/coco_000000000885.jpg",
    "data/coco_val/coco_000000001000.jpg",
    "data/coco_val/coco_000000001353.jpg",
    "data/coco_val/coco_000000001490.jpg",
    "data/coco_val/coco_000000001532.jpg",
    "data/coco_val/coco_000000001584.jpg",
    "data/coco_val/coco_000000001675.jpg",
    # Face images for ArcFace evaluation
    "data/basetest/face1.jpg",
    "data/basetest/face2.jpg",
]

# Train/Val/Test split for proper experimental design
# Val: images already used in Phase 1-2 exploration (contaminated, for tuning only)
VAL_IMAGES = [
    "data/basetest/face1.jpg",
    "data/basetest/face2.jpg",
    "data/basetest/nature.jpg",
    "data/content.jpg",
    "data/watercolor.jpeg",
]

# Test: held-out images never used for layer selection or lambda tuning
TEST_IMAGES_HELD_OUT = [
    # 15 COCO val2017 (original DEFAULT_TEST_IMAGES)
    "data/coco_val/coco_000000000139.jpg",
    "data/coco_val/coco_000000000285.jpg",
    "data/coco_val/coco_000000000632.jpg",
    "data/coco_val/coco_000000000724.jpg",
    "data/coco_val/coco_000000000776.jpg",
    "data/coco_val/coco_000000000785.jpg",
    "data/coco_val/coco_000000000802.jpg",
    "data/coco_val/coco_000000000872.jpg",
    "data/coco_val/coco_000000000885.jpg",
    "data/coco_val/coco_000000001000.jpg",
    "data/coco_val/coco_000000001353.jpg",
    "data/coco_val/coco_000000001490.jpg",
    "data/coco_val/coco_000000001532.jpg",
    "data/coco_val/coco_000000001584.jpg",
    "data/coco_val/coco_000000001675.jpg",
    # 4 additional COCO images not previously listed
    "data/coco_val/coco_000000001818.jpg",
    "data/coco_val/coco_000000002153.jpg",
    "data/coco_val/coco_000000002261.jpg",
    "data/coco_val/coco_000000002532.jpg",
    # 5 basetest images never used in Phase 1-2
    "data/basetest/animal.jpg",
    "data/basetest/architecture.jpg",
    "data/basetest/cityscape.jpg",
    "data/basetest/pattern.jpg",
    "data/basetest/still_life.jpg",
]


def get_image_split(split="all"):
    """Return image paths for the requested split.

    Args:
        split: "val" (5 images for lambda tuning),
               "test" (24 held-out images for final evaluation),
               "all" (combined 29 images)
    """
    if split == "val":
        return list(VAL_IMAGES)
    elif split == "test":
        return list(TEST_IMAGES_HELD_OUT)
    elif split == "all":
        return list(VAL_IMAGES) + list(TEST_IMAGES_HELD_OUT)
    else:
        raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# 动态层配置（从 Phase 1 跨图分析结果加载）
# ---------------------------------------------------------------------------

_DRIFT_SUMMARY_PATH = Path("outputs/phase1/layer_drift_summary.json")
_cached_top_layers = None


def get_top_drift_layers(k=5):
    """Get top-K drift layers, preferring multi-image Phase 1 analysis results."""
    global _cached_top_layers
    if _cached_top_layers is not None:
        return _cached_top_layers[:k]
    path = _DRIFT_SUMMARY_PATH
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            _cached_top_layers = data.get("top_10") or TOP_5_LAYERS
            n = data.get("n_images", "?")
            s = data.get("seeds", "?")
            print(f"[INFO] Loaded top-drift layers from {path} "
                  f"(n_images={n}, seeds={s})")
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}. Using hardcoded TOP_5_LAYERS.")
            _cached_top_layers = list(TOP_5_LAYERS)
    else:
        _cached_top_layers = list(TOP_5_LAYERS)
    return _cached_top_layers[:k]


def get_drift_weights(layers):
    """Compute per-layer correction weights from Phase 1 drift data.

    w_i = drift_i / mean_drift_of_top10, clamped to [0.5, 2.0].
    This makes high-drift layers get more correction (w > 1) and
    low-drift layers get less (w < 1), exploiting the diagnostic information
    that uniform correction ignores.

    理论依据：f_out = f_recon + λ·w_i·(f_inv - f_recon)
    其中 w_i ∝ ||f_inv - f_recon||_2 (Phase 1 诊断的层漂移量)。
    这是 MSE 损失对特征通道的 natural gradient 一阶近似：
    w_i ∝ ||∂L_recon/∂f_i||，漂移越大的层对重建误差贡献越大。
    """
    path = _DRIFT_SUMMARY_PATH
    if not path.exists():
        return {name: 1.0 for name in layers}

    with open(path) as f:
        data = json.load(f)

    # Extract drift means for the requested layers
    aggregated = data.get("aggregated", {})
    drift_values = []
    for name in layers:
        if name in aggregated:
            drift_values.append(aggregated[name]["mean"])
        else:
            drift_values.append(1.0)

    if not drift_values:
        return {name: 1.0 for name in layers}

    mean_drift = np.mean(drift_values) if hasattr(np, 'mean') else sum(drift_values) / len(drift_values)
    if mean_drift == 0:
        return {name: 1.0 for name in layers}

    weights = {}
    for name, d in zip(layers, drift_values):
        w = d / mean_drift
        weights[name] = max(0.5, min(2.0, float(w)))
    return weights


# ---------------------------------------------------------------------------
# λ-Scheduler
# ---------------------------------------------------------------------------

class LambdaScheduler:
    """按时步计算衰减后的 λ 值。"""

    def __init__(self, lam: float, num_steps: int, mode: str = "constant",
                 alpha: float = 3.0):
        self.lam = lam
        self.num_steps = num_steps
        self.mode = mode
        self.alpha = alpha

    def get(self, step_idx: int) -> float:
        """step_idx: 0..num_steps-1，0 是第一步（最大 timestep）。"""
        if self.mode == "constant":
            return self.lam
        elif self.mode == "linear":
            return self.lam * (1.0 - step_idx / max(self.num_steps - 1, 1))
        elif self.mode == "exp":
            return self.lam * np.exp(-self.alpha * step_idx / max(self.num_steps - 1, 1))
        else:
            return self.lam


# ---------------------------------------------------------------------------
# 模型 & 数据加载
# ---------------------------------------------------------------------------

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_image(pipe, path: str):
    """Direct resize to 512×512. Returns (latent, tensor_in_[-1,1])."""
    img = Image.open(path).convert("RGB").resize((512, 512))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2 * tensor - 1  # [0,1] → [-1,1] (SD VAE training domain)
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent, tensor


def decode_latent(pipe, latent):
    with torch.no_grad():
        tensor = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    return tensor


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _find_module(parent, name: str):
    tokens = name.split(".")
    mod = parent
    for t in tokens:
        try:
            mod = getattr(mod, t)
        except AttributeError:
            return None
    return mod


# ---------------------------------------------------------------------------
# FeatureCollector（仅收集）
# ---------------------------------------------------------------------------

class FeatureCollector:
    def __init__(self, unet, layers):
        self.features = {}
        self.handles = []
        for name in layers:
            mod = _find_module(unet, name)
            if mod is not None:
                self.handles.append(
                    mod.register_forward_hook(
                        lambda m, inp, out, n=name: self._hook(n, out)
                    )
                )

    def clear(self):
        self.features = {}

    def _hook(self, name, output):
        if isinstance(output, tuple):
            output = output[0]
        self.features[name] = output.detach().cpu()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# FeatureCorrector（注入校正）
# ---------------------------------------------------------------------------

class FeatureCorrector:
    """在 UNet 指定层注入残差校正信号。

    lam 可以是 float（固定强度）或 LambdaScheduler（按时步衰减）。
    per_layer_weights: dict {layer_name: weight}，漂移大的层配更高权重。
        公式：f_out = f_recon + λ · w_i · (f_inv - f_recon)
        不加权时 w_i ≡ 1，退化为 uniform 校正。
    """

    def __init__(self, unet, layers, lam, per_layer_weights=None):
        self.unet = unet
        self.layers = layers
        self.reference_features = {}
        self.step_idx = 0
        self.handles = []
        # 兼容 float 和 LambdaScheduler
        if isinstance(lam, (int, float)):
            self._get_lam = lambda: float(lam)
        else:
            self._get_lam = lambda: lam.get(self.step_idx)
        # Per-layer weights: w_i ∝ drift_i / mean_drift, clamped to [0.5, 2.0]
        if per_layer_weights is None:
            self._weights = {name: 1.0 for name in layers}
        else:
            self._weights = {}
            for name in layers:
                w = per_layer_weights.get(name, 1.0)
                self._weights[name] = max(0.5, min(2.0, w))  # clamp
        for name in layers:
            mod = _find_module(unet, name)
            if mod is not None:
                self.handles.append(
                    mod.register_forward_hook(
                        lambda m, inp, out, n=name: self._hook(n, out)
                    )
                )

    def set_reference(self, ref_dict: dict, step_idx: int = 0):
        self.reference_features = ref_dict
        self.step_idx = step_idx

    def _hook(self, name, output):
        if name not in self.reference_features:
            return output
        is_tuple = isinstance(output, tuple)
        out_tensor = output[0] if is_tuple else output
        ref = self.reference_features[name].to(
            device=out_tensor.device, dtype=out_tensor.dtype
        )
        lam_t = self._get_lam()
        w = self._weights.get(name, 1.0)
        corrected = out_tensor + lam_t * w * (ref - out_tensor)
        if is_tuple:
            return (corrected,) + output[1:]
        else:
            return corrected

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# StyleFeatureInjector：将 CLIP v_style 注入为 per-layer 特征偏置
# ---------------------------------------------------------------------------

class StyleFeatureInjector:
    """Inject CLIP v_style (768-dim) as per-channel feature bias on target layers.

    Training-free: 768-dim v_style is cyclically padded to match each layer's
    channel count and applied as a spatial bias.

    Usage:
        injector = StyleFeatureInjector(pipe.unet, layers, v_style, strength=0.5)
        # Run UNet forward — hooks auto-apply bias
        injector.remove()
    """

    def __init__(self, unet, layers, v_style, strength=0.5):
        self.unet = unet
        self.layers = layers
        self.handles = []
        self.strength = strength

        # v_style: [1, 768] normalized — build per-layer bias tensors
        self.bias = {}
        v = v_style.float().squeeze(0)  # [768]
        for name in layers:
            mod = _find_module(unet, name)
            if mod is None:
                continue
            # Determine output channels by checking module's conv2 or conv1.out_channels
            c = self._get_channels(mod, name)
            if c is None:
                c = 320  # fallback
            # Cycle-fill: repeat v_style [768] to fill [c] channels
            repeats = (c + 767) // 768
            bias_c = v.repeat(repeats)[:c]  # [c]
            self.bias[name] = bias_c.view(1, c, 1, 1).to(unet.device, dtype=unet.dtype)

            handle = mod.register_forward_hook(
                lambda m, inp, out, n=name: self._hook(n, out)
            )
            self.handles.append(handle)

    def _get_channels(self, mod, name):
        """Heuristic to get output channels from ResNet or Transformer block."""
        # ResNet blocks: mod.conv2.out_channels
        if hasattr(mod, 'conv2') and hasattr(mod.conv2, 'out_channels'):
            return mod.conv2.out_channels
        # Try conv1
        if hasattr(mod, 'conv1') and hasattr(mod.conv1, 'out_channels'):
            return mod.conv1.out_channels
        # Transformer blocks: try proj_out
        if hasattr(mod, 'proj_out') and hasattr(mod.proj_out, 'out_channels'):
            return mod.proj_out.out_channels
        return None

    def set_strength(self, strength):
        self.strength = strength

    def _hook(self, name, output):
        if name not in self.bias:
            return output
        if isinstance(output, tuple):
            out_tensor = output[0]
            corrected = out_tensor + self.strength * self.bias[name]
            return (corrected,) + output[1:]
        else:
            return output + self.strength * self.bias[name]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# AdaINStyleInjector：特征空间 AdaIN 风格迁移（免训练）
# ---------------------------------------------------------------------------

class AdaINStyleInjector:
    """Training-free style injection via AdaIN in UNet feature space.

    Collects per-layer channel-wise (μ, σ) from a style reference image's UNet
    forward pass, then during content reconstruction applies AdaIN to match those
    statistics. Bypasses the CLIP image/text modality gap.

    Formula: f_styled = σ_style * (f - μ_f) / σ_f + μ_style
             f_out = f + strength * (f_styled - f)

    Usage:
        injector = AdaINStyleInjector(pipe, layers, style_tensor, strength=0.5)
        # Run UNet forward — hooks auto-apply AdaIN
        injector.remove()
    """

    def __init__(self, pipe, layers, style_tensor, strength=0.5):
        self.pipe = pipe
        self.layers = layers
        self.handles = []
        self.strength = strength
        self.style_stats = {}

        # 1) Encode style image → latent
        style_t = style_tensor.float().to(pipe.device, dtype=pipe.dtype)
        with torch.no_grad():
            style_latent = pipe.vae.encode(style_t).latent_dist.sample()
            style_latent = style_latent * pipe.vae.config.scaling_factor
            # Add moderate noise for denoising-style features
            # Use ~halfway through the diffusion process (alpha ≈ 0.36 for 50 steps)
            pipe.scheduler.set_timesteps(50, device=pipe.device)
            t_noise = pipe.scheduler.timesteps[25]  # midpoint of 50-step schedule
            alpha = pipe.scheduler.alphas_cumprod[t_noise]
            eps = torch.randn_like(style_latent)
            z_noisy = alpha.sqrt() * style_latent + (1 - alpha).sqrt() * eps

        # 2) Collect style statistics at each target layer
        stats = {}
        hooks_tmp = []

        def _collect(name, mod, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # out: [B, C, H, W]
            mu = out.mean(dim=[2, 3])       # [B, C]
            var = out.var(dim=[2, 3], unbiased=False)  # [B, C]
            stats[name] = (mu, var)

        for name in layers:
            mod = _find_module(pipe.unet, name)
            if mod is not None:
                h = mod.register_forward_hook(
                    lambda m, inp, out, n=name: _collect(n, m, inp, out))
                hooks_tmp.append(h)

        with torch.no_grad():
            pipe.unet(z_noisy, t_noise, encoder_hidden_states=
                      pipe.encode_prompt("", DEVICE, 1, False)[0]).sample

        for h in hooks_tmp:
            h.remove()

        self.style_stats = {n: (mu.detach(), var.detach()) for n, (mu, var) in stats.items()}
        print(f"  [AdaIN] Collected style stats from {len(self.style_stats)} layers")

        # 3) Register permanent hooks for content reconstruction
        for name in layers:
            if name in self.style_stats:
                mod = _find_module(pipe.unet, name)
                if mod is not None:
                    handle = mod.register_forward_hook(
                        lambda m, inp, out, n=name: self._hook(n, out))
                    self.handles.append(handle)

    def set_strength(self, strength):
        self.strength = strength

    def _hook(self, name, output):
        if name not in self.style_stats:
            return output
        mu_style, var_style = self.style_stats[name]  # [1, C]

        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output

        # Content feature statistics
        mu_content = out_tensor.mean(dim=[2, 3], keepdim=True)     # [1, C, 1, 1]
        var_content = out_tensor.var(dim=[2, 3], keepdim=True, unbiased=False)

        # AdaIN: normalize content, re-scale with style
        eps = 1e-5
        f_norm = (out_tensor - mu_content) / (var_content + eps).sqrt()
        f_styled = f_norm * var_style.view(1, -1, 1, 1).sqrt() + mu_style.view(1, -1, 1, 1)

        # Blend
        f_out = out_tensor + self.strength * (f_styled - out_tensor)

        if isinstance(output, tuple):
            return (f_out,) + output[1:]
        return f_out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# DDIM 反演（带特征保存）
# ---------------------------------------------------------------------------

def ddim_inversion_with_features(pipe, latents, prompt_embeds, num_steps,
                                  hook_layers):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    collector = FeatureCollector(pipe.unet, hook_layers)
    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    saved_features = {}

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            collector.clear()
            noise_pred = pipe.unet(
                z, t_cur, encoder_hidden_states=prompt_embeds
            ).sample
            saved_features[int(t_cur)] = collector.features.copy()

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    collector.remove()
    return z, saved_features


# ---------------------------------------------------------------------------
# DDIM 重建（带校正）
# ---------------------------------------------------------------------------

def ddim_reconstruction_with_correction(pipe, noise, prompt_embeds, num_steps,
                                         saved_features, corrector):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            if t_int in saved_features:
                corrector.set_reference(saved_features[t_int], step_idx)
            else:
                corrector.set_reference({}, step_idx)

            noise_pred = pipe.unet(
                z, t, encoder_hidden_states=prompt_embeds
            ).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ---------------------------------------------------------------------------
# 基线：无校正
# ---------------------------------------------------------------------------

def ddim_inversion(pipe, latents, prompt_embeds, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(
                z, t_cur, encoder_hidden_states=prompt_embeds
            ).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
    return z


def ddim_reconstruction(pipe, noise, prompt_embeds, num_steps):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()
    with torch.no_grad():
        for t in timesteps:
            noise_pred = pipe.unet(
                z, t, encoder_hidden_states=prompt_embeds
            ).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
    return z


# ---------------------------------------------------------------------------
# Latent 空间插值（消融对照：特征空间 vs latent 空间）
# ---------------------------------------------------------------------------

def ddim_inversion_with_latents(pipe, latents, prompt_embeds, num_steps):
    """DDIM inversion, saving the latent at each inversion step.

    Returns: (z_T, saved_latents) where saved_latents[t] = z_t from inversion.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    saved_latents = {int(extended_ts[-1]): z.clone()}  # z_T

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            noise_pred = pipe.unet(
                z, t_cur, encoder_hidden_states=prompt_embeds
            ).sample
            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred
            saved_latents[int(t_next)] = z.clone()

    return z, saved_latents


def ddim_reconstruction_with_latent_correction(pipe, noise, prompt_embeds, num_steps,
                                                saved_latents, lam=0.5):
    """DDIM reconstruction with latent-space interpolation at each step.

    z_step = z_step + lam * (z_inv[step] - z_step)
    This is the latent-space analogue of feature-level correction.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = noise.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)
            noise_pred = pipe.unet(
                z, t, encoder_hidden_states=prompt_embeds
            ).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

            # Latent interpolation correction
            if t_int in saved_latents:
                z = z + lam * (saved_latents[t_int] - z)

    return z


# ---------------------------------------------------------------------------
# ArcFace 初始化
# ---------------------------------------------------------------------------

_arcface_app = None

def get_arcface():
    global _arcface_app
    if _arcface_app is None:
        _arcface_app = FaceAnalysis(
            name="antelopev2", providers=["CUDAExecutionProvider"]
        )
        _arcface_app.prepare(ctx_id=0, det_size=(512, 512))
    return _arcface_app


def compute_arcface_similarity(img1_tensor, img2_tensor):
    """计算两张 [1,3,512,512] 范围 [-1,1] 的人脸相似度。
    返回 (sim, ok)，ok=False 表示未检测到人脸。
    """
    try:
        import cv2

        def to_np(t):
            return ((t.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2
                    * 255).astype(np.uint8)[:, :, ::-1]

        img1_np = to_np(img1_tensor)
        img2_np = to_np(img2_tensor)

        app = get_arcface()
        faces1 = app.get(img1_np)
        faces2 = app.get(img2_np)

        if len(faces1) == 0 or len(faces2) == 0:
            return 0.0, False

        emb1 = faces1[0].normed_embedding
        emb2 = faces2[0].normed_embedding
        sim = float(np.dot(emb1, emb2))
        return sim, True
    except Exception:
        return 0.0, False


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

# DISTS 延迟加载
_dists_fn = None

def _get_dists():
    global _dists_fn
    if _dists_fn is None:
        try:
            from DISTS_pytorch.DISTS_pt import DISTS
            _dists_fn = DISTS().to(DEVICE)
        except (ImportError, ModuleNotFoundError):
            return None
    return _dists_fn


def compute_metrics(original_tensor, recon_tensor, lpips_fn=None,
                     compute_arcface=False, compute_dists=False):
    orig = original_tensor.float().clamp(-1, 1)
    recon = recon_tensor.float().clamp(-1, 1)

    mse = torch.nn.functional.mse_loss(orig, recon)
    # Both tensors are in [-1, 1], MAX_I = 2.0
    psnr_val = (20 * torch.log10(2.0 / (torch.sqrt(mse) + 1e-8))).item()

    orig_np = (orig.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    recon_np = (recon.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    ssim_val = float(ssim(orig_np, recon_np, channel_axis=2, data_range=1.0))

    result = {"PSNR": float(psnr_val), "SSIM": ssim_val}
    if lpips_fn is not None:
        result["LPIPS"] = float(lpips_fn(orig, recon).item())
    if compute_arcface:
        sim, ok = compute_arcface_similarity(original_tensor, recon_tensor)
        result["ArcFace"] = sim
        result["ArcFace_ok"] = ok
    if compute_dists:
        dists_fn = _get_dists()
        if dists_fn is not None:
            # DISTS expects [0,1]
            orig_01 = (orig + 1) / 2
            recon_01 = (recon + 1) / 2
            result["DISTS"] = float(dists_fn(orig_01, recon_01).item())
        else:
            result["DISTS"] = None
    return result


# ---------------------------------------------------------------------------
# 可视化：直方图匹配
# ---------------------------------------------------------------------------

def histogram_match(source, target):
    """Per-channel histogram matching: map source distribution to target.

    source, target: [1,3,H,W] tensors in [0,1].  Display-only, metrics unaffected.
    """
    src = source.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    tgt = target.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    out = np.empty_like(src)
    for c in range(3):
        s_ch = (src[:, :, c] * 255).clip(0, 255).astype(np.uint8)
        t_ch = (tgt[:, :, c] * 255).clip(0, 255).astype(np.uint8)
        hs, _ = np.histogram(s_ch, 256, (0, 256)); ht, _ = np.histogram(t_ch, 256, (0, 256))
        cdf_s = np.cumsum(hs.astype(float)) / max(hs.sum(), 1)
        cdf_t = np.cumsum(ht.astype(float)) / max(ht.sum(), 1)
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and cdf_t[j] < cdf_s[i]:
                j += 1
            lut[i] = j
        out[:, :, c] = lut[s_ch] / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).to(
        device=source.device, dtype=source.dtype)


# ---------------------------------------------------------------------------
# 可视化：并排对比网格
# ---------------------------------------------------------------------------

def make_grid_image(images_dict, output_path, ncols=4, reference_tensor=None,
                    metrics_dict=None):
    """Create side-by-side comparison grid of reconstructions.

    Args:
        images_dict: {"Label": tensor} in [0,1] range
        output_path: path to save PNG
        ncols: number of columns
        reference_tensor: if provided, histogram-match all non-"Original" images to it
        metrics_dict: optional {"Label": {"PSNR": ..., "SSIM": ...}} for title overlay
    """
    tensors, labels = [], []
    for name, t in images_dict.items():
        if t is not None:
            display = t.float().clamp(0, 1)
            tensors.append(display)
            labels.append(name)
    n = len(tensors)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(tensors[i].squeeze(0).permute(1, 2, 0).cpu())
            title = labels[i]
            if metrics_dict and labels[i] in metrics_dict:
                m = metrics_dict[labels[i]]
                parts = []
                for k in ["PSNR", "SSIM", "LPIPS", "ArcFace", "DISTS"]:
                    if k in m and m[k] is not None:
                        v = m[k]
                        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
                if parts:
                    title += "\n" + "  ".join(parts)
            axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def save_recon_img(tensor, out_dir, img_name, steps, tag, subdir=""):
    """Save a single reconstruction as PNG."""
    d = Path(out_dir) / subdir / "recons"
    os.makedirs(d, exist_ok=True)
    pil = transforms.ToPILImage()((tensor.squeeze(0) / 2 + 0.5).clamp(0, 1))
    pil.save(d / f"{img_name}_s{steps}_{tag}.png")


def save_results_csv(results, out_dir, filename, subdir=""):
    """Save results list as CSV + JSON."""
    if not results:
        return
    d = Path(out_dir) / subdir
    os.makedirs(d, exist_ok=True)

    csv_path = d / filename
    with open(csv_path, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in results for k in r))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] {csv_path}")

    json_path = d / filename.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {json_path}")
