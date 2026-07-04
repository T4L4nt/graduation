"""
CLIP utility module: CLIPFeatureExtractor and style-related helpers.

Extracted from phase3_common.py to avoid dependency on dead Phase 3 code.
Used by: phase4_scenes.py, phase4_p2p.py, phase2_nti.py,
         gen_thesis_figures.py, phase4_controlnet.py
"""

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLIP Feature Extraction
# ---------------------------------------------------------------------------

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
    """CLIP ViT-L/14 multimodal encoder: image → shared embedding space ← text.

    v_img  = CLIP.get_image_features(image)        [1, 768]  visual embedding
    v_text = CLIP.get_text_features(tokenize(text)) [1, 768]  text embedding

    Both live in the same 768-dim normalized multimodal space.
    """

    def __init__(self, device=None):
        from phase2_common import DEVICE
        self._device = device or DEVICE
        self.model, self.processor = _get_clip_model(self._device)
        self.hidden_dim = self.model.config.projection_dim  # 768

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """CLIP text encoding → 768-dim normalized multimodal embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs).pooler_output  # [1, 768]
        return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """CLIP visual encoding → 768-dim normalized multimodal embedding.

        image: PIL.Image or str (file path)
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
        img_np = (tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1) / 2
        img_np = (img_np.clip(0, 1) * 255).astype('uint8')
        pil = Image.fromarray(img_np)
        return self.encode_image(pil)

    def compute_content_projection(self, v: torch.Tensor, v_content: torch.Tensor) -> float:
        """Scalar projection |proj_{v_content}(v)| as content preservation score."""
        vc_norm = v_content / (v_content.norm(dim=-1, keepdim=True) + 1e-8)
        return float((v * vc_norm).sum())

    def clip_direction_similarity(self, edited_image, target_image_or_vector):
        """CLIP direction similarity: cos(CLIP(edited), CLIP(target)).

        Returns: (sim, v_edited, v_target)
        """
        v_edited = self.encode_image_from_tensor(edited_image) \
            if isinstance(edited_image, torch.Tensor) else self.encode_image(edited_image)

        if isinstance(target_image_or_vector, torch.Tensor) and target_image_or_vector.dim() == 2:
            v_target = target_image_or_vector
        elif isinstance(target_image_or_vector, torch.Tensor):
            v_target = self.encode_image_from_tensor(target_image_or_vector)
        else:
            v_target = self.encode_image(target_image_or_vector)

        sim = float((v_edited * v_target).sum())
        return sim, v_edited, v_target

    def compute_content_preservation(self, edited_tensor, original_v):
        """Content preservation: cos(CLIP(edited), v_original)."""
        v_edited = self.encode_image_from_tensor(edited_tensor)
        sim = float((v_edited * original_v).sum())
        return sim

    def compute_orthogonal_decomposition(self, v_img, v_content):
        """StyleTex: v_style = v_img - proj_{v_content}(v_img).

        Decomposes v_img into content-parallel (proj) and content-orthogonal
        (v_style) components. Verifies: cos(v_style, v_content) ≈ 0.
        """
        vc_norm = v_content.detach() / (v_content.detach().norm(dim=-1, keepdim=True) + 1e-8)
        proj = (v_img.detach() * vc_norm).sum(dim=-1, keepdim=True) * vc_norm
        v_style = v_img.detach() - proj
        cos = float((v_style * vc_norm).sum())
        return proj, v_style, cos


def find_closest_style(v_style, extractor):
    """Find closest style candidate to v_style (cosine similarity)."""
    best = ("", "", -1.0)
    for text, key in STYLE_CANDIDATES:
        emb = extractor.encode_text(text)
        sim = float((v_style * emb).sum())
        if sim > best[2]:
            best = (text, key, sim)
    return best


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
    from phase2_common import DEVICE
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
# Phase 3 style pipeline helpers (legacy, kept for gen_thesis_figures.py)
# =========================================================================

def make_styled_prompt_embedding(pipe, base_prompt, style_text, strength=0.5):
    """Linear interpolation of prompt embeddings: (1-λ)*base + λ*style."""
    from phase2_common import DEVICE
    base_emb = pipe.encode_prompt(base_prompt, DEVICE, 1, False)[0]
    style_emb = pipe.encode_prompt(style_text, DEVICE, 1, False)[0]
    return (1.0 - strength) * base_emb + strength * style_emb


def run_baseline(pipe, original_latent, original_tensor, prompt_embeds, num_steps,
                 lpips_fn=None, compute_arcface=False):
    """Baseline DDIM inversion → reconstruction."""
    import time
    from phase2_common import (
        ddim_inversion, ddim_reconstruction, decode_latent, compute_metrics,
    )
    t0 = time.perf_counter()
    noise = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)
    recon_latent = ddim_reconstruction(pipe, noise, prompt_embeds, num_steps)
    recon = decode_latent(pipe, recon_latent)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)
    return m, recon, elapsed


def run_correction_only(pipe, original_latent, original_tensor, prompt_embeds,
                         num_steps, lam, layers, lpips_fn=None, compute_arcface=False):
    """Phase 2 correction only."""
    import time
    from phase2_common import (
        ddim_inversion_with_features, ddim_reconstruction_with_correction,
        decode_latent, compute_metrics, FeatureCorrector, LambdaScheduler,
    )
    t0 = time.perf_counter()
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, layers)
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
    """Phase 2 correction + style injection (cross-attention and/or feature bias)."""
    import time
    from phase2_common import (
        DEVICE, ddim_inversion_with_features, decode_latent, compute_metrics,
        FeatureCorrector, LambdaScheduler,
    )
    t0 = time.perf_counter()
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, corr_layers)
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
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

    Returns: (metrics, recon_tensor, elapsed, pinning_log)
      pinning_log: list of (step_idx, content_proj, deviation, effective_strength)
    """
    import time
    from phase2_common import (
        DEVICE, ddim_inversion_with_features, decode_latent, compute_metrics,
        FeatureCorrector, LambdaScheduler,
    )
    t0 = time.perf_counter()

    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(original_tensor), v_content)

    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, corr_layers)
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    if "strength" in getattr(style_injector, "__dict__", {}):
        base_strength = style_injector.strength
    else:
        base_strength = 0.5

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
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

        if step_idx > 0 and pinning_freq > 0 and step_idx % pinning_freq == 0:
            with torch.no_grad():
                current_img = decode_latent(pipe, z.clone())
                v_current = extractor.encode_image_from_tensor(current_img)
                cur_proj = extractor.compute_content_projection(v_current, v_content)
                deviation = abs(cur_proj - ref_proj)
                pinning_log.append((step_idx, cur_proj, deviation, effective_strength))

                if deviation > pinning_threshold:
                    scale = max(0.0, 1.0 - pinning_strength * (deviation / max(ref_proj, 0.01)))
                    effective_strength = base_strength * scale
                    if style_injector is not None:
                        style_injector.set_strength(effective_strength)

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
    return m, recon, elapsed, pinning_log
