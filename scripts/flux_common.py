"""
FLUX.1-dev shared utilities: model loading, flow-matching inversion,
feature extraction hooks, and residual correction.

Architecture reference (FluxTransformer2DModel):
  - 19 double-stream (joint) blocks: process [text, image] tokens together
  - 38 single-stream blocks: process [image] tokens, text via modulation
  - Total: 57 transformer blocks
  - Each block outputs (encoder_hidden_states, hidden_states)

Key difference from DDIM: flow matching uses straight-line trajectories
  x_t = (1 - t) * x_0 + t * epsilon
  v_prediction: model predicts velocity v = d(x_t)/dt
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from diffusers import FluxPipeline
from PIL import Image
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase6_flux"
DATA_DIR = PROJECT_ROOT / "data" / "coco_val"

# FLUX block counts
N_JOINT_BLOCKS = 19
N_SINGLE_BLOCKS = 38
N_TOTAL_BLOCKS = N_JOINT_BLOCKS + N_SINGLE_BLOCKS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_flux_pipeline(device="cuda", dtype=torch.bfloat16, offload_t5=False):
    """Load FLUX.1-dev with memory optimizations.

    Args:
        offload_t5: if True, load T5-XXL on CPU (saves ~10GB VRAM)
    """
    import os

    # Use ModelScope cache if available, otherwise download from HuggingFace
    modelscope_path = os.path.join(
        os.path.expanduser("~"),
        ".cache/modelscope/models/AI-ModelScope--FLUX.1-dev/snapshots/master",
    )
    if os.path.isdir(modelscope_path):
        model_id = modelscope_path
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print(f"Loading FLUX.1-dev from ModelScope cache: {model_id}")
    else:
        model_id = "black-forest-labs/FLUX.1-dev"
        print("Loading FLUX.1-dev from HuggingFace...")

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=os.path.isdir(modelscope_path),
    )

    if offload_t5:
        # Keep T5 on CPU to save VRAM for feature extraction
        pipe.text_encoder_2.to("cpu")
        print("T5-XXL offloaded to CPU")

    pipe.to(device)
    pipe.vae.enable_tiling()

    print("FLUX pipeline loaded.")
    return pipe


# ---------------------------------------------------------------------------
# Feature extraction hooks
# ---------------------------------------------------------------------------

class FluxFeatureExtractor:
    """Hook-based feature extraction from FLUX transformer blocks.

    Captures (encoder_hidden_states, hidden_states) after each block.
    For single-stream blocks, encoder_hidden_states is unchanged but still captured.

    Usage:
        extractor = FluxFeatureExtractor(pipe.transformer)
        output = pipe.transformer(...)
        features = extractor.get_features()
        # features["joint_0"]["hidden"]  -> image token features
        # features["joint_0"]["encoder"] -> text token features
        # features["single_0"]["hidden"] -> image token features
        extractor.remove_hooks()
    """

    def __init__(self, transformer: nn.Module):
        self.transformer = transformer
        self.features: dict[str, dict[str, torch.Tensor]] = {}
        self._handles = []

    def _make_joint_hook(self, block_idx: int):
        def hook(module, args, kwargs, output):
            # output is (encoder_hidden_states, hidden_states)
            enc, hid = output
            self.features[f"joint_{block_idx}"] = {
                "encoder": enc.detach().cpu(),
                "hidden": hid.detach().cpu(),
            }

        return hook

    def _make_single_hook(self, block_idx: int):
        def hook(module, args, kwargs, output):
            enc, hid = output
            self.features[f"single_{block_idx}"] = {
                "encoder": enc.detach().cpu(),
                "hidden": hid.detach().cpu(),
            }

        return hook

    def register_hooks(self):
        """Register forward hooks on all 57 transformer blocks."""
        self.remove_hooks()

        for i, block in enumerate(self.transformer.transformer_blocks):
            handle = block.register_forward_hook(
                self._make_joint_hook(i), with_kwargs=True
            )
            self._handles.append(handle)

        for i, block in enumerate(self.transformer.single_transformer_blocks):
            handle = block.register_forward_hook(
                self._make_single_hook(i), with_kwargs=True
            )
            self._handles.append(handle)

    def get_features(self) -> dict:
        return dict(self.features)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()


# ---------------------------------------------------------------------------
# Flow matching inversion
# ---------------------------------------------------------------------------

def flux_invert(
    pipe: FluxPipeline,
    image: Image.Image,
    num_steps: int = 50,
    guidance_scale: float = 3.5,
    prompt: str = "",
    seed: int = 42,
    extract_features: bool = False,
) -> dict:
    """Flow-matching inversion: forward diffusion along straight-line trajectory.

    Flow matching forward process:
        x_t = (1 - t) * x_0 + t * epsilon

    For inversion, we encode the image with VAE, then simulate the forward
    process by running the ODE solver in reverse along the straight-line path.

    The key insight: since flow matching uses straight-line paths, "inversion"
    means running the ODE backward from t=0 to t=1, recording the velocity at
    each step. Reconstruction runs forward from t=1 to t=0 using the recorded
    velocities.

    Returns:
        dict with:
            - "latents_inv": inverted latents at t=1 (noise-like)
            - "latents_recon": reconstructed latents at t=0
            - "image_recon": PIL Image of reconstruction
            - "features_inv": features during inversion (if extract_features)
            - "features_recon": features during reconstruction (if extract_features)
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Encode image to latent space
    with torch.no_grad():
        image_tensor = pipe.image_processor.preprocess(
            image, height=image.height, width=image.width
        )
        image_tensor = image_tensor.to(pipe.device, dtype=pipe.vae.dtype)
        z_0_raw = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
        z_0_raw = (
            z_0_raw - pipe.vae.config.shift_factor
        ) * pipe.vae.config.scaling_factor

    # FLUX requires latent dimensions divisible by 2 for packing.
    batch_size, num_channels, latent_h, latent_w = z_0_raw.shape
    packed_h = 2 * (latent_h // 2)
    packed_w = 2 * (latent_w // 2)

    # Image dimensions (used by _unpack_latents)
    img_h = latent_h * pipe.vae_scale_factor
    img_w = latent_w * pipe.vae_scale_factor

    # Pad/crop latent to divisible dimensions
    z_0_raw = z_0_raw[:, :, :packed_h, :packed_w]

    # Pack latents into patch tokens for transformer
    z_0 = pipe._pack_latents(
        z_0_raw, batch_size, num_channels, packed_h, packed_w
    )

    # Prepare latent_image_ids (must match packed latent dimensions)
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size, packed_h // 2, packed_w // 2, pipe.device, pipe.transformer.dtype
    )

    # Prepare text embeddings
    if not prompt:
        prompt = ""
    with torch.no_grad():
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=pipe.device,
        )

    # Guidance tensor (required by FLUX transformer)
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([batch_size], guidance_scale, device=pipe.device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
    else:
        guidance = None

    # Flow matching timesteps
    # FLUX scheduler expects timesteps in [0, 1], transformer expects /1000
    dt = 1.0 / num_steps
    timesteps_t = np.linspace(0.0, 1.0, num_steps + 1)

    if extract_features:
        extractor = FluxFeatureExtractor(pipe.transformer)
        extractor.register_hooks()
        features_inv = {}
        features_recon = {}
    else:
        extractor = None
        features_inv = {}
        features_recon = {}

    # --- Forward inversion: z_0 -> z_1 along straight-line flow ---
    z_t = z_0.clone()
    latents_inv_path = [z_0_raw.cpu()]  # store raw latents for reference

    for i in tqdm(range(num_steps), desc="FLUX invert"):
        t_cur = timesteps_t[i]

        # Transformer expects timestep / 1000
        t_tensor = torch.full(
            (batch_size,), t_cur, device=pipe.device, dtype=pipe.transformer.dtype
        )

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_tensor,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=True,
            ).sample

        # Euler step forward along flow: z_{t+dt} = z_t + v(z_t, t) * dt
        z_t = z_t + noise_pred * dt

        # Only store features at the last step to save CPU RAM
        if extract_features and extractor is not None and i == num_steps - 1:
            features_inv["final"] = extractor.get_features()

    z_T = z_t.clone()
    latents_inv_path.append(
        pipe._unpack_latents(z_T.cpu(), img_h, img_w, pipe.vae_scale_factor)
    )

    # --- Reverse reconstruction: z_1 -> z_0 ---
    z_t = z_T.clone()

    for i in tqdm(range(num_steps), desc="FLUX recon"):
        t_cur = timesteps_t[num_steps - i]  # going backwards

        t_tensor = torch.full(
            (batch_size,), t_cur, device=pipe.device, dtype=pipe.transformer.dtype
        )

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_tensor,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=True,
            ).sample

        # Euler step backward along flow
        z_t = z_t - noise_pred * dt

        # Only store features at first recon step to save CPU RAM
        if extract_features and extractor is not None and i == 0:
            features_recon["first"] = extractor.get_features()

    # Unpack final latent and decode
    z_0_recon_packed = z_t
    z_0_recon_raw = pipe._unpack_latents(
        z_0_recon_packed, img_h, img_w, pipe.vae_scale_factor
    )

    z_0_recon_raw = z_0_recon_raw.to(pipe.device, dtype=pipe.vae.dtype)
    z_0_recon_raw = (
        z_0_recon_raw / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    )

    with torch.no_grad():
        image_recon = pipe.vae.decode(z_0_recon_raw, return_dict=False)[0]
        image_recon = pipe.image_processor.postprocess(image_recon, output_type="pil")[0]

    if extractor is not None:
        extractor.remove_hooks()

    return {
        "latents_inv": latents_inv_path,
        "z_0": z_0_raw.cpu(),        # raw (unpacked) initial latent
        "z_T": z_T.cpu(),            # packed terminal latent
        "z_0_recon_raw": z_0_recon_raw.cpu(),
        "image_recon": image_recon,
        "features_inv": features_inv,
        "features_recon": features_recon,
    }


# ---------------------------------------------------------------------------
# Drift computation
# ---------------------------------------------------------------------------

def compute_block_drift(features_inv: dict, features_recon: dict) -> dict:
    """Compute per-block feature drift between inversion and reconstruction.

    For each of the 57 blocks, computes:
      drift_hidden = ||f_inv_hidden - f_recon_hidden|| / ||f_inv_hidden||
      drift_encoder = ||f_inv_encoder - f_recon_encoder|| / ||f_inv_encoder||
                         (for joint blocks only; single blocks pass text through)

    Args:
        features_inv: features captured at the LAST step of inversion
        features_recon: features captured at the FIRST step of reconstruction
                        (both at approximately t=1)

    Returns:
        dict keyed by block name with drift statistics.
    """
    drift = {}

    # Compare at t=1 turnaround: last inversion step vs first reconstruction step
    if "final" not in features_inv or "first" not in features_recon:
        return drift

    finv = features_inv["final"]
    frecon = features_recon["first"]

    all_blocks = sorted(
        set(finv.keys()) & set(frecon.keys()),
        key=lambda x: (
            int(x.split("_")[0] == "single"),
            int(x.split("_")[1]),
        ),  # joint blocks first, then single
    )

    for block_name in all_blocks:
        feats_inv = finv[block_name]
        feats_recon = frecon[block_name]

        result = {}

        # Image token drift
        h_inv = feats_inv["hidden"].float()
        h_recon = feats_recon["hidden"].float()
        diff = h_inv - h_recon
        result["hidden_drift"] = float(
            torch.norm(diff).item() / (torch.norm(h_inv).item() + 1e-8)
        )

        # Text token drift (meaningful for joint blocks)
        e_inv = feats_inv["encoder"].float()
        e_recon = feats_recon["encoder"].float()
        diff_e = e_inv - e_recon
        result["encoder_drift"] = float(
            torch.norm(diff_e).item() / (torch.norm(e_inv).item() + 1e-8)
        )

        drift[block_name] = result

    return drift


# ---------------------------------------------------------------------------
# Residual correction in latent space
# ---------------------------------------------------------------------------

def apply_correction_latent(
    z_recon_raw: torch.Tensor,
    z_inv_raw: torch.Tensor,
    lam: float = 0.7,
) -> torch.Tensor:
    """Apply residual correction in raw (unpacked) latent space.

    z_corrected = z_recon_raw + lambda * (z_inv_raw - z_recon_raw)

    Both tensors must be in raw VAE latent format [B, 16, H, W].
    """
    return z_recon_raw + lam * (z_inv_raw.to(z_recon_raw.device) - z_recon_raw)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(original: Image.Image, reconstructed: Image.Image) -> dict:
    """Compute PSNR, SSIM, LPIPS between original and reconstructed images."""
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    import lpips

    # Resize if needed
    if original.size != reconstructed.size:
        reconstructed = reconstructed.resize(original.size)

    orig_np = np.array(original).astype(np.float32) / 255.0
    recon_np = np.array(reconstructed).astype(np.float32) / 255.0

    # PSNR
    psnr_val = psnr_fn(orig_np, recon_np, data_range=1.0)

    # SSIM (multichannel)
    ssim_val = ssim_fn(orig_np, recon_np, channel_axis=-1, data_range=1.0)

    # LPIPS
    lpips_fn = lpips.LPIPS(net="alex").to("cuda")
    orig_t = (
        torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).float().to("cuda") * 2
        - 1
    )
    recon_t = (
        torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).float().to("cuda") * 2
        - 1
    )
    lpips_val = float(lpips_fn(orig_t, recon_t).item())

    return {"PSNR": round(psnr_val, 3), "SSIM": round(ssim_val, 4), "LPIPS": round(lpips_val, 4)}


# ---------------------------------------------------------------------------
# Per-token text encoder drift analysis
# ---------------------------------------------------------------------------

def compute_per_token_drift(features_inv: dict, features_recon: dict) -> dict:
    """Per-token text encoder drift across joint blocks.

    For each joint block, computes drift per token position in the
    encoder_hidden_states sequence. Reveals which text positions
    (BOS, semantic tokens, padding) are most unstable.

    Args:
        features_inv/features_recon: Feature dicts from FluxFeatureExtractor.
            Each entry has "encoder": Tensor [1, N_tokens, D].

    Returns:
        Dict with:
          per_block: {block_name: [drift_per_position]}
          block_means: {block_name: mean_drift_across_positions}
          position_means: {pos_idx: mean_drift_across_blocks}
          overall_mean: float
    """
    per_block = {}
    position_sums = {}
    position_counts = {}

    if "final" not in features_inv or "first" not in features_recon:
        return {"per_block": {}, "block_means": {}, "position_means": {}, "overall_mean": 0.0}

    finv = features_inv["final"]
    frecon = features_recon["first"]

    for block_name in sorted(finv.keys()):
        if not block_name.startswith("joint"):
            continue

        e_inv = finv[block_name]["encoder"].float()    # [1, N_tokens, D]
        e_recon = frecon[block_name]["encoder"].float()  # [1, N_tokens, D]

        n_tokens = e_inv.shape[1]
        per_pos = []
        for t in range(n_tokens):
            diff = e_inv[0, t] - e_recon[0, t]
            drift_t = float(torch.norm(diff).item() / (torch.norm(e_inv[0, t]).item() + 1e-8))
            per_pos.append(drift_t)

            pos_idx = str(t)
            position_sums[pos_idx] = position_sums.get(pos_idx, 0.0) + drift_t
            position_counts[pos_idx] = position_counts.get(pos_idx, 0) + 1

        per_block[block_name] = per_pos

    block_means = {k: float(np.mean(v)) for k, v in per_block.items()}
    position_means = {k: position_sums[k] / position_counts[k] for k in position_sums}
    overall_mean = float(np.mean(list(block_means.values()))) if block_means else 0.0

    return {
        "per_block": per_block,
        "block_means": block_means,
        "position_means": position_means,
        "overall_mean": overall_mean,
    }


def compute_text_image_drift_correlation(per_block_drift: dict) -> float:
    """Pearson r between encoder_drift and hidden_drift across joint blocks.

    Args:
        per_block_drift: Output from compute_block_drift().

    Returns:
        Pearson correlation coefficient.
    """
    encoder_drifts = []
    hidden_drifts = []
    for name, d in per_block_drift.items():
        if name.startswith("joint"):
            encoder_drifts.append(d["encoder_drift"])
            hidden_drifts.append(d["hidden_drift"])

    if len(encoder_drifts) < 3:
        return 0.0

    encoder_drifts = np.array(encoder_drifts)
    hidden_drifts = np.array(hidden_drifts)
    return float(np.corrcoef(encoder_drifts, hidden_drifts)[0, 1])


# ---------------------------------------------------------------------------
# Feature-level residual correction via hooks
# ---------------------------------------------------------------------------

class FluxFeatureCorrector:
    """Hook-based feature-level residual correction for FLUX transformer blocks.

    Injects residuals f_out = f_recon + lambda * (f_inv - f_recon) during
    reconstruction, at the block level. Supports both image token (hidden)
    and text token (encoder) correction independently.

    Usage:
        # 1. Run inversion to capture reference features
        out_inv = flux_invert(pipe, img, extract_features=True)

        # 2. Set up corrector with captured features
        corrector = FluxFeatureCorrector(pipe.transformer, lam_hidden=0.7)
        corrector.set_reference(out_inv["features_inv"])

        # 3. Run reconstruction with correction active
        #    (reconstruction loop uses flux_invert with corrector arg)
    """

    def __init__(
        self,
        transformer: nn.Module,
        lam_hidden: float = 0.7,
        lam_encoder: float = 0.0,
        joint_indices: list = None,
        single_indices: list = None,
    ):
        """
        Args:
            transformer: FluxTransformer2DModel instance.
            lam_hidden: Correction strength for image tokens.
            lam_encoder: Correction strength for text tokens (joint blocks only).
            joint_indices: Which joint blocks to correct (default: all 19).
            single_indices: Which single blocks to correct (default: all 38).
        """
        self.transformer = transformer
        self.lam_hidden = lam_hidden
        self.lam_encoder = lam_encoder
        self.joint_indices = joint_indices or list(range(N_JOINT_BLOCKS))
        self.single_indices = single_indices or list(range(N_SINGLE_BLOCKS))
        self.reference: dict[str, dict[str, torch.Tensor]] = {}
        self._handles = []

    def set_reference(self, inv_features: dict):
        """Set inversion features as correction targets.

        Args:
            inv_features: Features from inversion pass, in FluxFeatureExtractor
                          format. Expected: {"final": {"joint_0": {...}, ...}}
                          or flat format: {"joint_0": {...}, ...}.
        """
        # Handle nested (FluxFeatureExtractor) format: {"final": {...}}
        if "final" in inv_features:
            inv_features = inv_features["final"]

        self.reference = {}
        for name, feats in inv_features.items():
            self.reference[name] = {
                "encoder": feats["encoder"].clone(),
                "hidden": feats["hidden"].clone(),
            }

    def register_hooks(self):
        """Register forward hooks on all specified blocks."""
        self.remove_hooks()

        for i in self.joint_indices:
            block = self.transformer.transformer_blocks[i]
            handle = block.register_forward_hook(
                self._make_joint_hook(i), with_kwargs=True
            )
            self._handles.append(handle)

        for i in self.single_indices:
            block = self.transformer.single_transformer_blocks[i]
            handle = block.register_forward_hook(
                self._make_single_hook(i), with_kwargs=True
            )
            self._handles.append(handle)

    def _make_joint_hook(self, idx: int):
        name = f"joint_{idx}"

        def hook(module, args, kwargs, output):
            enc, hid = output
            if name not in self.reference:
                return output
            ref = self.reference[name]

            if self.lam_hidden > 0:
                ref_hid = ref["hidden"].to(hid.device, dtype=hid.dtype)
                hid = hid + self.lam_hidden * (ref_hid - hid)

            if self.lam_encoder > 0:
                ref_enc = ref["encoder"].to(enc.device, dtype=enc.dtype)
                enc = enc + self.lam_encoder * (ref_enc - enc)

            return (enc, hid)

        return hook

    def _make_single_hook(self, idx: int):
        name = f"single_{idx}"

        def hook(module, args, kwargs, output):
            enc, hid = output
            if name not in self.reference or self.lam_hidden <= 0:
                return output
            ref_hid = self.reference[name]["hidden"].to(hid.device, dtype=hid.dtype)
            hid = hid + self.lam_hidden * (ref_hid - hid)
            return (enc, hid)

        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Reconstruction with feature-level correction
# ---------------------------------------------------------------------------

def run_correction_feature(
    pipe,
    image: Image.Image,
    num_steps: int = 50,
    prompt: str = "",
    guidance_scale: float = 3.5,
    seed: int = 42,
    lam_hidden: float = 0.7,
    lam_encoder: float = 0.0,
    joint_indices: list = None,
    single_indices: list = None,
) -> dict:
    """Run inversion + feature-level corrected reconstruction.

    1. Inversion captures per-block features at turnaround (t=1).
    2. FluxFeatureCorrector registers hooks on specified blocks.
    3. Reconstruction applies correction at each block output.

    Args:
        pipe: FluxPipeline instance.
        image: Input PIL image.
        num_steps: Number of flow matching steps.
        prompt: Text prompt (empty string = unconditional-like).
        guidance_scale: CFG scale.
        seed: Random seed.
        lam_hidden: Correction strength for image tokens.
        lam_encoder: Correction strength for text tokens (joint only).
        joint_indices: Which joint blocks to correct (None = all).
        single_indices: Which single blocks to correct (None = all).

    Returns:
        dict with image_recon, image_corrected, metrics keys.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # Encode image
    with torch.no_grad():
        image_tensor = pipe.image_processor.preprocess(
            image, height=image.height, width=image.width
        )
        image_tensor = image_tensor.to(pipe.device, dtype=pipe.vae.dtype)
        z_0_raw = pipe.vae.encode(image_tensor).latent_dist.sample(generator)
        z_0_raw = (
            z_0_raw - pipe.vae.config.shift_factor
        ) * pipe.vae.config.scaling_factor

    batch_size, num_channels, latent_h, latent_w = z_0_raw.shape
    packed_h = 2 * (latent_h // 2)
    packed_w = 2 * (latent_w // 2)
    img_h = latent_h * pipe.vae_scale_factor
    img_w = latent_w * pipe.vae_scale_factor
    z_0_raw = z_0_raw[:, :, :packed_h, :packed_w]
    z_0 = pipe._pack_latents(z_0_raw, batch_size, num_channels, packed_h, packed_w)

    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size, packed_h // 2, packed_w // 2, pipe.device, pipe.transformer.dtype
    )

    # Text embeddings
    if not prompt:
        prompt = ""
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=pipe.device,
        )

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([batch_size], guidance_scale, device=pipe.device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
    else:
        guidance = None

    dt = 1.0 / num_steps
    timesteps_t = np.linspace(0.0, 1.0, num_steps + 1)

    # --- Inversion pass (with feature extraction) ---
    extractor = FluxFeatureExtractor(pipe.transformer)
    extractor.register_hooks()
    z_t = z_0.clone()

    for i in tqdm(range(num_steps), desc="Invert"):
        t_cur = timesteps_t[i]
        t_tensor = torch.full(
            (batch_size,), t_cur, device=pipe.device, dtype=pipe.transformer.dtype
        )
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_tensor,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=True,
            ).sample
        z_t = z_t + noise_pred * dt

        if i == num_steps - 1:
            features_inv = extractor.get_features()

    z_T = z_t.clone()
    extractor.remove_hooks()

    # --- Reconstruction pass (with feature correction) ---
    corrector = FluxFeatureCorrector(
        pipe.transformer,
        lam_hidden=lam_hidden,
        lam_encoder=lam_encoder,
        joint_indices=joint_indices,
        single_indices=single_indices,
    )
    corrector.set_reference(features_inv)
    corrector.register_hooks()

    z_t = z_T.clone()

    for i in tqdm(range(num_steps), desc="Recon+Corr"):
        t_cur = timesteps_t[num_steps - i]
        t_tensor = torch.full(
            (batch_size,), t_cur, device=pipe.device, dtype=pipe.transformer.dtype
        )
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_tensor,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=True,
            ).sample
        z_t = z_t - noise_pred * dt

    corrector.remove_hooks()

    # Decode corrected reconstruction
    z_0_recon = pipe._unpack_latents(z_t, img_h, img_w, pipe.vae_scale_factor)
    z_0_recon = z_0_recon.to(pipe.device, dtype=pipe.vae.dtype)
    z_0_recon = (
        z_0_recon / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    )

    with torch.no_grad():
        img_corr = pipe.vae.decode(z_0_recon, return_dict=False)[0]
        img_corr = pipe.image_processor.postprocess(img_corr, output_type="pil")[0]

    # Baseline reconstruction (no correction) for comparison
    z_t = z_T.clone()
    for i in tqdm(range(num_steps), desc="Recon (baseline)"):
        t_cur = timesteps_t[num_steps - i]
        t_tensor = torch.full(
            (batch_size,), t_cur, device=pipe.device, dtype=pipe.transformer.dtype
        )
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=z_t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t_tensor,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=True,
            ).sample
        z_t = z_t - noise_pred * dt

    z_0_recon_bl = pipe._unpack_latents(z_t, img_h, img_w, pipe.vae_scale_factor)
    z_0_recon_bl = z_0_recon_bl.to(pipe.device, dtype=pipe.vae.dtype)
    z_0_recon_bl = (
        z_0_recon_bl / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    )

    with torch.no_grad():
        img_baseline = pipe.vae.decode(z_0_recon_bl, return_dict=False)[0]
        img_baseline = pipe.image_processor.postprocess(img_baseline, output_type="pil")[0]

    # Metrics
    m_baseline = compute_metrics(image, img_baseline)
    m_corr = compute_metrics(image, img_corr)

    return {
        "image_recon": img_baseline,
        "image_corrected": img_corr,
        "baseline_PSNR": m_baseline["PSNR"],
        "baseline_LPIPS": m_baseline["LPIPS"],
        "baseline_SSIM": m_baseline["SSIM"],
        "corr_PSNR": m_corr["PSNR"],
        "corr_LPIPS": m_corr["LPIPS"],
        "corr_SSIM": m_corr["SSIM"],
        "delta_PSNR": round(m_corr["PSNR"] - m_baseline["PSNR"], 3),
    }


# ---------------------------------------------------------------------------
# Batch diagnosis
# ---------------------------------------------------------------------------

def run_diagnosis(
    pipe: FluxPipeline,
    image_paths: list[Path],
    num_steps: int = 50,
    lam: float = 0.7,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run full diagnosis pipeline on a set of images.

    1. Flow matching inversion + reconstruction (no correction)
    2. Flow matching inversion + reconstruction (with latent correction)
    3. Per-block feature drift measurement (on 1 image for efficiency)
    4. Compute metrics for all images

    Returns summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"images": [], "drift": None, "summary": {}}

    # Run first image with full feature extraction for drift analysis
    print(f"\n=== Drift diagnosis on {image_paths[0].name} ===")
    img = Image.open(image_paths[0]).convert("RGB")

    out = flux_invert(
        pipe, img, num_steps=num_steps, extract_features=True
    )
    drift = compute_block_drift(out["features_inv"], out["features_recon"])
    results["drift"] = drift

    print(f"\nTop-10 drift blocks:")
    sorted_drift = sorted(
        drift.items(), key=lambda x: x[1]["hidden_drift"], reverse=True
    )
    for name, d in sorted_drift[:10]:
        hdrift = d["hidden_drift"]
        edrift = d.get("encoder_drift", 0)
        print(f"  {name:20s}: hidden={hdrift:.6f}, encoder={edrift:.6f}")

    # Run all images for metrics
    metrics_no_corr = []
    metrics_corr = []

    for img_path in tqdm(image_paths, desc="Evaluating images"):
        img = Image.open(img_path).convert("RGB")

        # Without correction
        out = flux_invert(pipe, img, num_steps=num_steps)
        m = compute_metrics(img, out["image_recon"])
        m["image"] = img_path.name
        metrics_no_corr.append(m)

        # With latent correction
        z_corrected = apply_correction_latent(out["z_0"], out["latents_inv"][0], lam=lam)
        z_corrected = z_corrected.to(pipe.device, dtype=pipe.vae.dtype)
        z_corrected = z_corrected / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor

        with torch.no_grad():
            img_corr = pipe.vae.decode(z_corrected, return_dict=False)[0]
            img_corr = pipe.image_processor.postprocess(img_corr, output_type="pil")[0]

        m_corr = compute_metrics(img, img_corr)
        m_corr["image"] = img_path.name
        metrics_corr.append(m_corr)

    # Summarize
    for key in ["PSNR", "SSIM", "LPIPS"]:
        vals = [m[key] for m in metrics_no_corr]
        results["summary"][f"{key}_nocorr_mean"] = round(np.mean(vals), 3)
        results["summary"][f"{key}_nocorr_std"] = round(np.std(vals), 3)

        vals_c = [m[key] for m in metrics_corr]
        results["summary"][f"{key}_corr_mean"] = round(np.mean(vals_c), 3)
        results["summary"][f"{key}_corr_std"] = round(np.std(vals_c), 3)

    delta_psnr = results["summary"]["PSNR_corr_mean"] - results["summary"]["PSNR_nocorr_mean"]
    results["summary"]["delta_PSNR"] = round(delta_psnr, 3)

    print(f"\n=== Summary (n={len(image_paths)}, {num_steps} steps, λ={lam}) ===")
    print(f"PSNR: {results['summary']['PSNR_nocorr_mean']} -> {results['summary']['PSNR_corr_mean']} "
          f"(Δ={delta_psnr:+.2f})")
    print(f"LPIPS: {results['summary']['LPIPS_nocorr_mean']} -> {results['summary']['LPIPS_corr_mean']}")

    # Save results
    drift_json = {}
    for name, d in drift.items():
        drift_json[name] = {"hidden_drift": d["hidden_drift"], "encoder_drift": d["encoder_drift"]}

    summary_path = output_dir / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"drift": drift_json, "metrics_no_corr": metrics_no_corr,
                    "metrics_corr": metrics_corr, "summary": results["summary"]},
                  f, indent=2)
    print(f"Saved: {summary_path}")

    return results


if __name__ == "__main__":
    print("FLUX common utilities loaded.")
    print(f"  Blocks: {N_JOINT_BLOCKS} joint + {N_SINGLE_BLOCKS} single = {N_TOTAL_BLOCKS}")
    print(f"  Output dir: {OUTPUT_DIR}")
