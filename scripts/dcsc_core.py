"""
DCSC: Drift-Aware Closed-Loop Style Controller.

Core components:
  - CorrectableSubspace: incremental Gram-Schmidt orthonormal basis in CLIP space
  - DCSCStyleController: proportional feedback control law + subspace projection
  - dcsc_controlled_generation(): main generation loop
  - resolve_style_direction(): compute style vector from text or reference image

Style definition: externally driven — from a user-specified text prompt
(e.g. "oil painting") or style reference image. This replaces the earlier
self-decomposition v_style = v_orig - proj(v_orig) which was circular.

Usage:
  from dcsc_core import DCSCStyleController, dcsc_controlled_generation
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from phase2_common import (
    DEVICE, DTYPE, decode_latent, compute_metrics,
    ddim_inversion, ddim_inversion_with_features, ddim_reconstruction,
    ddim_reconstruction_with_correction,
    FeatureCorrector, LambdaScheduler, StyleFeatureInjector,
    get_top_drift_layers, save_recon_img,
)

# ---------------------------------------------------------------------------
# Style direction resolution
# ---------------------------------------------------------------------------

def resolve_style_direction(
    extractor,
    v_content: torch.Tensor,
    style_text: Optional[str] = None,
    style_ref_image: Optional[str] = None,
) -> Tuple[torch.Tensor, str]:
    """Compute CLIP style direction from external target.

    Text-driven (recommended):
        v_style_text = CLIP(style_text)
        v_style = v_style_text - proj_{v_content}(v_style_text)
    This is StyleTex applied to the TARGET text, not the content image.

    Reference-image-driven:
        v_style_img = CLIP(style_ref_image)
        v_style = v_style_img - proj_{v_content}(v_style_img)

    Returns: (v_style [1, 768], style_label) where style_label is a human-readable tag.
    """
    if style_text is not None:
        v_raw = extractor.encode_text(style_text)
        label = style_text
    elif style_ref_image is not None:
        from PIL import Image
        v_raw = extractor.encode_image(style_ref_image)
        label = f"ref:{Path(style_ref_image).stem}"
    else:
        raise ValueError("Either style_text or style_ref_image must be provided")

    # StyleTex: decompose the TARGET (not the content image)
    _, v_style, cos = extractor.compute_orthogonal_decomposition(v_raw, v_content)
    print(f"  [Style] target='{label}'  cos(v_style, v_content)={cos:.6f}")
    return v_style, label


# ---------------------------------------------------------------------------
# CorrectableSubspace
# ---------------------------------------------------------------------------

class CorrectableSubspace:
    """Incrementally-built orthonormal basis in CLIP embedding space.

    At each control step, computes v_residual = v_current - v_original
    and adds the component orthogonal to the existing basis (Gram-Schmidt).
    The orthogonal complement defines "content-safe style directions" —
    directions the correction mechanism can recover.

    Basis grows monotonically, saturates at 3-5 vectors in practice.
    Empty basis → falls back to standard CLIP orthogonal projection.
    """

    def __init__(self, dim: int = 768, energy_threshold: float = 1e-4,
                 max_basis: int = 10):
        self.basis: List[torch.Tensor] = []
        self.dim = dim
        self.energy_threshold = energy_threshold
        self.max_basis = max_basis

    @torch.no_grad()
    def update(self, v_residual: torch.Tensor) -> bool:
        if self._size() >= self.max_basis:
            return False
        v = v_residual.detach().float()
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        for u in self.basis:
            coeff = (v * u).sum(dim=-1, keepdim=True)
            v = v - coeff * u
        residual_norm = v.norm(dim=-1).item()
        if residual_norm < self.energy_threshold:
            return False
        v = v / residual_norm
        self.basis.append(v)
        return True

    @torch.no_grad()
    def project_orthogonal(self, v: torch.Tensor) -> torch.Tensor:
        if self._size() == 0:
            return v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        v_float = v.detach().float()
        result = v_float.clone()
        for u in self.basis:
            coeff = (result * u).sum(dim=-1, keepdim=True)
            result = result - coeff * u
        return result / (result.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def project_into(self, v: torch.Tensor) -> torch.Tensor:
        if self._size() == 0:
            return torch.zeros_like(v)
        v_float = v.detach().float()
        result = torch.zeros_like(v_float)
        for u in self.basis:
            coeff = (v_float * u).sum(dim=-1, keepdim=True)
            result = result + coeff * u
        return result

    @torch.no_grad()
    def compute_energy_fraction(self, v: torch.Tensor) -> float:
        if self._size() == 0:
            return 0.0
        proj_in = self.project_into(v)
        return float((proj_in.norm(dim=-1) ** 2) / (v.norm(dim=-1) ** 2 + 1e-8))

    def _size(self) -> int:
        return len(self.basis)

    def reset(self):
        self.basis.clear()


# ---------------------------------------------------------------------------
# DCSCStyleController
# ---------------------------------------------------------------------------

class DCSCStyleController:
    """Drift-Aware Closed-Loop Style Controller.

    Control law (proportional):
        lambda_style(t) = lambda_0 * max(0, 1 - Kp * d_content(t))

    where d_content(t) = |proj(v_current(t), v_content) - ref_proj|
    is the CLIP-space content projection deviation from the reference.

    Subspace projection:
        v_style_used(t) = subspace.project_orthogonal(v_style_base)

    Empirical stability: under moderate Kp (<= 2.0), content drift is
    observed to remain bounded. At high Kp (>= 5.0), oscillations may
    occur — boundedness is not guaranteed for all parameter regimes.
    """

    def __init__(
        self,
        v_style_base: torch.Tensor,
        v_content: torch.Tensor,
        lambda_0: float = 0.5,
        Kp: float = 1.0,
        dim: int = 768,
        monotonic_lambda: bool = True,
        ema_alpha: float = 0.3,
    ):
        self.subspace = CorrectableSubspace(dim=dim)
        self.v_style_base = v_style_base / (v_style_base.norm(dim=-1, keepdim=True) + 1e-8)
        self.v_content = v_content / (v_content.norm(dim=-1, keepdim=True) + 1e-8)
        self.lambda_0 = lambda_0
        self.Kp = Kp
        self.monotonic_lambda = monotonic_lambda
        self.ema_alpha = ema_alpha
        self.ref_proj: Optional[float] = None
        self.v_orig: Optional[torch.Tensor] = None
        self.prev_lambda: Optional[float] = None
        self.ema_d_content: Optional[float] = None
        self.log: List[Dict] = []

    @torch.no_grad()
    def initialize(self, original_tensor: torch.Tensor, extractor) -> None:
        v_orig = extractor.encode_image_from_tensor(original_tensor)
        self.ref_proj = extractor.compute_content_projection(v_orig, self.v_content)
        self.v_orig = v_orig

    @torch.no_grad()
    def compute_control(
        self, current_tensor: torch.Tensor, extractor,
    ) -> Tuple[float, torch.Tensor, Dict]:
        if self.ref_proj is None:
            raise RuntimeError("Must call initialize() before compute_control()")

        v_current = extractor.encode_image_from_tensor(current_tensor)
        cur_proj = extractor.compute_content_projection(v_current, self.v_content)
        d_content = abs(cur_proj - self.ref_proj)

        v_residual = v_current - self.v_orig
        self.subspace.update(v_residual)

        if self.ema_d_content is None:
            self.ema_d_content = d_content
        else:
            self.ema_d_content = (self.ema_alpha * d_content
                                  + (1 - self.ema_alpha) * self.ema_d_content)
        d_smooth = self.ema_d_content

        lambda_style = self.lambda_0 * max(0.0, 1.0 - self.Kp * d_smooth)
        if self.monotonic_lambda and self.prev_lambda is not None:
            lambda_style = min(lambda_style, self.prev_lambda)
        self.prev_lambda = lambda_style

        v_style_proj = self.subspace.project_orthogonal(self.v_style_base)
        proj_energy = self.subspace.compute_energy_fraction(self.v_style_base)

        info = {
            "d_content_raw": d_content, "d_content_smooth": d_smooth,
            "cur_proj": cur_proj, "ref_proj": self.ref_proj,
            "lambda_style": lambda_style, "basis_size": self.subspace._size(),
            "proj_energy": proj_energy,
        }
        self.log.append(info)
        return lambda_style, v_style_proj, info

    def empirical_stability_ratio(self) -> float:
        """Observed max d_content / (lambda_0 / Kp). < 2.0 indicates stable regime."""
        if len(self.log) == 0:
            return float("inf")
        max_d = max(e["d_content_raw"] for e in self.log)
        denom = self.lambda_0 / max(self.Kp, 1e-8)
        return float("inf") if denom == 0 else max_d / denom

    def reset(self):
        self.subspace.reset()
        self.ref_proj = None; self.v_orig = None
        self.prev_lambda = None; self.ema_d_content = None
        self.log.clear()


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def dcsc_controlled_generation(
    pipe,
    original_latent: torch.Tensor,
    original_tensor: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int,
    corr_lam: float,
    corr_layers: List[str],
    extractor,                               # CLIPFeatureExtractor
    v_content: torch.Tensor,                 # [1, 768] content direction
    # Style specification (EXTERNAL — one of these required):
    style_text: Optional[str] = None,
    style_ref_image: Optional[str] = None,
    # DCSC control params:
    lambda_0: float = 0.5,
    Kp: float = 1.0,
    control_freq: int = 5,
    style_mode: str = "extra_token",
    lpips_fn=None,
    compute_arcface: bool = False,
    monotonic_lambda: bool = True,
    ema_alpha: float = 0.3,
    max_control_steps: Optional[int] = None,
    use_subspace: bool = True,
    **kwargs,
) -> Tuple[Dict, torch.Tensor, float, Dict]:
    """DCSC-controlled generation with external style target.

    Args:
        style_text: style prompt e.g. "oil painting" or "a watercolor sketch"
        style_ref_image: path to style reference image (alternative to text)
            At least one of style_text / style_ref_image must be provided.
        max_control_steps: stop controller updates after N calls (None = all steps)
        use_subspace: if False, skip subspace projection (ablation)

    Returns:
        metrics: dict with PSNR, SSIM, LPIPS, CLIP_style, CLIP_content, ArcFace
        recon: [1, 3, 512, 512] tensor
        elapsed: float seconds
        control_trajectory: dict with trajectory logs + style_label
    """
    from phase3_prep import build_style_cross_attn_tokens

    # ---- Resolve style direction from external target ----
    v_style, style_label = resolve_style_direction(
        extractor, v_content,
        style_text=style_text, style_ref_image=style_ref_image,
    )

    t0 = time.perf_counter()

    # ---- Controller ----
    controller = DCSCStyleController(
        v_style_base=v_style, v_content=v_content,
        lambda_0=lambda_0, Kp=Kp,
        monotonic_lambda=monotonic_lambda, ema_alpha=ema_alpha,
    )
    controller.initialize(original_tensor, extractor)
    print(f"  [DCSC] style='{style_label}' λ_0={lambda_0:.2f} Kp={Kp:.1f}  "
          f"ref_proj={controller.ref_proj:.4f}")

    # ---- DDIM inversion with feature collection ----
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, corr_layers)

    # ---- FeatureCorrector ----
    sched = LambdaScheduler(corr_lam, num_steps, "constant")
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    # ---- Style injector (feature bias mode) ----
    style_injector = None
    if style_mode == "feature_bias":
        style_injector = StyleFeatureInjector(pipe.unet, corr_layers, v_style, strength=lambda_0)

    # ---- Initial cross-attn tokens ----
    if style_mode in ("extra_token", "interpolate"):
        styled_prompt_embeds = build_style_cross_attn_tokens(
            pipe, "", v_style, strength=lambda_0, mode=style_mode)
    else:
        styled_prompt_embeds = prompt_embeds

    # ---- Reconstruction loop ----
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()

    control_calls = 0
    trajectory = {
        "steps": [], "lambda": [], "d_content_raw": [], "d_content_smooth": [],
        "basis_size": [], "proj_energy": [], "cur_proj": [],
    }

    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        if t_int in saved:
            corrector.set_reference(saved[t_int], step_idx)
        else:
            corrector.set_reference({}, step_idx)

        # ---- DCSC control step ----
        do_control = (
            step_idx > 0 and control_freq > 0
            and step_idx % control_freq == 0
            and (max_control_steps is None or control_calls < max_control_steps)
        )
        if do_control:
            with torch.no_grad():
                current_img = decode_latent(pipe, z.clone())
                lambda_eff, v_style_proj, info = controller.compute_control(
                    current_img, extractor)

            control_calls += 1
            trajectory["steps"].append(step_idx)
            trajectory["lambda"].append(lambda_eff)
            trajectory["d_content_raw"].append(info["d_content_raw"])
            trajectory["d_content_smooth"].append(info["d_content_smooth"])
            trajectory["basis_size"].append(info["basis_size"])
            trajectory["proj_energy"].append(info["proj_energy"])
            trajectory["cur_proj"].append(info["cur_proj"])

            v_style_actual = v_style_proj if use_subspace else v_style
            if style_mode in ("extra_token", "interpolate"):
                styled_prompt_embeds = build_style_cross_attn_tokens(
                    pipe, "", v_style_actual, strength=lambda_eff, mode=style_mode)
            if style_injector is not None:
                style_injector.set_strength(lambda_eff)

            if control_calls <= 3:
                print(f"  [DCSC] step={step_idx:3d}  d_content={info['d_content_raw']:.4f}  "
                      f"λ={lambda_eff:.3f}  basis={info['basis_size']}  "
                      f"proj_e={info['proj_energy']:.3f}")

        # ---- UNet forward + scheduler step ----
        inp = scheduler.scale_model_input(z, t)
        with torch.no_grad():
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=styled_prompt_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    # ---- Cleanup ----
    corrector.remove()
    if style_injector is not None:
        style_injector.remove()

    # ---- Decode & metrics ----
    recon = decode_latent(pipe, z)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)

    # ---- CLIP metrics ----
    v_orig = extractor.encode_image_from_tensor(original_tensor)
    v_recon = extractor.encode_image_from_tensor(recon)
    m["CLIP_content"] = float((v_recon * v_orig).sum())
    # Style similarity to the TARGET style text/image, not to self-decomposition
    m["CLIP_style"] = float((v_recon * v_style).sum())
    m["method"] = f"DCSC_Kp{Kp:.1f}_lam{lambda_0:.2f}"

    # ---- Trajectory summary ----
    traj_summary = {
        "style_label": style_label,
        "trajectory": trajectory,
        "controller_log": [{k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in entry.items()}
                           for entry in controller.log],
        "empirical_stability_ratio": controller.empirical_stability_ratio(),
        "final_lambda": controller.prev_lambda if controller.prev_lambda is not None else lambda_0,
        "n_control_calls": control_calls,
        "final_basis_size": controller.subspace._size(),
        "lambda_0": lambda_0, "Kp": Kp, "corr_lam": corr_lam,
        "control_freq": control_freq, "use_subspace": use_subspace,
    }

    if controller.log:
        max_d = max(e["d_content_raw"] for e in controller.log)
        ratio = traj_summary["empirical_stability_ratio"]
        print(f"  [DCSC] calls={control_calls}  max_d={max_d:.4f}  "
              f"final_λ={traj_summary['final_lambda']:.3f}  stability_ratio={ratio:.3f}")

    return m, recon, elapsed, traj_summary


# ---------------------------------------------------------------------------
# Drift-Bounded Editing: perturbation + closed-loop control
# ---------------------------------------------------------------------------

class DynamicLambda:
    """Mutable lambda value compatible with LambdaScheduler's .get(step_idx) API."""
    def __init__(self, value: float):
        self.value = value
    def get(self, step_idx: int) -> float:
        return self.value


def drift_bounded_generation(
    pipe,
    original_latent: torch.Tensor,
    original_tensor: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int,
    corr_lam: float,
    corr_layers: List[str],
    extractor,                               # CLIPFeatureExtractor
    v_content: torch.Tensor,                 # content direction
    # Editing freedom budget:
    editing_strength: float = 0.5,            # σ ∈ [0,1]: 0=preserve all, 1=full editing
    # Control params:
    control_mode: str = "dcsc",              # "dcsc" | "open_loop" | "phase3_pin"
    Kp: float = 1.0,
    control_freq: int = 5,
    pinning_threshold: float = 0.02,
    pinning_strength: float = 0.5,
    monotonic_lambda: bool = True,
    ema_alpha: float = 0.3,
    lpips_fn=None,
    compute_arcface: bool = False,
    **kwargs,
) -> Tuple[Dict, torch.Tensor, float, Dict]:
    """Drift-bounded editing: correction strength as control variable.

    Editing freedom σ maps to correction strength:
        λ_eff = λ_max * (1 - σ)     σ=0 → full preservation, σ=1 → no correction

    DCSC monitors CLIP content drift d(t) and adjusts λ_eff:
        λ_eff(t) = min(λ_max, λ_base + Kp * d(t))
    Drift → controller increases correction to protect content.

    Args:
        editing_strength: σ ∈ [0,1], user's desired editing freedom
        control_mode: "open_loop" | "phase3_pin" | "dcsc"
    """
    t0 = time.perf_counter()

    # ---- DDIM inversion with feature collection ----
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, corr_layers)

    # ---- Dynamic correction strength ----
    lam_base = corr_lam * (1.0 - editing_strength)  # λ_base from user preference
    dyn_lam = DynamicLambda(lam_base)
    sched = dyn_lam  # pass as LambdaScheduler-compatible object
    corrector = FeatureCorrector(pipe.unet, corr_layers, sched)
    corrector.set_reference(saved, 0)

    # ---- Reference content projection ----
    ref_proj = extractor.compute_content_projection(
        extractor.encode_image_from_tensor(original_tensor), v_content)

    # ---- Controller: simple CLIP drift tracker (no DCSCStyleController needed) ----
    # Direct P-control: λ_corr = λ_base + Kp * d_content, capped at corr_lam
    base_sigma = editing_strength
    print(f"  [{control_mode.upper().replace('_',' ')}] σ_user={editing_strength:.2f}  "
          f"λ_base={lam_base:.3f}  Kp={Kp:.1f}  ref_proj={ref_proj:.4f}")

    # ---- Reconstruction loop ----
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    z = noise.clone()

    control_calls = 0
    trajectory = {
        "steps": [], "lambda_corr": [], "d_content_raw": [], "mode": control_mode,
    }
    pinning_log = []
    max_drift = 0.0

    for step_idx, t in enumerate(timesteps):
        t_int = int(t)
        if t_int in saved:
            corrector.set_reference(saved[t_int], step_idx)
        else:
            corrector.set_reference({}, step_idx)

        # ---- Control update ----
        do_control = (
            step_idx > 0 and control_freq > 0
            and step_idx % control_freq == 0
        )
        if do_control:
            with torch.no_grad():
                current_img = decode_latent(pipe, z.clone())
            v_current = extractor.encode_image_from_tensor(current_img)
            cur_proj = extractor.compute_content_projection(v_current, v_content)
            d_content = abs(cur_proj - ref_proj)
            if d_content > max_drift:
                max_drift = d_content
            control_calls += 1

            if control_mode == "dcsc":
                # P-control: λ_corr = λ_base + Kp * d_content
                lam_corr = min(corr_lam, lam_base + Kp * d_content)
                dyn_lam.value = lam_corr
            elif control_mode == "phase3_pin":
                if d_content > pinning_threshold:
                    scale = max(0.0, 1.0 - pinning_strength * (d_content / max(ref_proj, 0.01)))
                    effective_sigma = base_sigma * scale
                    dyn_lam.value = corr_lam * (1.0 - effective_sigma)
                # else: keep current lam
            # open_loop: no change to dyn_lam

            trajectory["steps"].append(step_idx)
            trajectory["lambda_corr"].append(dyn_lam.value)
            trajectory["d_content_raw"].append(d_content)
            if control_calls <= 3:
                print(f"  [{control_mode.upper()}] step={step_idx:3d}  "
                      f"d={d_content:.4f}  λ={dyn_lam.value:.3f}")

        # ---- UNet forward + scheduler step ----
        inp = scheduler.scale_model_input(z, t)
        with torch.no_grad():
            noise_pred = pipe.unet(inp, t, encoder_hidden_states=prompt_embeds).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    # ---- Cleanup ----
    corrector.remove()

    # ---- Decode & metrics ----
    recon = decode_latent(pipe, z)
    elapsed = time.perf_counter() - t0
    m = compute_metrics(original_tensor, recon, lpips_fn, compute_arcface)

    v_orig = extractor.encode_image_from_tensor(original_tensor)
    v_recon = extractor.encode_image_from_tensor(recon)
    m["CLIP_content"] = float((v_recon * v_orig).sum())

    traj_summary = {
        "control_mode": control_mode,
        "editing_strength": editing_strength,
        "max_content_drift": max_drift,
        "final_lambda_corr": dyn_lam.value,
        "n_control_calls": control_calls,
        "trajectory": trajectory,
    }

    if control_mode == "dcsc":
        print(f"  [DCSC] calls={control_calls}  max_drift={max_drift:.4f}  "
              f"final_λ={dyn_lam.value:.3f}")
    elif control_mode == "phase3_pin":
        triggered = sum(1 for d in pinning_log if d[2] > pinning_threshold) if pinning_log else 0
        print(f"  [Phase3] calls={control_calls}  max_drift={max_drift:.4f}  "
              f"triggered={triggered}/{len(pinning_log)}")

    return m, recon, elapsed, traj_summary
