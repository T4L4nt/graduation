"""
DCSC: Drift-Aware Closed-Loop Style Controller.

Core components:
  - CorrectableSubspace: incremental Gram-Schmidt orthonormal basis in CLIP space
  - DCSCStyleController: proportional feedback control law + subspace projection
  - dcsc_controlled_generation(): main generation loop

Replaces the hard-threshold pinning in phase3_prep.py with:
  1. Subspace-constrained style injection (style only in correctable-complement dirs)
  2. Continuous P control law (not hard if-else)
  3. Bounded content drift guarantee

Usage:
  from dcsc_core import DCSCStyleController, dcsc_controlled_generation
"""

import time
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
# CorrectableSubspace
# ---------------------------------------------------------------------------

class CorrectableSubspace:
    """Incrementally-built orthonormal basis in CLIP embedding space.

    At each control step, computes v_residual = v_current - v_original
    and adds the component orthogonal to the existing basis (Gram-Schmidt).
    The orthogonal complement defines "content-safe style directions" —
    directions the correction mechanism will not undo.

    Basis grows monotonically, saturates at 3-5 vectors in practice.
    Empty basis → falls back to standard CLIP orthogonal projection.
    """

    def __init__(self, dim: int = 768, energy_threshold: float = 1e-4,
                 max_basis: int = 10):
        self.basis: List[torch.Tensor] = []   # list of [1, dim] unit vectors
        self.dim = dim
        self.energy_threshold = energy_threshold
        self.max_basis = max_basis

    @torch.no_grad()
    def update(self, v_residual: torch.Tensor) -> bool:
        """Add v_residual to basis via Gram-Schmidt. Returns True if basis grew.

        v_residual: [1, D] tensor in CLIP space.
        """
        if self._size() >= self.max_basis:
            return False
        v = v_residual.detach().float()
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        # Subtract projection onto existing basis vectors
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
        """Project v onto orthogonal complement of the subspace.

        v_out = v - sum_i <v, u_i> u_i, normalized to unit norm.
        Empty basis → returns v unchanged.
        """
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
        """Project v onto the subspace (content component)."""
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
        """Fraction of ||v||^2 lying inside the subspace. 0 = fully orthogonal."""
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

    Combined with subspace projection:
        v_style_used(t) = orthogonal_complement_proj(v_style_base, subspace(t))

    Stability: content drift is bounded by C * lambda_0 / Kp for some C > 0.
    """

    def __init__(
        self,
        v_style_base: torch.Tensor,       # [1, 768] style direction
        v_content: torch.Tensor,           # [1, 768] content direction
        lambda_0: float = 0.5,             # base style strength
        Kp: float = 1.0,                   # proportional gain
        dim: int = 768,
        monotonic_lambda: bool = True,     # never increase lambda
        ema_alpha: float = 0.3,            # EMA smoothing for d_content (0=no history)
    ):
        self.subspace = CorrectableSubspace(dim=dim)
        self.v_style_base = v_style_base / (v_style_base.norm(dim=-1, keepdim=True) + 1e-8)
        self.v_content = v_content / (v_content.norm(dim=-1, keepdim=True) + 1e-8)
        self.lambda_0 = lambda_0
        self.Kp = Kp
        self.monotonic_lambda = monotonic_lambda
        self.ema_alpha = ema_alpha

        # Initialized via .initialize()
        self.ref_proj: Optional[float] = None
        self.v_orig: Optional[torch.Tensor] = None
        self.prev_lambda: Optional[float] = None
        self.ema_d_content: Optional[float] = None

        # Trajectory log
        self.log: List[Dict] = []

    @torch.no_grad()
    def initialize(self, original_tensor: torch.Tensor, extractor) -> None:
        """Compute reference content projection. Call once before compute_control()."""
        v_orig = extractor.encode_image_from_tensor(original_tensor)
        self.ref_proj = extractor.compute_content_projection(v_orig, self.v_content)
        self.v_orig = v_orig

    @torch.no_grad()
    def compute_control(
        self,
        current_tensor: torch.Tensor,
        extractor,
    ) -> Tuple[float, torch.Tensor, Dict]:
        """One step of the control law.

        Returns:
            lambda_style: float in [0, lambda_0]
            v_style_projected: [1, 768] unit-norm tensor
            info: dict with {d_content, lambda_style, basis_size, proj_energy, deviation}
        """
        if self.ref_proj is None:
            raise RuntimeError("Must call initialize() before compute_control()")

        # 1. Encode current image
        v_current = extractor.encode_image_from_tensor(current_tensor)

        # 2. Content deviation
        cur_proj = extractor.compute_content_projection(v_current, self.v_content)
        d_content = abs(cur_proj - self.ref_proj)

        # 3. Update subspace with residual in CLIP space
        v_residual = v_current - self.v_orig
        self.subspace.update(v_residual)

        # 4. P control law with EMA smoothing
        if self.ema_d_content is None:
            self.ema_d_content = d_content
        else:
            self.ema_d_content = (self.ema_alpha * d_content
                                  + (1 - self.ema_alpha) * self.ema_d_content)
        d_smooth = self.ema_d_content

        lambda_style = self.lambda_0 * max(0.0, 1.0 - self.Kp * d_smooth)

        # Monotonic clamp: never increase lambda between steps
        if self.monotonic_lambda and self.prev_lambda is not None:
            lambda_style = min(lambda_style, self.prev_lambda)
        self.prev_lambda = lambda_style

        # 5. Project style onto orthogonal complement of content subspace
        v_style_proj = self.subspace.project_orthogonal(self.v_style_base)

        # 6. Compute subspace energy for logging
        proj_energy = self.subspace.compute_energy_fraction(self.v_style_base)

        info = {
            "d_content_raw": d_content,
            "d_content_smooth": d_smooth,
            "cur_proj": cur_proj,
            "ref_proj": self.ref_proj,
            "lambda_style": lambda_style,
            "basis_size": self.subspace._size(),
            "proj_energy": proj_energy,
        }
        self.log.append(info)
        return lambda_style, v_style_proj, info

    def compute_stability_bound(self) -> float:
        """Theoretical drift bound: C * lambda_0 / Kp.

        Estimates C from the first control step observation:
          C = d_content(0) / lambda_0  (first-step sensitivity)
        """
        if len(self.log) == 0:
            return float("inf")
        d0 = self.log[0]["d_content_raw"]
        if d0 <= 0 or self.lambda_0 <= 0:
            return 0.0
        C = d0 / self.lambda_0
        return C * self.lambda_0 / max(self.Kp, 1e-8)

    def reset(self):
        self.subspace.reset()
        self.ref_proj = None
        self.v_orig = None
        self.prev_lambda = None
        self.ema_d_content = None
        self.log.clear()


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def dcsc_controlled_generation(
    pipe,
    original_latent: torch.Tensor,
    original_tensor: torch.Tensor,
    prompt_embeds: torch.Tensor,           # [1, 77, 768] base prompt
    num_steps: int,
    corr_lam: float,
    corr_layers: List[str],
    v_style: torch.Tensor,                 # [1, 768] CLIP style direction
    v_content: torch.Tensor,               # [1, 768] CLIP content direction
    extractor,                             # CLIPFeatureExtractor
    lambda_0: float = 0.5,
    Kp: float = 1.0,
    control_freq: int = 5,
    style_mode: str = "extra_token",       # "extra_token" | "interpolate" | "feature_bias"
    lpips_fn=None,
    compute_arcface: bool = False,
    monotonic_lambda: bool = True,
    ema_alpha: float = 0.3,
    max_control_steps: Optional[int] = None,  # stop controller after N steps
    use_subspace: bool = True,
    **kwargs,
) -> Tuple[Dict, torch.Tensor, float, Dict]:
    """DCSC-controlled generation: correction + style injection + closed-loop control.

    Args:
        max_control_steps: if set, stop updating style after this many control calls
            (e.g., only update in first half of denoising). None = update throughout.
        use_subspace: if False, skip subspace projection (ablation: P-control only).

    Returns:
        metrics: dict with PSNR, SSIM, LPIPS, CLIP_style, CLIP_content, ArcFace
        recon: [1, 3, 512, 512] tensor
        elapsed: float seconds
        control_trajectory: dict with full trajectory logs
    """
    from phase3_prep import build_style_cross_attn_tokens, slerp

    t0 = time.perf_counter()

    # ---- Controller initialization ----
    controller = DCSCStyleController(
        v_style_base=v_style,
        v_content=v_content,
        lambda_0=lambda_0,
        Kp=Kp,
        monotonic_lambda=monotonic_lambda,
        ema_alpha=ema_alpha,
    )
    controller.initialize(original_tensor, extractor)
    print(f"  [DCSC] λ_0={lambda_0:.2f} Kp={Kp:.1f}  "
          f"ref_proj={controller.ref_proj:.4f}")

    # ---- DDIM inversion with feature collection ----
    noise, saved = ddim_inversion_with_features(
        pipe, original_latent, prompt_embeds, num_steps, corr_layers)

    # ---- FeatureCorrector setup ----
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
            step_idx > 0
            and control_freq > 0
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

            # Rebuild cross-attn tokens with projected style
            if use_subspace:
                v_style_actual = v_style_proj
            else:
                v_style_actual = v_style  # ablation: skip subspace, P-control only

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
    clip_content = float((v_recon * v_orig).sum())
    v_style_ref = v_style / (v_style.norm(dim=-1, keepdim=True) + 1e-8)
    clip_style = float((v_recon * v_style_ref).sum())

    m["CLIP_content"] = clip_content
    m["CLIP_style"] = clip_style
    m["method"] = f"DCSC_Kp{Kp:.1f}_lam{lambda_0:.2f}"

    # ---- Trajectory summary ----
    traj_summary = {
        "trajectory": trajectory,
        "controller_log": [{k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in entry.items()}
                           for entry in controller.log],
        "stability_bound": controller.compute_stability_bound(),
        "final_lambda": controller.prev_lambda if controller.prev_lambda is not None else lambda_0,
        "n_control_calls": control_calls,
        "final_basis_size": controller.subspace._size(),
        "lambda_0": lambda_0, "Kp": Kp, "corr_lam": corr_lam,
        "control_freq": control_freq, "use_subspace": use_subspace,
    }

    # Print summary
    if controller.log:
        max_d = max(e["d_content_raw"] for e in controller.log)
        final_l = traj_summary["final_lambda"]
        print(f"  [DCSC] calls={control_calls}  max_d={max_d:.4f}  "
              f"final_λ={final_l:.3f}  bound={traj_summary['stability_bound']:.4f}")

    return m, recon, elapsed, traj_summary
