"""
Phase 4 SOTA 对比：Prompt-to-Prompt (P2P)

P2P (Hertz et al., 2022) 通过保存/替换交叉注意力图实现内容保持编辑。
核心思想：DDIM 反演时保存每步的交叉注意力图，重建时将其注入生成过程，
使新 prompt 引导的生成保持原始图像的空间结构。

对比维度：
- P2P 是"空间级"内容保持（cross-attn maps），我们是"特征级"（ResNet features）
- P2P 通过修改文本 prompt 触发编辑，我们通过 CLIP 风格向量注入
- 两者都是零训练方法

用法:
  python scripts/phase4_p2p.py --images data/coco_val/coco_000000000139.jpg
  python scripts/phase4_p2p.py --n-images 3
"""

import argparse, json, sys, time, os
from pathlib import Path

import torch, numpy as np
from PIL import Image
import lpips

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, MODEL_ID, load_pipeline, load_image, decode_latent,
    ddim_inversion, compute_metrics, save_recon_img,
    get_top_drift_layers,
)
from phase3_prep import CLIPFeatureExtractor, find_closest_style, STYLE_CANDIDATES

OUT_DIR = Path("outputs/phase4_sota/p2p")


# ---------------------------------------------------------------------------
# Cross-attention hooking for P2P
# ---------------------------------------------------------------------------

def _find_cross_attn_modules(unet):
    """Find all cross-attention modules in the UNet."""
    modules = []
    for name, mod in unet.named_modules():
        # In diffusers, BasicTransformerBlock contains attn2 (cross-attention)
        if hasattr(mod, 'attn2') and hasattr(mod.attn2, 'to_k'):
            modules.append((name, mod))
    return modules


class P2PAttentionStore:
    """Save cross-attention maps during inversion, replay during reconstruction."""

    def __init__(self, unet):
        self.unet = unet
        self.stored_maps = {}  # {step_idx: {module_name: attn_map}}
        self.active_maps = {}  # current maps to inject
        self.handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        for name, mod in _find_cross_attn_modules(self.unet):
            h = mod.attn2.register_forward_hook(
                lambda m, inp, out, n=name: self._hook(n, m, inp, out)
            )
            self.handles.append(h)

    def _hook(self, name, module, inp, output):
        """Intercept cross-attention forward pass.

        During store mode: save attention_probs.
        During replay mode: replace attention_probs with stored map.
        """
        # We can't easily intercept the softmax inside the attention module.
        # Instead, we patch the attention output by using stored query-key dot products.
        # This is a simplified approach: directly replace the attention output.
        #
        # The attention module computes: attn @ V where attn = softmax(Q@K^T/sqrt(d))
        # We save the final output (attn @ V) and replace it during generation.
        pass

    def store(self, step_idx):
        """Not directly used - we patch differently."""
        pass

    def set_maps(self, maps_dict):
        self.active_maps = maps_dict

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# Simplified P2P: attention output replacement at feature level
# ---------------------------------------------------------------------------
#
# Full P2P requires patching the softmax attention inside scaled_dot_product_attention,
# which is complex. Instead, we implement a simplified version:
#
# Key insight: cross-attention OUTPUTS are per-token weighted sums of V.
# During inversion, we save the full output of each cross-attention layer.
# During reconstruction, we interpolate between the current output and the saved one.
#
# This is conceptually similar to P2P's "refine" mode and operates at the same
# feature level as our residual correction — making the comparison cleaner.


class P2PFeatureStore:
    """Save cross-attention OUTPUTS, replay with interpolation during generation.

    This is the feature-level analogue of P2P attention map replacement.
    Formula: f_out = (1 - λ) * f_cur + λ * f_saved
    where λ is the P2P mixing weight (analogous to our correction λ).
    """

    def __init__(self, unet, lam=0.7):
        self.unet = unet
        self.lam = lam
        self.saved_outputs = {}
        self.step_idx = 0
        self.handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        for name, mod in _find_cross_attn_modules(self.unet):
            h = mod.register_forward_hook(
                lambda m, inp, out, n=name: self._replay_hook(n, out)
            )
            self.handles.append(h)

    def _replay_hook(self, name, output):
        if name not in self.saved_outputs:
            return output
        saved = self.saved_outputs[name].to(device=output.device, dtype=output.dtype)
        return output + self.lam * (saved - output)

    def set_saved(self, saved_dict, step_idx):
        self.saved_outputs = saved_dict
        self.step_idx = step_idx

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# DDIM inversion/reconstruction with P2P cross-attention saving
# ---------------------------------------------------------------------------

def ddim_inversion_p2p(pipe, latents, prompt_embeds, num_steps):
    """DDIM inversion, saving cross-attention outputs at each step."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # Collect all cross-attn module names
    attn_modules = _find_cross_attn_modules(pipe.unet)

    # Register hooks for all cross-attn modules
    saved_all = {}  # {t_int: {name: output}}
    handles = []

    def make_hook(name):
        def hook(m, inp, out):
            if isinstance(out, tuple):
                o = out[0]
            else:
                o = out
            if t_int not in saved_all:
                saved_all[t_int] = {}
            saved_all[t_int][name] = o.detach().cpu()
        return hook

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]
    t_int = None

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]
            t_int = int(t_cur)

            # Register fresh hooks
            for name, mod in attn_modules:
                h = mod.register_forward_hook(make_hook(name))
                handles.append(h)

            noise_pred = pipe.unet(
                z, t_cur, encoder_hidden_states=prompt_embeds
            ).sample

            # Remove hooks after this step
            for h in handles:
                h.remove()
            handles.clear()

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    return z, saved_all


def ddim_reconstruction_p2p(pipe, noise, prompt_embeds, num_steps,
                              saved_attn, lam=0.7):
    """DDIM reconstruction with P2P cross-attention mixing.

    At each step, the cross-attention output is mixed:
      f = f_cur + lam * (f_saved - f_cur)
    This preserves spatial layout while allowing the new prompt to affect content.
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    attn_modules = _find_cross_attn_modules(pipe.unet)

    z = noise.clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            t_int = int(t)

            # Register hooks with saved attention outputs
            handles = []
            saved_for_step = saved_attn.get(t_int, {})

            def make_replay_hook(name):
                def hook(m, inp, out):
                    if name in saved_for_step:
                        saved_out = saved_for_step[name].to(
                            device=out[0].device if isinstance(out, tuple) else out.device,
                            dtype=out[0].dtype if isinstance(out, tuple) else out.dtype
                        )
                        if isinstance(out, tuple):
                            corrected = out[0] + lam * (saved_out - out[0])
                            return (corrected,) + out[1:]
                        else:
                            return out + lam * (saved_out - out)
                    return out
                return hook

            for name, mod in attn_modules:
                h = mod.register_forward_hook(make_replay_hook(name))
                handles.append(h)

            noise_pred = pipe.unet(
                z, t, encoder_hidden_states=prompt_embeds
            ).sample

            for h in handles:
                h.remove()

            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[P2P] {len(images)} images, {args.steps} steps, λ={args.lam}")

    print("[0] Loading SD pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    lpips_fn = None
    if not args.skip_lpips:
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        print(f"\n{'='*50}\n{img_name}")

        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image(img_path)

        # Baseline
        t0 = time.perf_counter()
        noise_base = ddim_inversion(pipe, original_latent, prompt_embeds, args.steps)
        recon_base_latent = None
        scheduler = pipe.scheduler
        scheduler.set_timesteps(args.steps, device=DEVICE)
        z = noise_base.clone()
        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
                z = scheduler.step(noise_pred, t, z).prev_sample
        recon_base_latent = z
        recon_base = decode_latent(pipe, recon_base_latent)
        base_metrics = compute_metrics(original_tensor, recon_base, lpips_fn)
        base_time = time.perf_counter() - t0
        print(f"  DDIM baseline: PSNR={base_metrics['PSNR']:.2f} LPIPS={base_metrics['LPIPS']:.3f}")
        save_recon_img(recon_base, OUT_DIR, img_name, args.steps, "ddim_baseline")

        all_results.append({
            "image": img_name, "method": "DDIM", "lam": 0.0,
            **base_metrics, "time_s": base_time,
        })

        # P2P: invert with empty prompt, reconstruct with same prompt + attn mixing
        # First do inversion with attention saving
        print(f"  P2P inversion (saving cross-attn outputs)...")
        noise_p2p, saved_attn = ddim_inversion_p2p(
            pipe, original_latent, prompt_embeds, args.steps
        )

        for lam in args.lam:
            t0 = time.perf_counter()
            recon_p2p_latent = ddim_reconstruction_p2p(
                pipe, noise_p2p, prompt_embeds, args.steps, saved_attn, lam
            )
            recon_p2p = decode_latent(pipe, recon_p2p_latent)
            elapsed = time.perf_counter() - t0

            m = compute_metrics(original_tensor, recon_p2p, lpips_fn)

            # CLIP metrics
            v_p2p = extractor.encode_image_from_tensor(recon_p2p)
            clip_content = float((v_p2p * v_orig).sum())

            tag = f"p2p_lam{lam:.1f}"
            print(f"  P2P λ={lam:.1f}: PSNR={m['PSNR']:.2f} LPIPS={m['LPIPS']:.3f}  "
                  f"CLIP_c={clip_content:.3f}  ΔPSNR={m['PSNR']-base_metrics['PSNR']:+.2f}  "
                  f"({elapsed:.1f}s)")

            save_recon_img(recon_p2p, OUT_DIR, img_name, args.steps, tag)

            all_results.append({
                "image": img_name, "method": f"P2P_attn", "lam": lam,
                **m, "CLIP_content": clip_content, "time_s": elapsed,
            })

            del recon_p2p, recon_p2p_latent
            torch.cuda.empty_cache()

        del original_latent, original_tensor, noise_base, recon_base
        torch.cuda.empty_cache()

    # Save
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("P2P Summary (50 steps)")
    for img_name in sorted(set(r["image"] for r in all_results)):
        img_results = [r for r in all_results if r["image"] == img_name]
        ddim = next(r for r in img_results if r["method"] == "DDIM")
        for r in img_results:
            if r["method"] != "DDIM":
                delta = r["PSNR"] - ddim["PSNR"]
                print(f"  {img_name} {r['method']}_λ={r['lam']:.1f}: "
                      f"PSNR={r['PSNR']:.2f}  Δ={delta:+.2f}  CLIP_c={r.get('CLIP_content',0):.3f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
