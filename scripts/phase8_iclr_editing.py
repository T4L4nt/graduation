"""
Phase 8 ICLR — Task B: Editing Capability Validation (Cut A + Noise A + P2P)

Memory-efficient design:
  1. Pre-compute all CLIP encodings → save to disk → free CLIP model
  2. Load SD 1.5 → run all editing tasks (no CLIP on GPU)
  3. Load CLIP results from disk → compute directional similarities

5 task types × 5 prompts each = 25 editing pairs × 3 conditions = 75 edits.
Metrics: edit consistency (LPIPS), source preservation (SSIM), CLIP dir sim.
"""

import argparse, json, csv, sys, os, pickle
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from phase7_skip_intervention import (
    SkipIntervention, ddim_inversion,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"

OUT_DIR = Path("outputs/phase8_iclr_editing")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Noise Intervention
# ---------------------------------------------------------------------------

class NoiseIntervention:
    def __init__(self, unet, cut_up_indices):
        self.unet = unet
        self.cut_up_indices = set(cut_up_indices)
        self._originals = {}

    def __enter__(self):
        for idx in self.cut_up_indices:
            up_block = self.unet.up_blocks[idx]
            self._originals[idx] = up_block.forward
            original = up_block.forward
            def make_patched(orig_fn):
                def patched_forward(hidden_states, res_hidden_states_tuple, *args, **kwargs):
                    noisy = tuple(torch.randn_like(t) * t.std() + t.mean()
                                  for t in res_hidden_states_tuple)
                    return orig_fn(hidden_states, noisy, *args, **kwargs)
                return patched_forward
            up_block.forward = make_patched(original)
        return self

    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.unet.up_blocks[idx].forward = orig
        self._originals.clear()


# ---------------------------------------------------------------------------
# CLIP pre-computation (runs first, freed before SD loads)
# ---------------------------------------------------------------------------

CLIP_CACHE = OUT_DIR / "clip_cache.pkl"


def _clip_encode_text(model, processor, text):
    out = model.get_text_features(
        **processor(text=[text], return_tensors="pt", padding=True,
                   truncation=True).to(DEVICE))
    return out.pooler_output if hasattr(out, 'pooler_output') else out


def _clip_encode_image(model, processor, img_pil):
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
    out = model.get_image_features(**inputs)
    return out.pooler_output if hasattr(out, 'pooler_output') else out


def precompute_clip(tasks):
    """Load CLIP, encode all source images & target texts, save to pickle, free CLIP."""
    from transformers import CLIPModel, CLIPProcessor

    print("[CLIP] Loading CLIP-ViT-L...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    ).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    )

    cache = {}
    for task in tasks:
        task_id, task_type, src_prompt, tgt_prompt = task
        with torch.no_grad():
            txt_src = _clip_encode_text(model, processor, src_prompt)
            txt_tgt = _clip_encode_text(model, processor, tgt_prompt)
        cache[task_id] = {
            "src_text_emb": txt_src.cpu(),
            "tgt_text_emb": txt_tgt.cpu(),
            "src_img_emb": None,
            "src_prompt": src_prompt,
            "tgt_prompt": tgt_prompt,
        }
        print(f"  [{task_id}] text encoded")

    del model, processor
    torch.cuda.empty_cache()
    print(f"[CLIP] Freed. Cache: {len(cache)} tasks ready.")

    with open(CLIP_CACHE, "wb") as f:
        pickle.dump(cache, f)

    return cache


def update_clip_cache_src_img(task_id, img_tensor):
    """Add source image CLIP embedding to cache (callable after SD generates image)."""
    from transformers import CLIPModel, CLIPProcessor
    import pickle as pkl

    if not CLIP_CACHE.exists():
        return

    with open(CLIP_CACHE, "rb") as f:
        cache = pkl.load(f)

    if task_id not in cache:
        return

    # Load CLIP, encode one image, free CLIP
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    ).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    )

    img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        img_emb = _clip_encode_image(model, processor, img_pil)

    cache[task_id]["src_img_emb"] = img_emb.cpu()
    del model, processor
    torch.cuda.empty_cache()

    with open(CLIP_CACHE, "wb") as f:
        pkl.dump(cache, f)


def compute_clip_dir_sim(task_id, edit_tensor, overwrite_cache=False):
    """Compute CLIP dir sim for an edit result vs pre-computed text embs."""
    from transformers import CLIPModel, CLIPProcessor
    import pickle as pkl

    if not CLIP_CACHE.exists():
        return 0.0

    with open(CLIP_CACHE, "rb") as f:
        cache = pkl.load(f)

    if task_id not in cache:
        return 0.0

    entry = cache[task_id]

    # Load CLIP briefly
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    ).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", local_files_only=True,
    )

    # Encode edit image
    img_np = (edit_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        img_edit = _clip_encode_image(model, processor, img_pil)

    # Move text embs to GPU for computation
    txt_src = entry["src_text_emb"].to(DEVICE)
    txt_tgt = entry["tgt_text_emb"].to(DEVICE)
    src_img_emb = entry["src_img_emb"]
    if src_img_emb is not None:
        src_img_emb = src_img_emb.to(DEVICE)

    # Compute CLIP dir sim
    if src_img_emb is not None:
        delta_img = img_edit - src_img_emb
    else:
        delta_img = img_edit  # fallback

    delta_txt = txt_tgt - txt_src
    delta_img = delta_img / (delta_img.norm(dim=-1, keepdim=True) + 1e-8)
    delta_txt = delta_txt / (delta_txt.norm(dim=-1, keepdim=True) + 1e-8)

    result = float((delta_img * delta_txt).sum(dim=-1).item())

    del model, processor
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# P2P Attention Injection (simplified)
# ---------------------------------------------------------------------------

def _find_attn_modules(unet):
    modules = []
    for name, mod in unet.named_modules():
        if hasattr(mod, 'attn2') and hasattr(mod.attn2, 'to_k'):
            modules.append((name, mod))
    return modules


class P2PAttentionInjector:
    def __init__(self, unet, inject_layers=None):
        self.unet = unet
        self.all_modules = dict(_find_attn_modules(unet))
        if inject_layers is None:
            self.inject_layers = [n for n in self.all_modules
                                  if "up_blocks" in n or "mid_block" in n]
        else:
            self.inject_layers = [n for n in inject_layers if n in self.all_modules]
        self.stored_attn = {}
        self.current_step = 0
        self.mode = "off"
        self.handles = []
        self._setup()

    def _setup(self):
        for name in self.inject_layers:
            mod = self.all_modules[name]
            h = mod.attn2.register_forward_hook(
                lambda m, inp, out, n=name: self._hook(n, out)
            )
            self.handles.append(h)

    def _hook(self, name, output):
        if self.mode == "off":
            return output
        if self.mode == "save":
            if self.current_step not in self.stored_attn:
                self.stored_attn[self.current_step] = {}
            self.stored_attn[self.current_step][name] = output.detach().clone()
            return output
        elif self.mode == "inject":
            if (self.current_step in self.stored_attn and
                name in self.stored_attn[self.current_step]):
                return self.stored_attn[self.current_step][name]
            return output

    def clear(self):
        self.stored_attn = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# 25 Editing tasks
# ---------------------------------------------------------------------------

EDITING_TASKS = [
    ("obj_01", "object_replacement", "a cat sitting on a sofa", "a dog sitting on a sofa"),
    ("obj_02", "object_replacement", "a red apple on a wooden table", "a green apple on a wooden table"),
    ("obj_03", "object_replacement", "a car parked on a street", "a bicycle parked on a street"),
    ("obj_04", "object_replacement", "a bowl of fresh strawberries", "a bowl of fresh blueberries"),
    ("obj_05", "object_replacement", "a white dove on a branch", "a blue jay on a branch"),
    ("sty_01", "style_transfer", "a mountain landscape, photorealistic", "a mountain landscape, oil painting style"),
    ("sty_02", "style_transfer", "a woman with long hair, photograph", "a woman with long hair, watercolor painting"),
    ("sty_03", "style_transfer", "a city skyline at sunset, photo", "a city skyline at sunset, van gogh style"),
    ("sty_04", "style_transfer", "a bowl of fruit, realistic photo", "a bowl of fruit, cubist painting"),
    ("sty_05", "style_transfer", "a forest path, photograph", "a forest path, pencil sketch"),
    ("att_01", "attribute_change", "a red car on a road", "a blue car on a road"),
    ("att_02", "attribute_change", "a small wooden house", "a large wooden house"),
    ("att_03", "attribute_change", "a young woman with short hair", "a young woman with long hair"),
    ("att_04", "attribute_change", "a brown leather sofa", "a black leather sofa"),
    ("att_05", "attribute_change", "a white ceramic mug", "a red ceramic mug"),
    ("bg_01", "background_change", "a dog in a forest", "a dog in a city street"),
    ("bg_02", "background_change", "a vase of flowers on a table", "a vase of flowers on a beach"),
    ("bg_03", "background_change", "a person in a garden", "a person in a snowy landscape"),
    ("bg_04", "background_change", "a sports car in a showroom", "a sports car on a race track"),
    ("bg_05", "background_change", "a cup of coffee on a desk", "a cup of coffee on a mountain top"),
    ("add_01", "object_addition", "a birthday cake on a table", "a birthday cake with candles on a table"),
    ("add_02", "object_addition", "a living room with a sofa", "a living room with a sofa and a christmas tree"),
    ("add_03", "object_addition", "a desk with a laptop", "a desk with a laptop and a potted plant"),
    ("add_04", "object_addition", "a beach at sunset", "a beach at sunset with a sailboat"),
    ("add_05", "object_addition", "a kitchen counter", "a kitchen counter with a fruit basket"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sd_pipeline():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, local_files_only=True,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def decode_latent(pipe, latent):
    with torch.no_grad():
        return pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample


def generate_image(pipe, prompt, seed=42, num_steps=50, guidance_scale=7.5):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_steps, device=DEVICE)
    with torch.no_grad():
        result = pipe(prompt=prompt, num_inference_steps=num_steps,
                      guidance_scale=guidance_scale, generator=generator,
                      output_type="pt")
    return result.images


def save_tensor(tensor, path):
    img_np = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


# ---------------------------------------------------------------------------
# P2P editing pipeline
# ---------------------------------------------------------------------------

def p2p_edit(pipe, src_img_tensor, src_prompt, tgt_prompt,
             injector, lpips_fn, num_steps=50):
    """P2P edit: inversion (save attn) → reconstruction (inject attn, new prompt).

    Returns (metrics dict, edit_tensor).
    """
    with torch.no_grad():
        src_latent = pipe.vae.encode(src_img_tensor.to(DEVICE, dtype=DTYPE)).latent_dist.sample()
        src_latent = src_latent * pipe.vae.config.scaling_factor

    src_embeds = pipe.encode_prompt(src_prompt, DEVICE, 1, False)[0]
    tgt_embeds = pipe.encode_prompt(tgt_prompt, DEVICE, 1, False)[0]

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # Phase 1: Inversion (save attention)
    injector.mode = "save"
    injector.clear()
    z = src_latent.clone()
    extended_ts = timesteps.tolist() + [0]
    for i in range(len(extended_ts) - 1, 0, -1):
        t_cur = extended_ts[i]
        t_next = extended_ts[i - 1]
        injector.current_step = i
        noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=src_embeds).sample
        alpha_cur = scheduler.alphas_cumprod[t_cur]
        alpha_next = scheduler.alphas_cumprod[t_next]
        coeff1 = (alpha_next / alpha_cur).sqrt()
        sigma_cur = (1 - alpha_cur).sqrt()
        sigma_next = (1 - alpha_next).sqrt()
        coeff2 = sigma_next - coeff1 * sigma_cur
        z = coeff1 * z + coeff2 * noise_pred

    # Phase 2: Reconstruction with target prompt (inject attention)
    injector.mode = "inject"
    with torch.no_grad():
        for step_i, t in enumerate(timesteps):
            injector.current_step = len(timesteps) - step_i
            noise_pred = pipe.unet(z, t, encoder_hidden_states=tgt_embeds).sample
            z = scheduler.step(noise_pred, t, z).prev_sample

    injector.mode = "off"
    edit_tensor = decode_latent(pipe, z)

    # Generate target reference
    tgt_ref = generate_image(pipe, tgt_prompt, seed=999, num_steps=num_steps)

    # Edit consistency: LPIPS(edit, target_reference)
    lpips_consistency = float(lpips_fn(edit_tensor, tgt_ref).item())

    # Source preservation: SSIM(edit, source)
    src_np = (src_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    edit_np = (edit_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    ssim_preservation = float(ssim(src_np, edit_np, channel_axis=2, data_range=1.0))

    # PSNR edit vs target_ref
    mse = torch.nn.functional.mse_loss(edit_tensor.float(), tgt_ref.float())
    psnr_edit = (20 * torch.log10(2.0 / (torch.sqrt(mse) + 1e-8))).item()

    return {
        "lpips_consistency": lpips_consistency,
        "ssim_preservation": ssim_preservation,
        "psnr_vs_target": float(psnr_edit),
    }, edit_tensor


# ---------------------------------------------------------------------------
# Run single task under one condition
# ---------------------------------------------------------------------------

def run_editing_task(pipe, task, condition, injector, lpips_fn, num_steps=50):
    task_id, task_type, src_prompt, tgt_prompt = task
    print(f"    [{condition}] {task_id}: '{src_prompt[:40]}...'",
          end=" ", flush=True)

    src_img = generate_image(pipe, src_prompt, seed=42, num_steps=num_steps)

    intervention = None
    if condition == "cut_a":
        intervention = SkipIntervention(pipe.unet, [2])
    elif condition == "noise_a":
        intervention = NoiseIntervention(pipe.unet, [2])

    try:
        if intervention:
            with intervention:
                metrics, edit_tensor = p2p_edit(
                    pipe, src_img, src_prompt, tgt_prompt,
                    injector, lpips_fn, num_steps)
        else:
            metrics, edit_tensor = p2p_edit(
                pipe, src_img, src_prompt, tgt_prompt,
                injector, lpips_fn, num_steps)
    except torch.cuda.OutOfMemoryError:
        print("OOM - skipping")
        torch.cuda.empty_cache()
        return None, None
    except Exception as e:
        print(f"ERROR: {e}")
        torch.cuda.empty_cache()
        return None, None

    torch.cuda.empty_cache()

    metrics["task_id"] = task_id
    metrics["task_type"] = task_type
    metrics["condition"] = condition
    metrics["src_prompt"] = src_prompt
    metrics["tgt_prompt"] = tgt_prompt

    # Update CLIP cache with source image embedding (only once per task)
    if condition == "original":
        update_clip_cache_src_img(task_id, src_img)

    print(f"LPIPS={metrics['lpips_consistency']:.3f} "
          f"SSIM={metrics['ssim_preservation']:.3f}")

    return metrics, edit_tensor


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_3x3_grid(tasks, conditions, edit_results, out_path):
    rep_tasks = []
    for ttype in ["object_replacement", "style_transfer", "background_change"]:
        for t in tasks:
            if t[1] == ttype:
                rep_tasks.append(t)
                break
    if len(rep_tasks) < 3:
        rep_tasks = tasks[:3]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
    for row, cond in enumerate(conditions):
        for col, task in enumerate(rep_tasks):
            ax = axes[row, col]
            key = (task[0], cond)
            if key in edit_results:
                img_tensor = edit_results[key]
                img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                ax.imshow(img_np)
                ax.set_title(f"{cond}\n{task[2][:40]}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")

    row_labels = ["Original", "Cut A (zero skip)", "Noise A (noise skip)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold",
                                rotation=90, va="center", labelpad=20)

    plt.suptitle("P2P Editing: Skip Intervention Impact",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] 3x3 grid → {out_path}")


def plot_metric_comparison(metrics_list, out_path):
    conditions = ["original", "cut_a", "noise_a"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    metric_names = ["lpips_consistency", "ssim_preservation", "clip_dir_sim"]
    metric_labels = ["LPIPS Consistency ↓", "SSIM Preservation ↑", "CLIP Dir Sim ↑"]

    for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
        means, stds = [], []
        for cond in conditions:
            vals = [m[mname] for m in metrics_list
                    if m.get("condition") == cond and mname in m]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        colors = ["#3498db", "#e74c3c", "#e67e22"]
        ax.bar(conditions, means, color=colors, width=0.5)
        ax.errorbar(conditions, means, yerr=stds, fmt="none",
                    ecolor="gray", capsize=6)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for i, m in enumerate(means):
            ax.annotate(f"{m:.3f}", (i, m), fontsize=9, ha="center",
                       va="bottom", xytext=(0, 3), textcoords="offset points")

    plt.suptitle("Editing Quality by Condition", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Metric comparison → {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(metrics_list):
    print(f"\n{'='*70}")
    print("EDITING CAPABILITY VALIDATION — RESULTS")
    conditions = ["original", "cut_a", "noise_a"]
    cond_labels = {"original": "Original", "cut_a": "Cut A", "noise_a": "Noise A"}

    for mname, mlabel, direction in [
        ("lpips_consistency", "LPIPS Consistency", "lower=better"),
        ("ssim_preservation", "SSIM Preservation", "higher=better"),
        ("clip_dir_sim", "CLIP Dir Sim", "higher=better"),
    ]:
        print(f"\n--- {mlabel} ({direction}) ---")
        cond_vals = {}
        for cond in conditions:
            vals = [m[mname] for m in metrics_list
                    if m.get("condition") == cond and mname in m]
            if vals:
                cond_vals[cond] = vals
                print(f"  {cond_labels[cond]}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})")

        for c1, c2 in [("original", "cut_a"), ("original", "noise_a"), ("cut_a", "noise_a")]:
            if c1 in cond_vals and c2 in cond_vals and len(cond_vals[c1]) >= 3:
                t, p = stats.ttest_rel(cond_vals[c1], cond_vals[c2])
                d = (np.mean(cond_vals[c2]) - np.mean(cond_vals[c1])) / max(np.std(cond_vals[c1]), 1e-8)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*" if p < 0.05 else "")
                print(f"  {cond_labels[c1]} vs {cond_labels[c2]}: "
                      f"t={t:.3f}, p={p:.4f}, d={d:+.3f} {sig}")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ICLR Task B: Editing Validation")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--quick", type=int, default=None)
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["original", "cut_a", "noise_a"])
    args = parser.parse_args()

    tasks = EDITING_TASKS[:args.tasks] if args.tasks else EDITING_TASKS
    if args.quick:
        tasks = tasks[:args.quick]
    conditions = args.conditions

    print(f"[Setup] {len(tasks)} tasks × {len(conditions)} conditions")
    print(f"[Output] {OUT_DIR.resolve()}")
    print(f"[GPU] free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB "
          f"/ {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")

    # Phase 1: Pre-compute CLIP text embeddings (before SD loads)
    precompute_clip(tasks)

    # Phase 2: Load SD, run editing
    torch.cuda.empty_cache()
    print(f"\n[SD] Loading SD 1.5 + LPIPS...")
    print(f"[GPU] free before SD: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
    pipe = load_sd_pipeline()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    print(f"[GPU] free after SD+LPIPS: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    injector = P2PAttentionInjector(pipe.unet)

    all_metrics = []
    edit_results = {}
    saved_edit_dir = OUT_DIR / "edits"
    saved_edit_dir.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(tasks):
        task_id = task[0]
        print(f"\n[{i+1}/{len(tasks)}] {task_id}: {task[2][:40]} → {task[3][:40]}")

        for cond in conditions:
            metrics, edit_tensor = run_editing_task(
                pipe, task, cond, injector, lpips_fn, args.steps)
            if metrics is not None:
                all_metrics.append(metrics)
            if edit_tensor is not None:
                edit_results[(task_id, cond)] = edit_tensor
                save_tensor(edit_tensor, saved_edit_dir / f"{task_id}_{cond}.png")

        # Free P2P stored maps after each task
        injector.clear()
        torch.cuda.empty_cache()

    injector.remove()

    # Phase 3: Compute CLIP directional similarities (post-hoc)
    print(f"\n[CLIP] Computing directional similarities for {len(all_metrics)} results...")
    for m in all_metrics:
        tid = m["task_id"]
        cond = m["condition"]
        key = (tid, cond)
        if key in edit_results:
            clip_sim = compute_clip_dir_sim(tid, edit_results[key])
            m["clip_dir_sim"] = clip_sim
        else:
            m["clip_dir_sim"] = 0.0
    print(f"[CLIP] Done. GPU free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Report
    print_report(all_metrics)

    # Figures
    print(f"\n[fig] Generating figures...")
    if len(edit_results) >= 3:
        plot_3x3_grid(tasks, conditions, edit_results,
                      OUT_DIR / "grid_3x3.png")
        plot_3x3_grid(tasks, conditions, edit_results,
                      OUT_DIR / "grid_3x3.pdf")
    if all_metrics:
        plot_metric_comparison(all_metrics, OUT_DIR / "metric_comparison.png")
        plot_metric_comparison(all_metrics, OUT_DIR / "metric_comparison.pdf")

    # Save
    print(f"\n[save] Writing results...")
    if all_metrics:
        fieldnames = ["task_id", "task_type", "condition",
                      "lpips_consistency", "ssim_preservation",
                      "clip_dir_sim", "psnr_vs_target"]
        with open(OUT_DIR / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"[CSV] → {OUT_DIR / 'results.csv'}")

    summary = {
        "config": {"n_tasks": len(tasks), "n_conditions": len(conditions),
                    "steps": args.steps, "conditions": conditions,
                    "n_successful": sum(1 for m in all_metrics if m.get("lpips_consistency", 0) > 0)},
        "per_condition": {},
    }
    for cond in conditions:
        cond_metrics = [m for m in all_metrics if m.get("condition") == cond]
        if not cond_metrics:
            continue
        summary["per_condition"][cond] = {}
        for mname in ["lpips_consistency", "ssim_preservation",
                       "clip_dir_sim", "psnr_vs_target"]:
            vals = [m[mname] for m in cond_metrics if mname in m]
            if vals:
                summary["per_condition"][cond][mname] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Summary → {OUT_DIR / 'summary.json'}")

    print(f"\n{'='*60}")
    print("Task B complete.")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
