"""
Phase 7c: Skip Connection Causal Intervention Experiment

验证三层预测框架的因果干预能力：
- Cut A: 切断 down_blocks.1 → up_blocks.2 (漂移峰所在层)
  预测：漂移峰 localized 加剧
- Cut B: 切断 down_blocks.3 → up_blocks.0 (低漂移区)
  预测：局部微弱变化，指纹形状基本不变
- Cut A vs Cut B 模式不同 → topology effect 而非 capacity effect

三条件对比：original / Cut A / Cut B
19图 coco_val, 50步 DDIM

Skip mapping in SD 1.5 UNet:
  down_blocks[0] → up_blocks[3]
  down_blocks[1] → up_blocks[2]  ← drift peak
  down_blocks[2] → up_blocks[1]
  down_blocks[3] → up_blocks[0]  ← low-drift region
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-v1-5"

OUT_DIR = Path("outputs/phase7_skip_intervention")
COCO_VAL_DIR = Path("data/coco_val")

# ---------------------------------------------------------------------------
# Skip connection intervention
# ---------------------------------------------------------------------------

class SkipIntervention:
    """Zero out res_hidden_states_tuple for specified up_blocks during forward.

    When the res_hidden_states_tuple (skip connection from corresponding down_block)
    is zeroed, the up_block operates without encoder skip information.
    The pretrained weights remain unchanged; only the skip tensor is modified
    at inference time. The full inversion-reconstruction pipeline is re-executed
    under the modified topology.
    """

    def __init__(self, unet, cut_up_indices):
        """
        Args:
            unet: SD UNet2DConditionModel
            cut_up_indices: list of up_block indices whose skip connections to zero
                            e.g. [2] for Cut A, [0] for Cut B
        """
        self.unet = unet
        self.cut_up_indices = set(cut_up_indices)
        self._originals = {}  # up_block_idx -> original forward

    def __enter__(self):
        for idx in self.cut_up_indices:
            up_block = self.unet.up_blocks[idx]
            self._originals[idx] = up_block.forward
            original = up_block.forward

            # Create patched forward with closure over original
            def make_patched(orig_fn):
                def patched_forward(hidden_states, res_hidden_states_tuple,
                                    *args, **kwargs):
                    # Zero ALL skip tensors received by this up_block
                    zeroed = tuple(torch.zeros_like(t) for t in res_hidden_states_tuple)
                    return orig_fn(hidden_states, zeroed, *args, **kwargs)
                return patched_forward

            up_block.forward = make_patched(original)
        return self

    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.unet.up_blocks[idx].forward = orig
        self._originals.clear()


# ---------------------------------------------------------------------------
# Model loading (same as phase1)
# ---------------------------------------------------------------------------

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_and_encode(pipe, path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2 * tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
    return latent, tensor


# ---------------------------------------------------------------------------
# DDIM inversion / reconstruction (copied from phase1 for self-containedness)
# ---------------------------------------------------------------------------

def ddim_inversion(pipe, latents, prompt_embeds, num_steps, guidance_scale=1.0):
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    z = latents.clone()
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]
            t_next = extended_ts[i - 1]

            noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]
            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur
            z = coeff1 * z + coeff2 * noise_pred

    return z


# ---------------------------------------------------------------------------
# UNet feature hooking (same as phase1)
# ---------------------------------------------------------------------------

def discover_hook_targets(unet):
    targets = []
    for name, module in unet.named_modules():
        parts = name.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(name)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(name)
    return sorted(targets)


class UNetFeatureHooker:
    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.handles = []

        targets = discover_hook_targets(unet)
        for name in targets:
            mod = self._find_module(name)
            if mod is not None:
                handle = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook_fn(n, out)
                )
                self.handles.append(handle)

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
        if output.dim() == 3:
            output = output.mean(dim=1, keepdim=True)
        self.features[name] = output.detach().cpu()

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# Per-layer drift analysis (adapted from phase1)
# ---------------------------------------------------------------------------

def analyze_layer_drift(pipe, original_latent, prompt_embeds, num_steps, seeds=None):
    """Full drift fingerprint: inversion → reconstruction → per-layer L2 comparison."""
    if seeds is None:
        seeds = [42]

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps
    n = len(timesteps)

    if n <= 6:
        key_indices = list(range(n))
    else:
        key_indices = [0, 1, 2, n // 2 - 1, n // 2, n // 2 + 1, n - 3, n - 2, n - 1]
    key_indices = sorted(set(max(0, min(n - 1, i)) for i in key_indices))

    hooker = UNetFeatureHooker(pipe.unet)
    embs = prompt_embeds

    # 1) Inversion (runs under modified topology if SkipIntervention active)
    inv_latent = ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)

    # 2) Reconstruction (runs under modified topology)
    recon_latents = [inv_latent.clone()]
    z = inv_latent.clone()
    with torch.no_grad():
        for _, t in enumerate(timesteps):
            noise_pred = pipe.unet(z, t, encoder_hidden_states=embs).sample
            z = scheduler.step(noise_pred, t, z).prev_sample
            recon_latents.append(z.clone())

    # 3) Feature comparison at key timesteps
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
# Main experiment
# ---------------------------------------------------------------------------

def get_coco_images():
    """Get all coco_val images."""
    if not COCO_VAL_DIR.exists():
        print(f"[WARN] {COCO_VAL_DIR} not found")
        return []
    return sorted([
        str(COCO_VAL_DIR / f) for f in os.listdir(COCO_VAL_DIR)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])


def run_condition(pipe, image_paths, condition_name, cut_indices, num_steps=50):
    """Run drift diagnosis under a skip-intervention condition.

    Args:
        pipe: SD pipeline
        image_paths: list of image paths
        condition_name: str label for output files
        cut_indices: list of up_block indices to cut (empty list = original)
        num_steps: DDIM steps

    Returns:
        dict: {image_name: {layer_name: drift_value}}
    """
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    all_drifts = {}  # img_name -> {layer: drift}

    for img_path in image_paths:
        img_name = Path(img_path).stem
        print(f"  [{condition_name}] {img_name}...", end=" ", flush=True)

        latent, _ = load_and_encode(pipe, img_path)

        if cut_indices:
            with SkipIntervention(pipe.unet, cut_indices):
                avg_drifts, std_drifts = analyze_layer_drift(
                    pipe, latent, prompt_embeds, num_steps, seeds=[42])
        else:
            avg_drifts, std_drifts = analyze_layer_drift(
                pipe, latent, prompt_embeds, num_steps, seeds=[42])

        if avg_drifts:
            all_drifts[img_name] = avg_drifts
            top_layer = sorted(avg_drifts.items(), key=lambda x: -x[1])[0]
            print(f"peak: {top_layer[0]}={top_layer[1]:.1f}")
        else:
            print("FAILED")

        torch.cuda.empty_cache()

    return all_drifts


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def aggregate_across_images(all_drifts):
    """Aggregate per-layer drift across images: mean, std per layer."""
    per_layer = defaultdict(list)
    for img_name, drifts in all_drifts.items():
        for layer, val in drifts.items():
            per_layer[layer].append(val)

    aggregated = {}
    for layer, vals in per_layer.items():
        aggregated[layer] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
    return aggregated


def paired_ttest_per_layer(drifts_a, drifts_b, common_images):
    """Paired t-test per layer between two conditions.

    Args:
        drifts_a: {img: {layer: val}}
        drifts_b: {img: {layer: val}}
        common_images: list of image names present in both

    Returns:
        {layer: {"t": float, "p": float, "mean_diff": float, "significant": bool}}
    """
    all_layers = set()
    for img in common_images:
        all_layers.update(drifts_a[img].keys())
        all_layers.update(drifts_b[img].keys())

    results = {}
    for layer in sorted(all_layers):
        vals_a = [drifts_a[img].get(layer, np.nan) for img in common_images]
        vals_b = [drifts_b[img].get(layer, np.nan) for img in common_images]

        # Remove pairs where either is NaN
        paired = [(a, b) for a, b in zip(vals_a, vals_b)
                  if not np.isnan(a) and not np.isnan(b)]
        if len(paired) < 3:
            continue

        a_arr = np.array([p[0] for p in paired])
        b_arr = np.array([p[1] for p in paired])

        t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
        results[layer] = {
            "t": float(t_stat),
            "p": float(p_val),
            "mean_diff": float(np.mean(b_arr - a_arr)),
            "mean_a": float(np.mean(a_arr)),
            "mean_b": float(np.mean(b_arr)),
            "significant": bool(p_val < 0.05),
            "n_pairs": len(paired),
        }
    return results


def compute_delta_map(aggregated_cut, aggregated_original):
    """Compute per-layer drift change: cut - original."""
    delta = {}
    for layer in aggregated_original:
        if layer in aggregated_cut:
            delta[layer] = aggregated_cut[layer]["mean"] - aggregated_original[layer]["mean"]
    return delta


def layer_sort_key(name):
    """Sort layers by UNet topology order."""
    for prefix in ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
                   "mid_block",
                   "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]:
        if name.startswith(prefix):
            return prefix + name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_single_fingerprint(aggregated, title, out_path, highlight_up=None):
    """Bar chart of per-layer drift."""
    names = sorted(aggregated.keys(), key=layer_sort_key)
    values = [aggregated[n]["mean"] for n in names]
    errors = [aggregated[n]["std"] for n in names]

    short_names = []
    for n in names:
        s = n.replace("down_blocks.", "D").replace("up_blocks.", "U") \
             .replace("mid_block.", "M").replace("resnets.", "R") \
             .replace("attentions.", "A").replace("transformer_blocks.", "T") \
             .replace(".", "")
        short_names.append(s)

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.55), 5))
    colors = []
    for n in names:
        if "down" in n:
            colors.append("#3498db")
        elif "mid" in n:
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    # Highlight the intervened up_block region
    if highlight_up is not None:
        colors_hl = []
        for n in names:
            if n.startswith(f"up_blocks.{highlight_up}"):
                colors_hl.append("#e74c3c")  # red for intervened region
            else:
                colors_hl.append(colors[len(colors_hl)])
        colors = colors_hl

    ax.bar(short_names, values, color=colors, yerr=errors, capsize=2,
           error_kw={"elinewidth": 0.5, "alpha": 0.5})
    ax.set_ylabel("Avg L2 Drift")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Down blocks"),
        Patch(facecolor="#e74c3c", label="Intervened (if red)" if highlight_up is not None else "Mid block"),
        Patch(facecolor="#2ecc71", label="Up blocks"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_map(delta, title, out_path, highlight_up=None):
    """Bar chart of per-layer drift change (delta)."""
    names = sorted(delta.keys(), key=layer_sort_key)
    values = [delta[n] for n in names]

    short_names = []
    for n in names:
        s = n.replace("down_blocks.", "D").replace("up_blocks.", "U") \
             .replace("mid_block.", "M").replace("resnets.", "R") \
             .replace("attentions.", "A").replace("transformer_blocks.", "T") \
             .replace(".", "")
        short_names.append(s)

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.55), 5))

    colors = []
    for n in names:
        v = delta[n]
        if highlight_up is not None and n.startswith(f"up_blocks.{highlight_up}"):
            colors.append("#e74c3c" if v > 0 else "#e67e22")
        elif v > 0:
            colors.append("#e74c3c")  # drift increased
        else:
            colors.append("#3498db")  # drift decreased

    ax.bar(short_names, values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Δ Drift (Cut - Original)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_three_way_comparison(agg_orig, agg_cut_a, agg_cut_b, out_path):
    """Three-panel figure: Original | Cut A | Cut B fingerprints."""
    names = sorted(agg_orig.keys(), key=layer_sort_key)

    short_names = []
    for n in names:
        s = n.replace("down_blocks.", "D").replace("up_blocks.", "U") \
             .replace("mid_block.", "M").replace("resnets.", "R") \
             .replace("attentions.", "A").replace("transformer_blocks.", "T") \
             .replace(".", "")
        short_names.append(s)

    fig, axes = plt.subplots(1, 3, figsize=(28, 6))

    conditions = [
        (agg_orig, "Original (no intervention)", None),
        (agg_cut_a, "Cut A: zero skip → up_blocks.2 (drift peak)", 2),
        (agg_cut_b, "Cut B: zero skip → up_blocks.0 (low drift)", 0),
    ]

    for ax, (agg, title, hl) in zip(axes, conditions):
        values = [agg[n]["mean"] for n in names]
        errors = [agg[n]["std"] for n in names]

        colors = []
        for n in names:
            if hl is not None and n.startswith(f"up_blocks.{hl}"):
                colors.append("#e74c3c")
            elif "down" in n:
                colors.append("#3498db")
            elif "mid" in n:
                colors.append("#f39c12")
            else:
                colors.append("#2ecc71")

        ax.bar(short_names, values, color=colors, width=0.8)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.grid(axis="y", alpha=0.3)

        # Mark top-3 layers
        ranked = sorted(zip(names, values), key=lambda x: -x[1])[:3]
        for layer_name, val in ranked:
            idx = names.index(layer_name)
            ax.annotate(f"#{ranked.index((layer_name, val))+1}",
                       (idx, val), fontsize=6, ha="center", va="bottom",
                       color="darkred")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Three-way comparison → {out_path}")


def plot_delta_comparison(delta_a, delta_b, out_path):
    """Side-by-side delta maps: Cut A vs Cut B."""
    names = sorted(delta_a.keys(), key=layer_sort_key)

    short_names = []
    for n in names:
        s = n.replace("down_blocks.", "D").replace("up_blocks.", "U") \
             .replace("mid_block.", "M").replace("resnets.", "R") \
             .replace("attentions.", "A").replace("transformer_blocks.", "T") \
             .replace(".", "")
        short_names.append(s)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(names) * 0.55), 9))

    for ax, (delta, title, hl) in zip(axes, [
        (delta_a, "Δ Drift: Cut A (up_blocks.2 skip cut) - Original", 2),
        (delta_b, "Δ Drift: Cut B (up_blocks.0 skip cut) - Original", 0),
    ]):
        values = [delta.get(n, 0) for n in names]
        colors = []
        for n in names:
            v = delta.get(n, 0)
            if hl is not None and n.startswith(f"up_blocks.{hl}"):
                colors.append("#c0392b" if v > 0 else "#e67e22")
            elif v > 0:
                colors.append("#e74c3c")
            else:
                colors.append("#3498db")

        ax.bar(short_names, values, color=colors)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Δ Drift")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Delta comparison → {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(agg_orig, agg_cut_a, agg_cut_b, ttest_a, ttest_b,
                 delta_a, delta_b, common_images):
    """Print structured report."""
    print(f"\n{'='*70}")
    print("SKIP INTERVENTION RESULTS")
    print(f"  Images: {len(common_images)}")
    print(f"{'='*70}")

    # Top-5 drift layers in original
    ranked_orig = sorted(agg_orig.items(), key=lambda x: -x[1]["mean"])

    print("\n--- Original Top-5 Drift Layers ---")
    for i, (layer, v) in enumerate(ranked_orig[:5]):
        print(f"  {i+1}. {layer:<50s} drift={v['mean']:.1f} ± {v['std']:.1f}")

    # Cut A: layers with largest drift increase
    print("\n--- Cut A: Top-5 Drift Increases ---")
    ranked_delta_a = sorted(delta_a.items(), key=lambda x: -x[1])
    for i, (layer, d) in enumerate(ranked_delta_a[:5]):
        orig_val = agg_orig[layer]["mean"]
        cut_val = agg_cut_a[layer]["mean"]
        p_val = ttest_a.get(layer, {}).get("p", 1.0)
        sig = "*" if p_val < 0.05 else ""
        print(f"  {i+1}. {layer:<50s} Δ=+{d:.1f} "
              f"(orig={orig_val:.1f} → cut={cut_val:.1f}) p={p_val:.4f} {sig}")

    # Cut B: layers with largest drift increase
    print("\n--- Cut B: Top-5 Drift Increases ---")
    ranked_delta_b = sorted(delta_b.items(), key=lambda x: -x[1])
    for i, (layer, d) in enumerate(ranked_delta_b[:5]):
        orig_val = agg_orig[layer]["mean"]
        cut_val = agg_cut_b[layer]["mean"]
        p_val = ttest_b.get(layer, {}).get("p", 1.0)
        sig = "*" if p_val < 0.05 else ""
        print(f"  {i+1}. {layer:<50s} Δ=+{d:.1f} "
              f"(orig={orig_val:.1f} → cut={cut_val:.1f}) p={p_val:.4f} {sig}")

    # Key question: are the delta maps different?
    # High positive correlation → both cuts produce similar patterns → capacity effect
    # Low or negative correlation → different spatial patterns → topology effect
    common_layers = sorted(set(delta_a.keys()) & set(delta_b.keys()))
    da_vals = [delta_a[l] for l in common_layers]
    db_vals = [delta_b[l] for l in common_layers]
    delta_corr = np.corrcoef(da_vals, db_vals)[0, 1]
    print(f"\n--- Spatial correlation of Δ maps ---")
    print(f"  Pearson r(Δ_CutA, Δ_CutB) = {delta_corr:.3f}")
    if delta_corr > 0.5:
        print(f"  → CAPACITY dominant: both cuts produce similar spatial patterns")
    elif delta_corr < 0:
        print(f"  → TOPOLOGY dominant: different cuts produce ANTI-CORRELATED patterns")
    else:
        print(f"  → TOPOLOGY dominant: different cuts produce weakly correlated patterns")

    # Count significant layers
    n_sig_a = sum(1 for v in ttest_a.values() if v["significant"])
    n_sig_b = sum(1 for v in ttest_b.values() if v["significant"])
    print(f"\n--- Significant layers (p < 0.05, paired t-test) ---")
    print(f"  Cut A vs Original: {n_sig_a}/{len(ttest_a)} layers significant")
    print(f"  Cut B vs Original: {n_sig_b}/{len(ttest_b)} layers significant")

    # Topology vs capacity summary
    print(f"\n{'='*70}")
    print("TOPOLOGY vs CAPACITY ASSESSMENT")
    print(f"{'='*70}")
    print(f"""
    Both Cut A and Cut B remove one skip connection (same "capacity" loss),
    but at different topological locations.

    If the effect were purely capacity-driven:
      → Δ maps should be highly correlated (same pattern everywhere)

    If the effect has a topological component:
      → Δ maps should have LOW correlation (different spatial patterns)
      → Each cut should LOCALIZE its effect near the cut site

    Observed: r = {delta_corr:.3f}
    → {'TOPOLOGY dominates: different cuts produce different (anti-correlated) spatial patterns' if delta_corr < 0.3
       else 'Weak evidence: run on full 19 images for stable estimate'}
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Skip Connection Intervention Experiment")
    parser.add_argument("--images", type=str, nargs="+", default=None,
                        help="Specific images to test (default: all coco_val)")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM steps (default: 50)")
    parser.add_argument("--quick", type=int, default=None,
                        help="Quick test on N images (default: all 19)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.images:
        image_paths = args.images
    else:
        image_paths = get_coco_images()

    if args.quick:
        image_paths = image_paths[:args.quick]

    print(f"[Setup] {len(image_paths)} images, {args.steps} steps")
    print(f"[Output] {OUT_DIR.resolve()}")

    # Load model
    print("[0] Loading SD 1.5...")
    pipe = load_pipeline()

    # Run three conditions
    print(f"\n[1] Running ORIGINAL (no intervention)...")
    drifts_orig = run_condition(pipe, image_paths, "original", [], args.steps)

    print(f"\n[2] Running Cut A (zero skip → up_blocks.2, peak location)...")
    drifts_cut_a = run_condition(pipe, image_paths, "cut_a", [2], args.steps)

    print(f"\n[3] Running Cut B (zero skip → up_blocks.0, low-drift location)...")
    drifts_cut_b = run_condition(pipe, image_paths, "cut_b", [0], args.steps)

    # Aggregate across images
    common = sorted(set(drifts_orig.keys()) & set(drifts_cut_a.keys()) & set(drifts_cut_b.keys()))
    print(f"\n[4] Aggregating across {len(common)} common images...")

    agg_orig = aggregate_across_images({k: drifts_orig[k] for k in common})
    agg_cut_a = aggregate_across_images({k: drifts_cut_a[k] for k in common})
    agg_cut_b = aggregate_across_images({k: drifts_cut_b[k] for k in common})

    # Statistical tests
    print("[5] Paired t-tests...")
    ttest_a = paired_ttest_per_layer(drifts_orig, drifts_cut_a, common)
    ttest_b = paired_ttest_per_layer(drifts_orig, drifts_cut_b, common)

    # Delta maps
    delta_a = compute_delta_map(agg_cut_a, agg_orig)
    delta_b = compute_delta_map(agg_cut_b, agg_orig)

    # Report
    print_report(agg_orig, agg_cut_a, agg_cut_b, ttest_a, ttest_b,
                 delta_a, delta_b, common)

    # Visualizations
    print("\n[6] Generating figures...")
    plot_three_way_comparison(agg_orig, agg_cut_a, agg_cut_b,
                              OUT_DIR / "three_way_fingerprint.png")
    plot_delta_comparison(delta_a, delta_b,
                          OUT_DIR / "delta_comparison.png")

    # Individual fingerprints
    plot_single_fingerprint(agg_orig, "Original (no intervention)",
                            OUT_DIR / "fingerprint_original.png")
    plot_single_fingerprint(agg_cut_a, "Cut A: zero skip → up_blocks.2",
                            OUT_DIR / "fingerprint_cut_a.png", highlight_up=2)
    plot_single_fingerprint(agg_cut_b, "Cut B: zero skip → up_blocks.0",
                            OUT_DIR / "fingerprint_cut_b.png", highlight_up=0)

    # Save full data
    print("\n[7] Saving data...")
    results = {
        "config": {
            "cut_a": {"description": "Zero skip from down_blocks.1 to up_blocks.2 (peak location)",
                      "cut_up_indices": [2]},
            "cut_b": {"description": "Zero skip from down_blocks.3 to up_blocks.0 (low-drift location)",
                      "cut_up_indices": [0]},
            "n_images": len(common),
            "images": common,
            "steps": args.steps,
        },
        "aggregated": {
            "original": {k: v for k, v in agg_orig.items()},
            "cut_a": {k: v for k, v in agg_cut_a.items()},
            "cut_b": {k: v for k, v in agg_cut_b.items()},
        },
        "delta": {
            "cut_a_minus_original": {k: float(v) for k, v in delta_a.items()},
            "cut_b_minus_original": {k: float(v) for k, v in delta_b.items()},
        },
        "ttest_cut_a_vs_original": {k: v for k, v in ttest_a.items()},
        "ttest_cut_b_vs_original": {k: v for k, v in ttest_b.items()},
        "delta_spatial_correlation": float(np.corrcoef(
            [delta_a[l] for l in sorted(set(delta_a.keys()) & set(delta_b.keys()))],
            [delta_b[l] for l in sorted(set(delta_a.keys()) & set(delta_b.keys()))]
        )[0, 1]),
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Full results → {OUT_DIR / 'results.json'}")

    print(f"\n{'='*60}")
    print("Skip intervention experiment complete.")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
