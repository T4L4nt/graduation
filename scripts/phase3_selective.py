"""
Phase 3: Selective Layer Correction via Phase 1 Diagnostic Guidance.

Core idea: Phase 1 layer-wise drift diagnosis reveals semantic decomposition
of the UNet hierarchy. We use this to selectively correct only structural
layers, allowing textural/editing layers to remain free.

Strategies derived from Phase 1 drift + Phase 4 info theory:
  - full:       top-5 drift layers (current Phase 2 default)
  - structural: high-ΔPSNR layers — encoder shallow + decoder deep
  - textural:   high-drift decoder layers — mid-upstream layers
  - minimal:    single best layer only
  - none:       DDIM baseline (no correction)

Hypothesis: selective correction produces a controllable Pareto frontier
between content preservation and editing freedom, without any additional
style injection mechanism.

Usage: python scripts/phase3_selective.py --n-images 19
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_image, decode_latent, compute_metrics,
    ddim_inversion, ddim_inversion_with_features, ddim_reconstruction,
    ddim_reconstruction_with_correction, FeatureCorrector,
    get_top_drift_layers, save_recon_img, make_grid_image,
)
from phase3_prep import CLIPFeatureExtractor, run_baseline, run_correction_only

OUT_DIR = Path("outputs/phase3_selective")

# ---------------------------------------------------------------------------
# Layer strategies — derived from Phase 1/4 diagnostics
# ---------------------------------------------------------------------------

# Structural: high ΔPSNR = most correctable = encoder shallow + decoder deep
# These layers encode edges, layout, object boundaries
STRUCTURAL_LAYERS = [
    "down_blocks.0.resnets.0",   # ΔPSNR 2.79 dB (highest), dim=4
    "down_blocks.0.resnets.1",   # ΔPSNR 2.75 dB
    "up_blocks.3.resnets.0",     # ΔPSNR 2.75 dB
    "up_blocks.3.resnets.1",     # ΔPSNR 2.78 dB
    "up_blocks.3.resnets.2",     # ΔPSNR 2.61 dB, dim=2 (tightest manifold)
]

# Textural: high drift decoder layers — these dominate the reconstruction appearance
TEXTURAL_LAYERS = [
    "up_blocks.2.resnets.0",     # drift 2.97 (highest)
    "up_blocks.2.resnets.1",     # drift 0.99
    "up_blocks.2.resnets.2",     # ΔPSNR 2.70 dB
    "up_blocks.1.resnets.1",     # drift 1.83
    "up_blocks.0.resnets.0",     # drift 2.21
]

# Full: top-k drift (current Phase 2 default) — loaded dynamically
FULL_LAYERS = get_top_drift_layers(5)

# Minimal: single best ΔPSNR layer
MINIMAL_LAYERS = ["down_blocks.0.resnets.0"]

# Baseline: no correction
NONE_LAYERS = []

STRATEGIES = {
    "full":       ("Full Correction (top-5)",          FULL_LAYERS),
    "structural": ("Structure-Only (encoder+deep)",     STRUCTURAL_LAYERS),
    "textural":   ("Texture-Only (decoder mid)",        TEXTURAL_LAYERS),
    "minimal":    ("Single Best Layer",                 MINIMAL_LAYERS),
    "none":       ("DDIM Baseline (no correction)",     NONE_LAYERS),
}

COLORS = {"full": "#2ecc71", "structural": "#3498db", "textural": "#e74c3c",
          "minimal": "#f39c12", "none": "#95a5a6"}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_strategy_experiment(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    num_steps: int = 50,
    corr_lam: float = 0.7,
    lpips_fn=None,
) -> Dict[str, List[Dict]]:
    """Run all strategies on all images."""
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")
    all_results = {name: [] for name in STRATEGIES}

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image_from_tensor(original_tensor)

        print(f"\n[Selective] {img_name}")

        # Collect features from the UNION of all strategy layers during inversion
        all_layers = list(set(sum((layers for _, (_, layers) in STRATEGIES.items() if layers), [])))
        print(f"  Collecting features from {len(all_layers)} unique layers...")
        noise, saved = ddim_inversion_with_features(
            pipe, original_latent, prompt_embeds, num_steps, all_layers)

        for strat_name, (label, layers) in STRATEGIES.items():
            if not layers:
                m, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                        prompt_embeds, num_steps, lpips_fn)
            else:
                t0 = time.perf_counter()
                corrector = FeatureCorrector(pipe.unet, layers, corr_lam)
                corrector.set_reference(saved, 0)
                recon_latent = ddim_reconstruction_with_correction(
                    pipe, noise, prompt_embeds, num_steps, saved, corrector)
                recon = decode_latent(pipe, recon_latent)
                corrector.remove()
                m = compute_metrics(original_tensor, recon, lpips_fn)

            # CLIP metrics from reconstruction
            recon_for_clip = decode_latent(pipe, recon_latent) if layers else decode_latent(
                pipe, ddim_reconstruction(pipe,
                    ddim_inversion(pipe, original_latent, prompt_embeds, num_steps),
                    prompt_embeds, num_steps))
            v_recon = extractor.encode_image_from_tensor(recon_for_clip)

            m["CLIP_content"] = float((v_recon * v_orig).sum())

            # Style similarity to various text styles: measure editing freedom
            style_scores = {}
            for style_name, style_text in [
                ("oil_painting", "an oil painting in impressionist style"),
                ("watercolor", "a watercolor sketch with soft brushstrokes"),
                ("cyberpunk", "a neon-lit cyberpunk scene"),
            ]:
                v_style_target = extractor.encode_text(style_text)
                style_scores[style_name] = float((v_recon * v_style_target).sum())

            m["CLIP_style"] = max(style_scores.values())  # best style match
            m["image"] = img_name
            m["strategy"] = strat_name
            m["n_layers"] = len(layers)
            m["label"] = label
            all_results[strat_name].append(m)

            print(f"  {strat_name:12s} ({len(layers)} layers): "
                  f"PSNR={m['PSNR']:.1f} LPIPS={m['LPIPS']:.3f} "
                  f"CLIP_s={m['CLIP_style']:.3f} CLIP_c={m['CLIP_content']:.3f}")
            torch.cuda.empty_cache()

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_flat = [r for results in all_results.values() for r in results]
    with open(OUT_DIR / "selective_results.json", "w") as f:
        json.dump(all_flat, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psnr_comparison(results: Dict[str, List[Dict]], output_path: str):
    """Bar chart: PSNR per strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(STRATEGIES.keys())
    means = [np.mean([r["PSNR"] for r in results[n]]) for n in names]
    stds = [np.std([r["PSNR"] for r in results[n]]) for n in names]
    colors = [COLORS[n] for n in names]
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("PSNR (dB)"); ax.set_title("Selective Layer Correction: PSNR")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close()
    print(f"[Figure] {output_path}")


def plot_clip_pareto(results: Dict[str, List[Dict]], output_path: str):
    """CLIP_content vs CLIP_style — editing freedom Pareto."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name in STRATEGIES:
        xs = [r["CLIP_content"] for r in results[name]]
        ys = [r["CLIP_style"] for r in results[name]]
        ax.scatter(xs, ys, c=COLORS[name], label=STRATEGIES[name][0],
                   s=60, alpha=0.8, edgecolors="white")
        if len(xs) >= 2:
            ax.annotate(name, (np.mean(xs), np.mean(ys)), fontsize=9,
                        ha="center", va="bottom",
                        color=COLORS[name], fontweight="bold")
    ax.set_xlabel("CLIP Content Preservation")
    ax.set_ylabel("Best CLIP Style Similarity")
    ax.set_title("Selective Correction: Content-Style Trade-off")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close()
    print(f"[Figure] {output_path}")


def plot_strategy_grid(results: Dict[str, List[Dict]], output_path: str):
    """Grid plot: PSNR, LPIPS, CLIP_c, CLIP_s per strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    names = list(STRATEGIES.keys())
    metrics = [("PSNR", "PSNR (dB)", axes[0,0]),
               ("LPIPS", "LPIPS (lower=better)", axes[0,1]),
               ("CLIP_content", "CLIP Content", axes[1,0]),
               ("CLIP_style", "CLIP Style (editing freedom)", axes[1,1])]
    for metric, ylabel, ax in metrics:
        means = [np.mean([r[metric] for r in results[n]]) for n in names]
        ax.bar(names, means, color=[COLORS[n] for n in names])
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Selective Layer Correction: Diagnostic-Driven Editing", fontsize=14)
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close()
    print(f"[Figure] {output_path}")


def generate_table(results: Dict[str, List[Dict]], output_path: str):
    """LaTeX table."""
    names = list(STRATEGIES.keys())
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Selective Layer Correction: Diagnostic-Guided Editing}" + "\n")
        f.write(r"\label{tab:selective_correction}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Strategy & Layers & PSNR$\uparrow$ & LPIPS$\downarrow$ & "
                r"CLIP$_c$$\uparrow$ & CLIP$_s$$\uparrow$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for name in names:
            entries = results[name]
            psnr = np.mean([e["PSNR"] for e in entries])
            lp = np.mean([e["LPIPS"] for e in entries])
            cc = np.mean([e["CLIP_content"] for e in entries])
            cs = np.mean([e["CLIP_style"] for e in entries])
            n = entries[0]["n_layers"] if entries else 0
            label = STRATEGIES[name][0]
            f.write(f"  {label} & {n} & {psnr:.2f} & {lp:.3f} & {cc:.3f} & {cs:.3f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[Table] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Selective Layer Correction")
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=19)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.7)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[Phase 3 Selective] {len(images)} images, {args.steps} steps, λ={args.corr_lam}")
    print(f"Strategies: {list(STRATEGIES.keys())}")
    for name, (label, layers) in STRATEGIES.items():
        print(f"  {name:12s}: {len(layers)} layers — {label}")

    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)

    results = run_strategy_experiment(
        pipe, images, extractor,
        num_steps=args.steps, corr_lam=args.corr_lam, lpips_fn=lpips_fn,
    )

    plot_psnr_comparison(results, str(OUT_DIR / "psnr_comparison.png"))
    plot_clip_pareto(results, str(OUT_DIR / "clip_pareto.png"))
    plot_strategy_grid(results, str(OUT_DIR / "strategy_grid.png"))
    generate_table(results, str(OUT_DIR / "selective_table.tex"))

    # Summary
    print(f"\n{'='*60}")
    print("Selective Correction Summary")
    print(f"{'Strategy':12s} {'Layers':>6s} {'PSNR':>7s} {'LPIPS':>7s} "
          f"{'CLIP_c':>7s} {'CLIP_s':>7s}")
    print("-" * 55)
    for name in STRATEGIES:
        entries = results[name]
        psnr = np.mean([e["PSNR"] for e in entries])
        lp = np.mean([e["LPIPS"] for e in entries])
        cc = np.mean([e["CLIP_content"] for e in entries])
        cs = np.mean([e["CLIP_style"] for e in entries])
        n = entries[0]["n_layers"] if entries else 0
        print(f"  {name:12s} {n:6d} {psnr:7.2f} {lp:7.3f} {cc:7.3f} {cs:7.3f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
