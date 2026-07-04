"""
DCSC Experiment: Pareto frontier evaluation, ablation, and cross-method comparison.

Usage:
  # Pareto scan on 5 images
  python scripts/dcsc_experiment.py --mode pareto --n-images 5

  # Subspace ablation
  python scripts/dcsc_experiment.py --mode ablation --n-images 5

  # Full comparison across methods
  python scripts/dcsc_experiment.py --mode compare --n-images 19
"""

import argparse, json, sys, os, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_image, decode_latent, compute_metrics,
    ddim_inversion, ddim_inversion_with_features, ddim_reconstruction,
    ddim_reconstruction_with_correction,
    FeatureCorrector, LambdaScheduler, get_top_drift_layers, save_recon_img,
)
from phase3_prep import (
    CLIPFeatureExtractor, build_style_cross_attn_tokens,
    find_closest_style, STYLE_CANDIDATES,
    run_baseline, run_correction_only, run_correction_with_style,
    run_correction_with_style_and_pinning,
)
from dcsc_core import DCSCStyleController, dcsc_controlled_generation

OUT_DIR = Path("outputs/dcsc")

# ---------------------------------------------------------------------------
# Pareto Scan
# ---------------------------------------------------------------------------

def pareto_scan(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    Kp_values: Optional[List[float]] = None,
    lambda_0_values: Optional[List[float]] = None,
    control_freq: int = 5,
    lpips_fn=None,
) -> List[Dict]:
    """Full Pareto scan over DCSC control parameters.

    For each image, for each Kp × lambda_0 combination:
      Run dcsc_controlled_generation, record all metrics + trajectory.

    Returns list of result dicts.
    """
    if Kp_values is None:
        Kp_values = [0.5, 1.0, 2.0, 5.0]
    if lambda_0_values is None:
        lambda_0_values = [0.3, 0.5, 0.7]
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)

    all_results = []
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)

        # Compute style direction
        v_orig = extractor.encode_image(img_path)
        _, v_style, _ = extractor.compute_orthogonal_decomposition(v_orig, v_content)

        print(f"\n{'='*60}\n[Pareto] {img_name}")

        # Baseline
        m_base, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                     prompt_embeds, num_steps, lpips_fn)
        print(f"  Baseline: PSNR={m_base['PSNR']:.2f}")

        # Correction only
        m_corr, _, _ = run_correction_only(pipe, original_latent, original_tensor,
                                            prompt_embeds, num_steps, corr_lam,
                                            corr_layers, lpips_fn)
        print(f"  Correction: PSNR={m_corr['PSNR']:.2f}  Δ={m_corr['PSNR']-m_base['PSNR']:+.2f}")

        # DCSC scan
        for lam0 in lambda_0_values:
            for kp in Kp_values:
                tag = f"dcsc_Kp{kp:.1f}_lam{lam0:.2f}"
                print(f"  {tag}...", end=" ", flush=True)

                try:
                    metrics, recon, elapsed, traj = dcsc_controlled_generation(
                        pipe, original_latent, original_tensor, prompt_embeds,
                        num_steps=num_steps, corr_lam=corr_lam,
                        corr_layers=corr_layers,
                        v_style=v_style, v_content=v_content,
                        extractor=extractor,
                        lambda_0=lam0, Kp=kp, control_freq=control_freq,
                        style_mode="extra_token", lpips_fn=lpips_fn,
                    )
                    delta_psnr = metrics["PSNR"] - m_base["PSNR"]
                    print(f"PSNR={metrics['PSNR']:.2f} Δ={delta_psnr:+.2f} "
                          f"CLIP_s={metrics['CLIP_style']:.3f} "
                          f"CLIP_c={metrics['CLIP_content']:.3f} "
                          f"final_λ={traj['final_lambda']:.3f} "
                          f"bound={traj['stability_bound']:.4f}")

                    all_results.append({
                        "image": img_name, "method": "DCSC",
                        "Kp": kp, "lambda_0": lam0,
                        **metrics,
                        "delta_psnr": delta_psnr,
                        "elapsed_s": elapsed,
                        "stability_bound": traj["stability_bound"],
                        "final_lambda": traj["final_lambda"],
                        "n_control_calls": traj["n_control_calls"],
                        "final_basis_size": traj["final_basis_size"],
                    })
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback; traceback.print_exc()

                torch.cuda.empty_cache()

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pareto_dir = OUT_DIR / "pareto_scan"
    pareto_dir.mkdir(exist_ok=True)
    with open(pareto_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ---------------------------------------------------------------------------
# Pareto Frontier Computation
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    points: List[Dict],
    content_key: str = "CLIP_content",
    style_key: str = "CLIP_style",
) -> List[Dict]:
    """Return Pareto-optimal (non-dominated) points for two-objective maximization."""
    if not points:
        return []
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (q[content_key] >= p[content_key] and q[style_key] >= p[style_key]
                    and (q[content_key] > p[content_key] or q[style_key] > p[style_key])):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda p: p[style_key])


def compute_pareto_dominance_ratio(dcsc_points, baseline_points,
                                    content_key="CLIP_content",
                                    style_key="CLIP_style") -> float:
    """Fraction of baseline points dominated by at least one DCSC point."""
    if not baseline_points:
        return 0.0
    dominated = 0
    for bp in baseline_points:
        for dp in dcsc_points:
            if (dp[content_key] >= bp[content_key] and dp[style_key] >= bp[style_key]
                    and (dp[content_key] > bp[content_key] or dp[style_key] > bp[style_key])):
                dominated += 1
                break
    return dominated / len(baseline_points)


# ---------------------------------------------------------------------------
# Subspace Ablation
# ---------------------------------------------------------------------------

def run_ablation_subspace(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    lambda_0: float = 0.5,
    Kp: float = 1.0,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    control_freq: int = 5,
    lpips_fn=None,
) -> Tuple[List[Dict], List[Dict]]:
    """Ablation: DCSC with subspace vs without subspace (P-control only)."""
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")

    results_with = []
    results_without = []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image(img_path)
        _, v_style, _ = extractor.compute_orthogonal_decomposition(v_orig, v_content)

        print(f"\n[Ablation] {img_name}")

        # With subspace
        print("  With subspace...", end=" ", flush=True)
        m_w, _, _, _ = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
            v_style=v_style, v_content=v_content, extractor=extractor,
            lambda_0=lambda_0, Kp=Kp, control_freq=control_freq,
            style_mode="extra_token", lpips_fn=lpips_fn,
            use_subspace=True,
        )
        m_w["image"] = img_name; m_w["ablation"] = "with_subspace"
        print(f"PSNR={m_w['PSNR']:.2f} CLIP_s={m_w['CLIP_style']:.3f} CLIP_c={m_w['CLIP_content']:.3f}")
        results_with.append(m_w)

        # Without subspace
        print("  Without subspace...", end=" ", flush=True)
        m_wo, _, _, _ = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
            v_style=v_style, v_content=v_content, extractor=extractor,
            lambda_0=lambda_0, Kp=Kp, control_freq=control_freq,
            style_mode="extra_token", lpips_fn=lpips_fn,
            use_subspace=False,
        )
        m_wo["image"] = img_name; m_wo["ablation"] = "without_subspace"
        print(f"PSNR={m_wo['PSNR']:.2f} CLIP_s={m_wo['CLIP_style']:.3f} CLIP_c={m_wo['CLIP_content']:.3f}")
        results_without.append(m_wo)

        torch.cuda.empty_cache()

    # Save
    ablation_dir = OUT_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    with open(ablation_dir / "ablation_subspace.json", "w") as f:
        json.dump({"with_subspace": results_with, "without_subspace": results_without}, f, indent=2)

    return results_with, results_without


# ---------------------------------------------------------------------------
# Cross-Method Comparison
# ---------------------------------------------------------------------------

def compare_across_methods(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    lpips_fn=None,
    # DCSC best params (set after Pareto scan)
    dcsc_Kp: float = 1.0,
    dcsc_lambda_0: float = 0.5,
    # Phase 3 pinning params
    pinning_lambdas: List[float] = None,
    # P2P lambdas
    p2p_lambdas: List[float] = None,
) -> Dict[str, List[Dict]]:
    """Run all methods at specified parameter points for direct comparison."""
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)
    if pinning_lambdas is None:
        pinning_lambdas = [0.3, 0.5, 0.7]
    if p2p_lambdas is None:
        p2p_lambdas = [0.3, 0.5, 0.7]

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")

    results: Dict[str, List[Dict]] = {
        "DDIM": [], "Correction": [], "StyleOnly": [],
        "Phase3_Pinning": [], "DCSC": [],
    }

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image(img_path)
        _, v_style, _ = extractor.compute_orthogonal_decomposition(v_orig, v_content)

        print(f"\n[Compare] {img_name}")

        # 1. DDIM baseline
        m, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                prompt_embeds, num_steps, lpips_fn)
        m["image"] = img_name; m["method"] = "DDIM"
        results["DDIM"].append(m)
        print(f"  DDIM: PSNR={m['PSNR']:.2f}")

        # 2. Correction only
        m, _, _ = run_correction_only(pipe, original_latent, original_tensor,
                                       prompt_embeds, num_steps, corr_lam,
                                       corr_layers, lpips_fn)
        m["image"] = img_name; m["method"] = "Correction"
        v_corr = extractor.encode_image_from_tensor(
            decode_latent(pipe, ddim_inversion(pipe, original_latent, prompt_embeds, num_steps)))
        m["CLIP_style"] = float((v_corr * v_style).sum())
        m["CLIP_content"] = float((v_corr * v_orig).sum())
        results["Correction"].append(m)
        print(f"  Correction: PSNR={m['PSNR']:.2f}")

        # 3. Style only (no correction, no pinning)
        for lam in pinning_lambdas:
            styled_emb = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=lam, mode="extra_token")
            m, recon, _ = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                num_steps, 0.0, corr_layers,  # corr_lam=0 → no correction
                styled_emb, lpips_fn=lpips_fn, style_injector=None,
            )
            v_recon = extractor.encode_image_from_tensor(recon)
            m["CLIP_style"] = float((v_recon * v_style).sum())
            m["CLIP_content"] = float((v_recon * v_orig).sum())
            m["image"] = img_name; m["method"] = f"StyleOnly_λ{lam:.1f}"
            m["style_lambda"] = lam
            results["StyleOnly"].append(m)
            print(f"  StyleOnly λ={lam:.1f}: PSNR={m['PSNR']:.2f} "
                  f"CLIP_s={m['CLIP_style']:.3f} CLIP_c={m['CLIP_content']:.3f}")

        # 4. Phase 3 pinning
        for lam in pinning_lambdas:
            styled_emb = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=lam, mode="extra_token")
            m, recon, _, pin_log = run_correction_with_style_and_pinning(
                pipe, original_latent, original_tensor, prompt_embeds,
                num_steps, corr_lam, corr_layers,
                styled_emb, extractor, v_content,
                lpips_fn=lpips_fn, style_injector=None,
                pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5,
            )
            v_recon = extractor.encode_image_from_tensor(recon)
            m["CLIP_style"] = float((v_recon * v_style).sum())
            m["CLIP_content"] = float((v_recon * v_orig).sum())
            m["image"] = img_name; m["method"] = f"Phase3Pin_λ{lam:.1f}"
            m["style_lambda"] = lam
            results["Phase3_Pinning"].append(m)
            triggered = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
            print(f"  Phase3Pin λ={lam:.1f}: PSNR={m['PSNR']:.2f}  "
                  f"triggered={triggered}/{len(pin_log) if pin_log else 0}")

        # 5. DCSC at optimal params
        m, recon, elapsed, traj = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
            v_style=v_style, v_content=v_content, extractor=extractor,
            lambda_0=dcsc_lambda_0, Kp=dcsc_Kp, control_freq=5,
            style_mode="extra_token", lpips_fn=lpips_fn,
        )
        m["image"] = img_name; m["method"] = f"DCSC_Kp{dcsc_Kp:.1f}_λ{dcsc_lambda_0:.2f}"
        results["DCSC"].append(m)
        print(f"  DCSC: PSNR={m['PSNR']:.2f}  CLIP_s={m['CLIP_style']:.3f}  "
              f"final_λ={traj['final_lambda']:.3f}  basis={traj['final_basis_size']}")

        torch.cuda.empty_cache()

    # Save
    compare_dir = OUT_DIR / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    # Convert to serializable format
    serializable = {}
    for method, entries in results.items():
        serializable[method] = []
        for e in entries:
            serializable[method].append({
                k: (float(v) if isinstance(v, (torch.Tensor, np.floating, np.integer)) else v)
                for k, v in e.items()
                if isinstance(v, (int, float, str, bool)) or (
                    isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 0)
            })
    with open(compare_dir / "comparison_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    all_points: Dict[str, List[Dict]],
    output_path: str,
    title: str = "Style-Content Pareto Frontier",
):
    """2D scatter with Pareto frontier curves for each method."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"DCSC": "#e74c3c", "Phase3_Pinning": "#3498db", "StyleOnly": "#95a5a6",
              "P2P": "#2ecc71", "Correction": "#9b59b6", "DDIM": "#34495e"}
    markers = {"DCSC": "o", "Phase3_Pinning": "s", "StyleOnly": "^",
               "P2P": "D", "Correction": "v", "DDIM": "X"}

    for method, points in all_points.items():
        if not points:
            continue
        xs = [p["CLIP_content"] for p in points if p.get("CLIP_content") is not None]
        ys = [p["CLIP_style"] for p in points if p.get("CLIP_style") is not None]
        if not xs:
            continue
        c = colors.get(method, "#333333")
        m = markers.get(method, ".")
        ax.scatter(xs, ys, c=c, marker=m, label=method, s=50, alpha=0.8, edgecolors="white")

        # Compute and draw Pareto frontier
        pareto = compute_pareto_frontier(points)
        if len(pareto) >= 2:
            px = [p["CLIP_content"] for p in pareto]
            py = [p["CLIP_style"] for p in pareto]
            ax.plot(px, py, c=c, linewidth=2, alpha=0.5, linestyle="--")

            # Highlight Pareto-optimal points
            ax.scatter(px, py, c=c, marker=m, s=120, edgecolors="black", linewidth=1.5, zorder=5)

    ax.set_xlabel("CLIP Content Preservation (higher = better)")
    ax.set_ylabel("CLIP Style Similarity (higher = better)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_control_trajectory(
    trajectory: Dict,
    output_path: str,
    stability_bound: Optional[float] = None,
):
    """Time-series: lambda(t), d_content(t), basis_size(t)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    steps = trajectory.get("steps", [])
    if not steps:
        print("[WARN] Empty trajectory, skipping plot")
        plt.close()
        return

    # Panel 1: lambda
    ax = axes[0]
    ax.plot(steps, trajectory["lambda"], "b-o", markersize=4)
    ax.set_ylabel("λ_style(t)")
    ax.set_title("DCSC Control Trajectory")
    ax.axhline(y=trajectory.get("lambda_0", 0.5), color="gray", linestyle=":", label="λ_0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: d_content
    ax = axes[1]
    ax.plot(steps, trajectory["d_content_raw"], "r-o", markersize=4, label="raw")
    if trajectory.get("d_content_smooth"):
        ax.plot(steps, trajectory["d_content_smooth"], "orange", linestyle="--", label="EMA smooth")
    if stability_bound is not None:
        ax.axhline(y=stability_bound, color="red", linestyle=":", linewidth=1.5,
                   label=f"Bound={stability_bound:.4f}")
    ax.set_ylabel("d_content(t)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: basis size
    ax = axes[2]
    ax.plot(steps, trajectory["basis_size"], "g-s", markersize=4)
    ax.set_ylabel("Basis Size")
    ax.set_xlabel("Denoising Step")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def generate_comparison_table(results: Dict[str, List[Dict]], output_path: str):
    """Generate LaTeX comparison table."""
    methods_order = ["DDIM", "Correction", "StyleOnly", "Phase3_Pinning", "DCSC"]

    # Aggregate per-method averages
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{DCSC vs Baselines: Content Preservation and Style Transfer}" + "\n")
        f.write(r"\label{tab:dcsc_comparison}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & PSNR$\uparrow$ & LPIPS$\downarrow$ & CLIP$_c$$\uparrow$ & CLIP$_s$$\uparrow$ \\" + "\n")
        f.write(r"\midrule" + "\n")

        for method in methods_order:
            entries = results.get(method, [])
            if not entries:
                continue
            psnr = np.mean([e["PSNR"] for e in entries if e.get("PSNR") is not None])
            lpips_v = np.mean([e["LPIPS"] for e in entries if e.get("LPIPS") is not None])
            clip_c = np.mean([e.get("CLIP_content", 0) for e in entries])
            clip_s = np.mean([e.get("CLIP_style", 0) for e in entries])
            f.write(f"  {method} & {psnr:.2f} & {lpips_v:.3f} & {clip_c:.3f} & {clip_s:.3f} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[Table] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pareto",
                        choices=["pareto", "ablation", "compare"])
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.5)
    parser.add_argument("--Kp", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--lambda-0", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--dcsc-Kp", type=float, default=1.0,
                        help="Optimal Kp for cross-method comparison")
    parser.add_argument("--dcsc-lambda-0", type=float, default=0.5,
                        help="Optimal lambda_0 for cross-method comparison")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine images
    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[DCSC Experiment] mode={args.mode}, {len(images)} images, {args.steps} steps")

    # Load
    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)
    corr_layers = get_top_drift_layers(5)

    if args.mode == "pareto":
        results = pareto_scan(
            pipe, images, extractor,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            Kp_values=args.Kp, lambda_0_values=args.lambda_0,
            control_freq=args.control_freq, lpips_fn=lpips_fn,
        )
        # Plot Pareto frontier
        by_method = {"DCSC": results}
        plot_pareto_frontier(
            by_method, str(OUT_DIR / "pareto_scan" / "pareto_frontier.png"),
            title=f"DCSC Pareto Frontier ({len(images)} images)")

    elif args.mode == "ablation":
        results_w, results_wo = run_ablation_subspace(
            pipe, images, extractor,
            lambda_0=args.dcsc_lambda_0, Kp=args.dcsc_Kp,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            control_freq=args.control_freq, lpips_fn=lpips_fn,
        )
        # Summary
        for label, res in [("With Subspace", results_w), ("Without Subspace", results_wo)]:
            psnr = np.mean([r["PSNR"] for r in res])
            cs = np.mean([r.get("CLIP_style", 0) for r in res])
            cc = np.mean([r.get("CLIP_content", 0) for r in res])
            print(f"  {label}: PSNR={psnr:.2f} CLIP_s={cs:.3f} CLIP_c={cc:.3f}")

    elif args.mode == "compare":
        results = compare_across_methods(
            pipe, images, extractor,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            lpips_fn=lpips_fn,
            dcsc_Kp=args.dcsc_Kp, dcsc_lambda_0=args.dcsc_lambda_0,
        )
        # Plot
        plot_pareto_frontier(
            results, str(OUT_DIR / "comparison" / "comparison_pareto.png"),
            title=f"Style-Content Pareto Frontier ({len(images)} images)")
        generate_comparison_table(
            results, str(OUT_DIR / "comparison" / "comparison_table.tex"))

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
