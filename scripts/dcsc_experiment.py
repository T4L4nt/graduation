"""
DCSC Experiment: Fair comparison with external style targets.

Key design:
  1. Style is EXTERNALLY defined (text prompt or reference image), not from
     self-decomposition of the content image.
  2. All methods share the same style target AND style-strength budget.
  3. Comparison metric: Pareto frontier dominance ratio, not single-point.

Usage:
  python scripts/dcsc_experiment.py --mode pareto --style-text "oil painting"
  python scripts/dcsc_experiment.py --mode compare --style-text "watercolor sketch"
  python scripts/dcsc_experiment.py --mode ablation --style-text "oil painting"
"""

import argparse, json, os, sys, time
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
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    FeatureCorrector, LambdaScheduler, get_top_drift_layers,
)
from phase3_prep import (
    CLIPFeatureExtractor, build_style_cross_attn_tokens,
    run_baseline, run_correction_only, run_correction_with_style,
    run_correction_with_style_and_pinning,
)
from dcsc_core import (
    resolve_style_direction, DCSCStyleController, dcsc_controlled_generation,
)

OUT_DIR = Path("outputs/dcsc")

# Style candidates for experiments (text-driven, externally defined)
STYLE_TEXTS = [
    "an oil painting in impressionist style",
    "a watercolor sketch with soft brushstrokes",
    "a neon-lit cyberpunk scene",
]

# Style strength budget — all methods sweep the SAME range
STYLE_BUDGETS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Helpers: compute CLIP metrics from reconstruction tensor
# ---------------------------------------------------------------------------

def _add_clip_metrics(metrics: Dict, recon: torch.Tensor,
                      extractor: CLIPFeatureExtractor,
                      v_orig: torch.Tensor, v_style: torch.Tensor):
    """Add CLIP_style / CLIP_content to a metrics dict from actual recon."""
    v_recon = extractor.encode_image_from_tensor(recon)
    metrics["CLIP_content"] = float((v_recon * v_orig).sum())
    metrics["CLIP_style"] = float((v_recon * v_style).sum())


# ---------------------------------------------------------------------------
# Pareto Scan
# ---------------------------------------------------------------------------

def pareto_scan(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    style_text: str,
    v_content: torch.Tensor,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    Kp_values: Optional[List[float]] = None,
    lambda_0_values: Optional[List[float]] = None,
    control_freq: int = 5,
    lpips_fn=None,
) -> List[Dict]:
    """Pareto scan over DCSC control parameters with external style target."""
    if Kp_values is None:
        Kp_values = [0.5, 1.0, 2.0, 5.0]
    if lambda_0_values is None:
        lambda_0_values = [0.3, 0.5, 0.7]
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)

    all_results = []
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image_from_tensor(original_tensor)

        # Baseline
        m_base, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                     prompt_embeds, num_steps, lpips_fn)
        print(f"\n[Pareto] {img_name}  baseline PSNR={m_base['PSNR']:.2f}")

        # Correction only
        m_corr, recon_corr, _ = run_correction_only(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps, corr_lam, corr_layers, lpips_fn)
        _add_clip_metrics(m_corr, recon_corr, extractor, v_orig,
                          resolve_style_direction(extractor, v_content, style_text=style_text)[0])
        m_corr["image"] = img_name; m_corr["method"] = "Correction"
        all_results.append(m_corr)

        # DCSC scan
        for lam0 in lambda_0_values:
            for kp in Kp_values:
                print(f"  DCSC λ_0={lam0:.1f} Kp={kp:.1f}...", end=" ", flush=True)
                try:
                    metrics, recon, elapsed, traj = dcsc_controlled_generation(
                        pipe, original_latent, original_tensor, prompt_embeds,
                        num_steps=num_steps, corr_lam=corr_lam,
                        corr_layers=corr_layers,
                        extractor=extractor, v_content=v_content,
                        style_text=style_text,
                        lambda_0=lam0, Kp=kp, control_freq=control_freq,
                        style_mode="extra_token", lpips_fn=lpips_fn,
                    )
                    metrics["image"] = img_name
                    metrics["delta_psnr"] = metrics["PSNR"] - m_base["PSNR"]
                    metrics["elapsed_s"] = elapsed
                    metrics["stability_ratio"] = traj["empirical_stability_ratio"]
                    metrics["final_lambda"] = traj["final_lambda"]
                    metrics["n_control_calls"] = traj["n_control_calls"]
                    metrics["final_basis_size"] = traj["final_basis_size"]
                    all_results.append(metrics)
                    print(f"PSNR={metrics['PSNR']:.2f} "
                          f"CLIP_s={metrics['CLIP_style']:.3f} "
                          f"CLIP_c={metrics['CLIP_content']:.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                torch.cuda.empty_cache()

    # Save
    pareto_dir = OUT_DIR / "pareto_scan"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    with open(pareto_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Pareto Frontier Computation
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    points: List[Dict],
    content_key: str = "CLIP_content",
    style_key: str = "CLIP_style",
) -> List[Dict]:
    """Return Pareto-optimal (non-dominated) points. Both metrics are maximized."""
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
    # Sort by style ascending for curve plotting
    return sorted(pareto, key=lambda p: p.get(style_key, 0))


def compute_pareto_area(points: List[Dict],
                        content_key="CLIP_content",
                        style_key="CLIP_style") -> float:
    """Area under the Pareto frontier (trapezoidal rule). Larger = better."""
    pareto = compute_pareto_frontier(points, content_key, style_key)
    if len(pareto) < 2:
        return 0.0
    xs = [p[content_key] for p in pareto]
    ys = [p[style_key] for p in pareto]
    # Sort by content (x-axis) for area computation
    sorted_pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    area = 0.0
    for i in range(len(sorted_pairs) - 1):
        dx = sorted_pairs[i+1][0] - sorted_pairs[i][0]
        area += dx * (sorted_pairs[i][1] + sorted_pairs[i+1][1]) / 2
    return area


def compute_dominance_ratio(method_points: List[Dict],
                            baseline_points: List[Dict],
                            content_key="CLIP_content",
                            style_key="CLIP_style") -> float:
    """Fraction of baseline points dominated by >=1 method point."""
    if not baseline_points:
        return 0.0
    dominated = 0
    for bp in baseline_points:
        for mp in method_points:
            if (mp.get(content_key, 0) >= bp.get(content_key, 0)
                    and mp.get(style_key, 0) >= bp.get(style_key, 0)
                    and (mp.get(content_key, 0) > bp.get(content_key, 0)
                         or mp.get(style_key, 0) > bp.get(style_key, 0))):
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
    style_text: str,
    v_content: torch.Tensor,
    lambda_0: float = 0.5,
    Kp: float = 1.0,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    control_freq: int = 5,
    lpips_fn=None,
) -> Tuple[List[Dict], List[Dict]]:
    """Ablation: DCSC with subspace projection vs P-control only."""
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    results_with, results_without = [], []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)

        print(f"\n[Ablation] {img_name}")

        # With subspace
        print("  With subspace...", end=" ", flush=True)
        m_w, _, _, _ = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
            extractor=extractor, v_content=v_content, style_text=style_text,
            lambda_0=lambda_0, Kp=Kp, control_freq=control_freq,
            style_mode="extra_token", lpips_fn=lpips_fn, use_subspace=True,
        )
        m_w["image"] = img_name; m_w["ablation"] = "with_subspace"
        print(f"PSNR={m_w['PSNR']:.2f} CLIP_s={m_w['CLIP_style']:.3f}")
        results_with.append(m_w)

        # Without subspace
        print("  Without subspace...", end=" ", flush=True)
        m_wo, _, _, _ = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
            extractor=extractor, v_content=v_content, style_text=style_text,
            lambda_0=lambda_0, Kp=Kp, control_freq=control_freq,
            style_mode="extra_token", lpips_fn=lpips_fn, use_subspace=False,
        )
        m_wo["image"] = img_name; m_wo["ablation"] = "without_subspace"
        print(f"PSNR={m_wo['PSNR']:.2f} CLIP_s={m_wo['CLIP_style']:.3f}")
        results_without.append(m_wo)
        torch.cuda.empty_cache()

    ablation_dir = OUT_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    with open(ablation_dir / "ablation_subspace.json", "w") as f:
        json.dump({"with_subspace": results_with, "without_subspace": results_without},
                  f, indent=2, default=str)
    return results_with, results_without


# ---------------------------------------------------------------------------
# Fair Cross-Method Comparison
# ---------------------------------------------------------------------------

def compare_across_methods(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    style_text: str,
    v_content: torch.Tensor,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    corr_layers: Optional[List[str]] = None,
    lpips_fn=None,
    dcsc_Kp: float = 1.0,
    dcsc_lambda_0: float = 0.5,
) -> Dict[str, List[Dict]]:
    """Fair comparison: all methods share same style target + same strength budget.

    Each method sweeps STYLE_BUDGETS and records (CLIP_content, CLIP_style).
    DCSC uses extra budget points (Kp × λ_0) since its control is adaptive.
    """
    if corr_layers is None:
        corr_layers = get_top_drift_layers(5)

    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_style, style_label = resolve_style_direction(extractor, v_content, style_text=style_text)

    results: Dict[str, List[Dict]] = {
        "DDIM": [], "Correction": [], "StyleOnly": [],
        "Phase3_Pinning": [], "DCSC": [],
    }

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image_from_tensor(original_tensor)

        print(f"\n[Compare] {img_name}  style='{style_label}'")

        # ---- DDIM baseline ----
        m, recon, _ = run_baseline(pipe, original_latent, original_tensor,
                                    prompt_embeds, num_steps, lpips_fn)
        _add_clip_metrics(m, recon, extractor, v_orig, v_style)
        m["image"] = img_name; m["method"] = "DDIM"; m["style_budget"] = 0.0
        results["DDIM"].append(m)
        print(f"  DDIM:          PSNR={m['PSNR']:.2f}  CLIP_s={m['CLIP_style']:.3f}  CLIP_c={m['CLIP_content']:.3f}")

        # ---- Correction only (no style) ----
        m, recon, _ = run_correction_only(pipe, original_latent, original_tensor,
                                           prompt_embeds, num_steps, corr_lam,
                                           corr_layers, lpips_fn)
        _add_clip_metrics(m, recon, extractor, v_orig, v_style)
        m["image"] = img_name; m["method"] = "Correction"; m["style_budget"] = 0.0
        results["Correction"].append(m)
        print(f"  Correction:    PSNR={m['PSNR']:.2f}  CLIP_s={m['CLIP_style']:.3f}  CLIP_c={m['CLIP_content']:.3f}")

        # ---- StyleOnly (no correction, just style injection) ----
        for budget in STYLE_BUDGETS:
            styled_emb = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=budget, mode="extra_token")
            m, recon, _ = run_correction_with_style(
                pipe, original_latent, original_tensor, prompt_embeds,
                num_steps, 0.0, corr_layers,  # corr_lam=0 → no correction
                styled_emb, lpips_fn=lpips_fn, style_injector=None,
            )
            _add_clip_metrics(m, recon, extractor, v_orig, v_style)
            m["image"] = img_name; m["method"] = f"StyleOnly_b{budget:.1f}"
            m["style_budget"] = budget
            results["StyleOnly"].append(m)

        # ---- Phase 3 pinning (correction + hard-threshold pinning) ----
        for budget in STYLE_BUDGETS:
            styled_emb = build_style_cross_attn_tokens(
                pipe, "", v_style, strength=budget, mode="extra_token")
            m, recon, _, pin_log = run_correction_with_style_and_pinning(
                pipe, original_latent, original_tensor, prompt_embeds,
                num_steps, corr_lam, corr_layers,
                styled_emb, extractor, v_content,
                lpips_fn=lpips_fn, style_injector=None,
                pinning_freq=5, pinning_threshold=0.02, pinning_strength=0.5,
            )
            _add_clip_metrics(m, recon, extractor, v_orig, v_style)
            m["image"] = img_name; m["method"] = f"Phase3Pin_b{budget:.1f}"
            m["style_budget"] = budget
            triggered = sum(1 for d in pin_log if d[2] > 0.02) if pin_log else 0
            m["pinning_triggered"] = triggered
            results["Phase3_Pinning"].append(m)

        # ---- DCSC ----
        for budget in STYLE_BUDGETS:
            m, recon, _, traj = dcsc_controlled_generation(
                pipe, original_latent, original_tensor, prompt_embeds,
                num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
                extractor=extractor, v_content=v_content, style_text=style_text,
                lambda_0=budget, Kp=dcsc_Kp, control_freq=5,
                style_mode="extra_token", lpips_fn=lpips_fn,
            )
            m["image"] = img_name; m["method"] = f"DCSC_b{budget:.1f}"
            m["style_budget"] = budget
            m["stability_ratio"] = traj["empirical_stability_ratio"]
            results["DCSC"].append(m)

        torch.cuda.empty_cache()

    # Save
    compare_dir = OUT_DIR / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for method, entries in results.items():
        serializable[method] = []
        for e in entries:
            clean = {}
            for k, v in e.items():
                if isinstance(v, (int, float, str, bool)):
                    clean[k] = v
                elif isinstance(v, (torch.Tensor,)):
                    clean[k] = float(v.detach().cpu().item())
                elif isinstance(v, np.floating):
                    clean[k] = float(v)
            serializable[method].append(clean)
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
    report_areas: bool = True,
):
    """Multi-method Pareto frontier with area/dominance annotations."""
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"DCSC": "#e74c3c", "Phase3_Pinning": "#3498db", "StyleOnly": "#95a5a6",
              "Correction": "#9b59b6", "DDIM": "#34495e"}
    markers = {"DCSC": "o", "Phase3_Pinning": "s", "StyleOnly": "^",
               "Correction": "v", "DDIM": "X"}
    method_order = ["DCSC", "Phase3_Pinning", "StyleOnly", "Correction", "DDIM"]

    for method in method_order:
        points = all_points.get(method, [])
        if not points:
            continue
        xs = [p.get("CLIP_content", 0) for p in points]
        ys = [p.get("CLIP_style", 0) for p in points]
        if not xs:
            continue
        c = colors.get(method, "#333")
        m = markers.get(method, ".")
        ax.scatter(xs, ys, c=c, marker=m, label=method, s=50, alpha=0.7, edgecolors="white")

        # Pareto frontier curve
        pareto = compute_pareto_frontier(points)
        if len(pareto) >= 2:
            px = [p["CLIP_content"] for p in pareto]
            py = [p["CLIP_style"] for p in pareto]
            ax.plot(px, py, c=c, linewidth=2, alpha=0.5, linestyle="--")
            ax.scatter(px, py, c=c, marker=m, s=100, edgecolors="black", linewidth=1.2, zorder=5)

    # Annotate with Pareto areas
    if report_areas:
        y_ann = 0.02
        for method in method_order:
            points = all_points.get(method, [])
            if len(points) >= 2:
                area = compute_pareto_area(points)
                ax.text(0.98, y_ann, f"{method} area={area:.4f}", transform=ax.transAxes,
                        fontsize=8, ha="right", color=colors.get(method, "#333"))
                y_ann += 0.035

    # Dominance ratio annotation
    dcsc_pts = all_points.get("DCSC", [])
    pinning_pts = all_points.get("Phase3_Pinning", [])
    if dcsc_pts and pinning_pts:
        dom = compute_dominance_ratio(dcsc_pts, pinning_pts)
        ax.text(0.98, 0.95, f"DCSC dominates Phase3: {dom:.0%}", transform=ax.transAxes,
                fontsize=9, ha="right", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("CLIP Content Preservation (higher = better)")
    ax.set_ylabel("CLIP Style Similarity (higher = better)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_control_trajectory(trajectory: Dict, output_path: str):
    """Time-series: lambda(t), d_content(t), basis_size(t)."""
    steps = trajectory.get("steps", [])
    if not steps:
        print("[WARN] Empty trajectory"); return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, trajectory["lambda"], "b-o", markersize=4)
    axes[0].set_ylabel("λ_style(t)"); axes[0].set_title("DCSC Control Trajectory")
    axes[0].axhline(y=trajectory.get("lambda_0", 0.5), color="gray", linestyle=":")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, trajectory["d_content_raw"], "r-o", markersize=4, label="raw")
    if trajectory.get("d_content_smooth"):
        axes[1].plot(steps, trajectory["d_content_smooth"], "orange", linestyle="--", label="EMA")
    axes[1].set_ylabel("d_content(t)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    axes[2].plot(steps, trajectory["basis_size"], "g-s", markersize=4)
    axes[2].set_ylabel("Basis Size"); axes[2].set_xlabel("Denoising Step")
    axes[2].set_ylim(bottom=0); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def generate_comparison_table(results: Dict[str, List[Dict]], output_path: str):
    """LaTeX comparison table."""
    methods_order = ["DDIM", "Correction", "StyleOnly", "Phase3_Pinning", "DCSC"]
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{DCSC Fair Comparison: Equal Style-Strength Budget}" + "\n")
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
            lp = np.mean([e.get("LPIPS", 0) for e in entries])
            cc = np.mean([e.get("CLIP_content", 0) for e in entries])
            cs = np.mean([e.get("CLIP_style", 0) for e in entries])
            f.write(f"  {method} & {psnr:.2f} & {lp:.3f} & {cc:.3f} & {cs:.3f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[Table] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DCSC Fair Comparison Experiments")
    parser.add_argument("--mode", type=str, default="pareto",
                        choices=["pareto", "ablation", "compare"])
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.5)
    parser.add_argument("--style-text", type=str, default="an oil painting in impressionist style",
                        help="External style TARGET text prompt")
    parser.add_argument("--style-ref", type=str, default=None,
                        help="External style reference image (overrides --style-text)")
    parser.add_argument("--Kp", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--lambda-0", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--dcsc-Kp", type=float, default=1.0)
    parser.add_argument("--dcsc-lambda-0", type=float, default=0.5)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    style_text = args.style_ref if args.style_ref else args.style_text
    style_is_ref = args.style_ref is not None
    print(f"[DCSC Experiment] mode={args.mode}  images={len(images)}  "
          f"style={'ref:'+style_text if style_is_ref else style_text}")

    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)
    corr_layers = get_top_drift_layers(5)
    v_content = extractor.encode_text("a photo")

    if args.mode == "pareto":
        results = pareto_scan(
            pipe, images, extractor, style_text, v_content,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            Kp_values=args.Kp, lambda_0_values=args.lambda_0,
            control_freq=args.control_freq, lpips_fn=lpips_fn,
        )
        plot_pareto_frontier(
            {"DCSC": results}, str(OUT_DIR / "pareto_scan" / "pareto_frontier.png"),
            title=f"DCSC Pareto Frontier — style: '{style_text[:40]}...'")

        # Print Pareto area
        area = compute_pareto_area(results)
        pareto_pts = compute_pareto_frontier(results)
        print(f"\n[Pareto] area={area:.4f}  non_dominated={len(pareto_pts)}/{len(results)}")

    elif args.mode == "ablation":
        results_w, results_wo = run_ablation_subspace(
            pipe, images, extractor, style_text, v_content,
            lambda_0=args.dcsc_lambda_0, Kp=args.dcsc_Kp,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            control_freq=args.control_freq, lpips_fn=lpips_fn,
        )
        for label, res in [("With Subspace", results_w), ("Without Subspace", results_wo)]:
            psnr = np.mean([r["PSNR"] for r in res])
            cs = np.mean([r.get("CLIP_style", 0) for r in res])
            cc = np.mean([r.get("CLIP_content", 0) for r in res])
            print(f"  {label}: PSNR={psnr:.2f} CLIP_s={cs:.3f} CLIP_c={cc:.3f}")

    elif args.mode == "compare":
        results = compare_across_methods(
            pipe, images, extractor, style_text, v_content,
            num_steps=args.steps, corr_lam=args.corr_lam, corr_layers=corr_layers,
            lpips_fn=lpips_fn,
            dcsc_Kp=args.dcsc_Kp, dcsc_lambda_0=args.dcsc_lambda_0,
        )
        plot_pareto_frontier(
            results, str(OUT_DIR / "comparison" / "comparison_pareto.png"),
            title=f"Style-Content Pareto Frontier — style: '{style_text[:40]}...'")
        generate_comparison_table(
            results, str(OUT_DIR / "comparison" / "comparison_table.tex"))

        # Print Pareto metrics
        for method in ["DCSC", "Phase3_Pinning", "StyleOnly"]:
            pts = results.get(method, [])
            if pts:
                area = compute_pareto_area(pts)
                print(f"  {method}: Pareto area={area:.4f}")

        dcsc_pts = results.get("DCSC", [])
        pinning_pts = results.get("Phase3_Pinning", [])
        if dcsc_pts and pinning_pts:
            dom = compute_dominance_ratio(dcsc_pts, pinning_pts)
            print(f"  DCSC dominates Phase3 Pinning: {dom:.0%}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
