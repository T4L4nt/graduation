"""
DCSC Stability: Numerical verification of content drift boundedness.

Proposition: Under DCSC control law, content drift d_content(t) satisfies:
    d_content(t) <= C * lambda_0 / Kp    where C = L * M

Assumptions:
    1. CLIP encoder is L-Lipschitz on [-1, +1] image space
    2. Feature residuals bounded: ||f_inv - f_recon|| <= M
    3. P control law: lambda(t) = lambda_0 * max(0, 1 - Kp * d_content(t))

Usage:
  python scripts/dcsc_stability.py --n-images 5
"""

import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_image, decode_latent,
    get_top_drift_layers,
)
from phase3_prep import CLIPFeatureExtractor, build_style_cross_attn_tokens
from dcsc_core import DCSCStyleController, dcsc_controlled_generation

OUT_DIR = Path("outputs/dcsc/stability")


# ---------------------------------------------------------------------------
# CLIP Lipschitz estimation
# ---------------------------------------------------------------------------

def estimate_lipschitz_constant(
    extractor: CLIPFeatureExtractor,
    pipe,
    images: List[str],
    n_pairs: int = 100,
    epsilon: float = 0.01,
) -> float:
    """Estimate CLIP Lipschitz constant L empirically.

    For each image, create epsilon-perturbed version, measure
    ||CLIP(x) - CLIP(x+eps)|| / ||eps||, return maximum ratio.
    """
    ratios = []
    rng = np.random.RandomState(42)

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        _, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image_from_tensor(original_tensor)

        # Multiple random perturbations
        n_per_image = max(1, n_pairs // len(images) + 1)
        for _ in range(n_per_image):
            noise = torch.randn_like(original_tensor) * epsilon
            perturbed = (original_tensor + noise).clamp(-1, 1)
            v_pert = extractor.encode_image_from_tensor(perturbed)

            pixel_diff = (noise.float().norm()).item()
            clip_diff = (v_orig - v_pert).float().norm().item()

            if pixel_diff > 1e-8:
                ratios.append(clip_diff / pixel_diff)

    if not ratios:
        return 1.0

    ratios_arr = np.array(ratios)
    L = float(np.max(ratios_arr))
    print(f"  CLIP Lipschitz estimate: L={L:.4f}  "
          f"(mean={np.mean(ratios_arr):.4f}, median={np.median(ratios_arr):.4f})")
    return L


# ---------------------------------------------------------------------------
# Drift bound verification
# ---------------------------------------------------------------------------

def verify_bounded_drift(
    pipe,
    images: List[str],
    extractor: CLIPFeatureExtractor,
    num_steps: int = 50,
    corr_lam: float = 0.5,
    Kp_values: List[float] = None,
    lambda_0_values: List[float] = None,
    control_freq: int = 5,
    lpips_fn=None,
    L_estimate: float = None,
) -> Dict:
    """Run DCSC, verify theoretical bound for all (Kp, lambda_0) combos.

    Returns dict with per-parameter bound check results.
    """
    if Kp_values is None:
        Kp_values = [0.5, 1.0, 2.0, 5.0]
    if lambda_0_values is None:
        lambda_0_values = [0.3, 0.5, 0.7]

    # Estimate L if not provided
    if L_estimate is None:
        L_estimate = estimate_lipschitz_constant(extractor, pipe, images)

    corr_layers = get_top_drift_layers(5)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_content = extractor.encode_text("a photo")

    verification_results = []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)
        v_orig = extractor.encode_image(img_path)
        _, v_style, _ = extractor.compute_orthogonal_decomposition(v_orig, v_content)

        print(f"\n[Stability] {img_name}")

        for lam0 in lambda_0_values:
            for kp in Kp_values:
                print(f"  λ_0={lam0:.2f} Kp={kp:.1f}...", end=" ", flush=True)

                metrics, recon, elapsed, traj = dcsc_controlled_generation(
                    pipe, original_latent, original_tensor, prompt_embeds,
                    num_steps=num_steps, corr_lam=corr_lam, corr_layers=corr_layers,
                    v_style=v_style, v_content=v_content, extractor=extractor,
                    lambda_0=lam0, Kp=kp, control_freq=control_freq,
                    style_mode="extra_token", lpips_fn=lpips_fn,
                )

                # Extract empirical bound data
                d_content_values = traj["trajectory"]["d_content_raw"]
                max_d = max(d_content_values) if d_content_values else 0.0
                theoretical_bound = traj["stability_bound"]

                # Check if bound holds
                bound_satisfied = max_d <= theoretical_bound * 1.5  # 50% tolerance

                print(f"max_d={max_d:.4f} bound={theoretical_bound:.4f} "
                      f"{'OK' if bound_satisfied else 'VIOLATED'}")

                verification_results.append({
                    "image": img_name, "Kp": kp, "lambda_0": lam0,
                    "max_d_content": max_d,
                    "theoretical_bound": theoretical_bound,
                    "bound_satisfied": bound_satisfied,
                    "n_control_calls": traj["n_control_calls"],
                    "final_basis_size": traj["final_basis_size"],
                    "L_estimate": L_estimate,
                })

                torch.cuda.empty_cache()

    # Summary
    n_total = len(verification_results)
    n_ok = sum(1 for r in verification_results if r["bound_satisfied"])
    print(f"\n[Stability Summary] {n_ok}/{n_total} bound checks passed "
          f"({100*n_ok/n_total:.1f}%)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "bound_verification.json", "w") as f:
        json.dump({
            "L_estimate": L_estimate,
            "results": verification_results,
            "pass_rate": n_ok / n_total if n_total > 0 else 0,
        }, f, indent=2)

    return {
        "L_estimate": L_estimate,
        "results": verification_results,
        "pass_rate": n_ok / n_total if n_total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_drift_bounds(bound_results: Dict, output_path: str):
    """Max observed d_content vs theoretical bound for each (Kp, lambda_0).

    X-axis: Kp (log scale). Y-axis: max d_content.
    Lines: different lambda_0. Dashed: theoretical bounds.
    """
    results = bound_results["results"]
    L = bound_results["L_estimate"]
    fig, ax = plt.subplots(figsize=(10, 6))

    lam0_values = sorted(set(r["lambda_0"] for r in results))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lam0_values)))

    for lam0, c in zip(lam0_values, colors):
        subset = [r for r in results if r["lambda_0"] == lam0]
        subset.sort(key=lambda r: r["Kp"])
        kps = [r["Kp"] for r in subset]
        max_ds = [r["max_d_content"] for r in subset]
        bounds = [r["theoretical_bound"] for r in subset]

        ax.plot(kps, max_ds, "o-", color=c, markersize=6, label=f"λ_0={lam0:.1f} (empirical)")
        ax.plot(kps, bounds, "--", color=c, alpha=0.5, linewidth=1.5,
                label=f"λ_0={lam0:.1f} (bound)")

    ax.set_xlabel("Kp (Proportional Gain)")
    ax.set_ylabel("Max Content Drift d_content")
    ax.set_title("DCSC Stability: Empirical Drift vs Theoretical Bound")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_drift_trajectories(
    bound_results: Dict,
    output_path: str,
    max_plots: int = 6,
):
    """d_content(t) time series with theoretical bound overlay."""
    results = bound_results["results"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Pick diverse (Kp, lambda_0) combos
    import itertools
    combos = list(itertools.product(
        sorted(set(r["lambda_0"] for r in results)),
        sorted(set(r["Kp"] for r in results)),
    ))
    selected = combos[:max_plots]

    for idx, (lam0, kp) in enumerate(selected):
        ax = axes[idx]
        subset = [r for r in results if r["lambda_0"] == lam0 and r["Kp"] == kp]
        for r in subset:
            ax.axhline(y=r["theoretical_bound"], color="red", linestyle=":",
                       label=f"Bound={r['theoretical_bound']:.4f}" if idx == 0 else "")
            ax.set_title(f"λ_0={lam0:.1f}, Kp={kp:.1f}")
            ax.set_xlabel("Step"); ax.set_ylabel("d_content")
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("DCSC Content Drift Trajectories", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_contraction_factors(bound_results: Dict, output_path: str):
    """Per-step contraction factor: d_content(t+1) / d_content(t)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    results = bound_results["results"]
    lam0_values = sorted(set(r["lambda_0"] for r in results))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lam0_values)))

    for lam0, c in zip(lam0_values, colors):
        subset = [r for r in results if r["lambda_0"] == lam0]
        subset.sort(key=lambda r: r["Kp"])
        kps = [r["Kp"] for r in subset]

        # Average bound/theoretical ratio
        ratios = [r["max_d_content"] / max(r["theoretical_bound"], 1e-8) for r in subset]
        ax.plot(kps, ratios, "o-", color=c, markersize=6, label=f"λ_0={lam0:.1f}")

    ax.axhline(y=1.0, color="red", linestyle=":", label="Bound=Empirical")
    ax.set_xlabel("Kp")
    ax.set_ylabel("Empirical/Bound Ratio")
    ax.set_title("Tightness of Stability Bound")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.5)
    parser.add_argument("--Kp", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--lambda-0", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[DCSC Stability] {len(images)} images, {args.steps} steps")

    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)

    # Estimate Lipschitz
    L = estimate_lipschitz_constant(extractor, pipe, images)

    # Verify bounds
    bound_results = verify_bounded_drift(
        pipe, images, extractor,
        num_steps=args.steps, corr_lam=args.corr_lam,
        Kp_values=args.Kp, lambda_0_values=args.lambda_0,
        control_freq=args.control_freq, lpips_fn=lpips_fn,
        L_estimate=L,
    )

    # Plot
    plot_drift_bounds(bound_results, str(OUT_DIR / "drift_bounds.png"))
    plot_drift_trajectories(bound_results, str(OUT_DIR / "drift_trajectories.png"))
    plot_contraction_factors(bound_results, str(OUT_DIR / "bound_tightness.png"))

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
