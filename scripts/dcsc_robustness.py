"""
DCSC Robustness: Editing Drift Boundedness Evaluation.

Narrative: DCSC is a drift-bounded editing controller. The user specifies an
editing freedom budget σ ∈ [0,1]. This maps to correction strength:
    λ_eff = λ_max * (1 - σ)
σ=0 → full content preservation (λ=λ_max). σ=1 → no correction (DDIM baseline).

DCSC monitors CLIP content drift d(t). When drift is detected, it INCREASES
correction to protect content:
    λ_eff(t) = min(λ_max, λ_max*(1-σ) + Kp * d(t))

Experiment: sweep σ and compare 3 control modes:
  - open_loop: λ fixed at λ_max*(1-σ), drift may grow
  - phase3_pin: hard-threshold λ boost when drift detected
  - dcsc: continuous P-control λ boost

Key metric: max_content_drift vs editing_strength.
Prediction: open_loop drift grows with σ; DCSC drift stays bounded.

Usage: python scripts/dcsc_robustness.py --n-images 5
"""

import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, load_pipeline, load_image, get_top_drift_layers,
)
from phase3_prep import CLIPFeatureExtractor, run_baseline, run_correction_only
from dcsc_core import drift_bounded_generation

OUT_DIR = Path("outputs/dcsc")

# Editing freedom values to sweep
EDITING_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
CONTROL_MODES = ["open_loop", "phase3_pin", "dcsc"]

COLORS = {"open_loop": "#e74c3c", "phase3_pin": "#3498db", "dcsc": "#2ecc71"}
MARKERS = {"open_loop": "s", "phase3_pin": "^", "dcsc": "o"}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_robustness_experiment(
    pipe, images, extractor, v_content,
    num_steps=50, corr_lam=0.7,
    editing_values=None, Kp=1.0, lpips_fn=None,
) -> List[Dict]:
    if editing_values is None:
        editing_values = EDITING_VALUES
    corr_layers = get_top_drift_layers(5)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    all_results = []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)

        # Baselines (σ=0 → full correction)
        m_base, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                     prompt_embeds, num_steps, lpips_fn)
        m_corr, _, _ = run_correction_only(pipe, original_latent, original_tensor,
                                            prompt_embeds, num_steps, corr_lam,
                                            corr_layers, lpips_fn)
        print(f"\n[Robustness] {img_name}  DDIM={m_base['PSNR']:.1f}  Corr={m_corr['PSNR']:.1f}")

        for sigma in editing_values:
            for mode in CONTROL_MODES:
                if sigma == 0.0 and mode != CONTROL_MODES[0]:
                    continue  # only one baseline at σ=0
                print(f"  σ={sigma:.2f} mode={mode}...", end=" ", flush=True)
                try:
                    metrics, recon, elapsed, traj = drift_bounded_generation(
                        pipe, original_latent, original_tensor, prompt_embeds,
                        num_steps=num_steps, corr_lam=corr_lam,
                        corr_layers=corr_layers,
                        extractor=extractor, v_content=v_content,
                        editing_strength=sigma,
                        control_mode=mode, Kp=Kp, control_freq=5,
                        lpips_fn=lpips_fn,
                    )
                    result = {
                        "image": img_name, "editing_strength": sigma,
                        "control_mode": mode, "Kp": Kp,
                        "PSNR": metrics["PSNR"],
                        "LPIPS": metrics["LPIPS"],
                        "SSIM": metrics["SSIM"],
                        "CLIP_content": metrics["CLIP_content"],
                        "max_content_drift": traj["max_content_drift"],
                        "final_lambda_corr": traj["final_lambda_corr"],
                        "stability_ratio": traj.get("stability_ratio", 0.0),
                        "n_control_calls": traj["n_control_calls"],
                        "elapsed_s": elapsed,
                    }
                    all_results.append(result)
                    print(f"PSNR={metrics['PSNR']:.1f} "
                          f"drift={traj['max_content_drift']:.4f} "
                          f"λ_final={traj['final_lambda_corr']:.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "robustness_results.json", "w") as f:
        clean = [{k: (float(v) if isinstance(v, (torch.Tensor, np.floating)) else v)
                  for k, v in r.items()} for r in all_results]
        json.dump(clean, f, indent=2)
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_drift_vs_editing(results: List[Dict], output_path: str):
    """Core figure: max_content_drift vs editing freedom."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["editing_strength"]][r["control_mode"]+"_drift"].append(r["max_content_drift"])

    for mode in CONTROL_MODES:
        sigmas = sorted(set(k for k in agg if agg[k].get(mode+"_drift")))
        if not sigmas:
            continue
        means = [np.mean(agg[s][mode+"_drift"]) for s in sigmas]
        stds = [np.std(agg[s][mode+"_drift"]) for s in sigmas]
        ax.errorbar(sigmas, means, yerr=stds,
                    color=COLORS[mode], marker=MARKERS[mode],
                    linewidth=2, markersize=8, capsize=4, label=mode)

    ax.set_xlabel("Editing Freedom σ (0=preserve, 1=full editing)")
    ax.set_ylabel("Max Content Drift d_content")
    ax.set_title("DCSC: Drift-Bounded Editing")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.annotate("Open-loop: drift ∝ σ", xy=(0.7, 0.08), fontsize=9, color=COLORS["open_loop"])
    ax.annotate("DCSC: drift bounded\n(P-control boosts correction)", xy=(0.4, 0.02),
                fontsize=9, color=COLORS["dcsc"], fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_psnr_vs_editing(results: List[Dict], output_path: str):
    """PSNR vs editing freedom."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["editing_strength"]][r["control_mode"]+"_psnr"].append(r["PSNR"])
    for mode in CONTROL_MODES:
        sigmas = sorted(set(k for k in agg if agg[k].get(mode+"_psnr")))
        if not sigmas:
            continue
        means = [np.mean(agg[s][mode+"_psnr"]) for s in sigmas]
        ax.plot(sigmas, means, color=COLORS[mode], marker=MARKERS[mode],
                linewidth=2, markersize=8, label=mode)
    ax.set_xlabel("Editing Freedom σ")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Image Quality Under Editing")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_correction_vs_editing(results: List[Dict], output_path: str):
    """Effective correction strength vs editing freedom."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["editing_strength"]][r["control_mode"]+"_lam"].append(r["final_lambda_corr"])
    for mode in CONTROL_MODES:
        sigmas = sorted(set(k for k in agg if agg[k].get(mode+"_lam")))
        if not sigmas:
            continue
        means = [np.mean(agg[s][mode+"_lam"]) for s in sigmas]
        ax.plot(sigmas, means, color=COLORS[mode], marker=MARKERS[mode],
                linewidth=2, markersize=8, label=mode)
    # Reference: λ_max*(1-σ)
    ref_x = np.linspace(0, 1, 50)
    ax.plot(ref_x, 0.7*(1-ref_x), "k:", alpha=0.3, label="λ=0.7·(1-σ) (open-loop)")
    ax.set_xlabel("Editing Freedom σ")
    ax.set_ylabel("Effective Correction λ_eff")
    ax.set_title("Adaptive Correction: DCSC Boosts λ When Drift Detected")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def generate_summary_table(results: List[Dict], output_path: str):
    """LaTeX table."""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["editing_strength"]][r["control_mode"]+"_d"].append(r["max_content_drift"])
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Content Drift Under Editing: DCSC vs Baselines}" + "\n")
        f.write(r"\label{tab:drift_boundedness}" + "\n")
        f.write(r"\begin{tabular}{l" + "c"*len(EDITING_VALUES) + "}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & " + " & ".join(f"σ={s:.1f}" for s in sorted(agg.keys())) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for mode in CONTROL_MODES:
            vals = []
            for s in sorted(agg.keys()):
                v = agg[s].get(mode+"_d", [0])
                vals.append(f"{np.mean(v):.4f}" if v else "—")
            f.write(f"  {mode} & " + " & ".join(vals) + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[Table] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DCSC Drift-Bounded Editing Evaluation")
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.7)
    parser.add_argument("--Kp", type=float, default=1.0)
    parser.add_argument("--editing-values", type=float, nargs="+", default=EDITING_VALUES)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[DCSC Robustness] {len(images)} images  editing={args.editing_values}  Kp={args.Kp}")
    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)
    v_content = extractor.encode_text("a photo")

    results = run_robustness_experiment(
        pipe, images, extractor, v_content,
        num_steps=args.steps, corr_lam=args.corr_lam,
        editing_values=args.editing_values, Kp=args.Kp, lpips_fn=lpips_fn,
    )

    plot_drift_vs_editing(results, str(OUT_DIR / "drift_vs_editing.png"))
    plot_psnr_vs_editing(results, str(OUT_DIR / "psnr_vs_editing.png"))
    plot_correction_vs_editing(results, str(OUT_DIR / "correction_vs_editing.png"))
    generate_summary_table(results, str(OUT_DIR / "drift_boundedness.tex"))

    # Summary
    print(f"\n{'='*60}")
    print("Drift Boundedness Summary")
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["editing_strength"]][r["control_mode"]+"_d"].append(r["max_content_drift"])
    for s in sorted(agg.keys()):
        print(f"  σ={s:.1f}:")
        for mode in CONTROL_MODES:
            vals = agg[s].get(mode+"_d", [])
            if vals:
                print(f"    {mode:15s}: drift={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
