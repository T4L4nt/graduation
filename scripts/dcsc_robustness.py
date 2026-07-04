"""
DCSC Robustness: Drift-Bounded Editing — Normal + Adversarial Stress Test.

Two scenarios:
1. Normal editing (adversarial_noise=0): correction mechanism is inherently
   robust. DCSC correctly stays idle — no false-trigger intervention.
2. Stress test (adversarial_noise > 0): random noise injected into correction
   residual. DCSC detects PSNR loss and boosts correction to protect content.

Key metric: final PSNR vs adversarial_noise. Prediction:
  - Normal: all modes identical (DCSC correctly idle)
  - Stress: DCSC > Phase3 > open_loop at high adversarial_noise

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

from phase2_common import DEVICE, DTYPE, load_pipeline, load_image, get_top_drift_layers
from phase3_prep import CLIPFeatureExtractor, run_baseline, run_correction_only
from dcsc_core import drift_bounded_generation

OUT_DIR = Path("outputs/dcsc")

ADVERSARIAL_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]
CONTROL_MODES = ["open_loop", "phase3_pin", "dcsc"]
COLORS = {"open_loop": "#e74c3c", "phase3_pin": "#3498db", "dcsc": "#2ecc71"}


def run_experiment(pipe, images, extractor, v_content,
                   num_steps=50, corr_lam=0.7, Kp=5.0, lpips_fn=None) -> List[Dict]:
    corr_layers = get_top_drift_layers(5)
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    all_results = []

    for img_path in images:
        if not os.path.exists(img_path): continue
        img_name = Path(img_path).stem
        original_latent, original_tensor = load_image(pipe, img_path)

        m_base, _, _ = run_baseline(pipe, original_latent, original_tensor,
                                     prompt_embeds, num_steps, lpips_fn)
        m_corr, _, _ = run_correction_only(pipe, original_latent, original_tensor,
                                            prompt_embeds, num_steps, corr_lam,
                                            corr_layers, lpips_fn)
        print(f"\n[DCSC] {img_name}  DDIM={m_base['PSNR']:.1f}  Corr={m_corr['PSNR']:.1f}")

        for adv in ADVERSARIAL_VALUES:
            for mode in CONTROL_MODES:
                tag = "normal" if adv == 0 else f"adv={adv:.1f}"
                print(f"  {tag} mode={mode}...", end=" ", flush=True)
                try:
                    metrics, recon, elapsed, traj = drift_bounded_generation(
                        pipe, original_latent, original_tensor, prompt_embeds,
                        num_steps=num_steps, corr_lam=corr_lam,
                        corr_layers=corr_layers,
                        extractor=extractor, v_content=v_content,
                        editing_strength=0.5,  # weak correction → controller has room to boost
                        adversarial_noise=adv,
                        control_mode=mode, Kp=Kp, control_freq=5,
                        lpips_fn=lpips_fn,
                    )
                    result = {
                        "image": img_name, "adversarial_noise": adv,
                        "control_mode": mode, "Kp": Kp,
                        "PSNR": metrics["PSNR"],
                        "LPIPS": metrics["LPIPS"],
                        "SSIM": metrics["SSIM"],
                        "max_psnr_degradation": traj["max_psnr_degradation"],
                        "final_lambda_corr": traj["final_lambda_corr"],
                        "n_control_calls": traj["n_control_calls"],
                        "ref_psnr": traj["ref_psnr"],
                        "elapsed_s": elapsed,
                    }
                    all_results.append(result)
                    print(f"PSNR={metrics['PSNR']:.1f} d={traj['max_psnr_degradation']:.1f} "
                          f"λ={traj['final_lambda_corr']:.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "robustness_results.json", "w") as f:
        clean = [{k: (float(v) if isinstance(v, (torch.Tensor, np.floating)) else v)
                  for k, v in r.items()} for r in all_results]
        json.dump(clean, f, indent=2)
    return all_results


def plot_psnr_vs_adversarial(results, output_path):
    """PSNR vs adversarial_noise — primary figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["adversarial_noise"]][r["control_mode"]+"_p"].append(r["PSNR"])
    for mode in CONTROL_MODES:
        advs = sorted(set(k for k in agg if agg[k].get(mode+"_p")))
        if not advs: continue
        means = [np.mean(agg[a][mode+"_p"]) for a in advs]
        ax.plot(advs, means, color=COLORS[mode], marker="o" if mode=="dcsc" else "s",
                linewidth=2, markersize=8, label=mode)
    ax.set_xlabel("Adversarial Noise (σ_adv)")
    ax.set_ylabel("Final PSNR (dB)")
    ax.set_title("DCSC: Adversarial Robustness — PSNR Under Perturbation")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.annotate("DCSC protects content\n(P-control detects drift, boosts correction)",
                xy=(3.0, 15), fontsize=10, color=COLORS["dcsc"], fontweight="bold")
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close()
    print(f"[Figure] {output_path}")


def plot_lambda_vs_adversarial(results, output_path):
    """Final correction lambda vs adversarial_noise."""
    fig, ax = plt.subplots(figsize=(10, 6))
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["adversarial_noise"]][r["control_mode"]+"_l"].append(r["final_lambda_corr"])
    for mode in CONTROL_MODES:
        advs = sorted(set(k for k in agg if agg[k].get(mode+"_l")))
        if not advs: continue
        means = [np.mean(agg[a][mode+"_l"]) for a in advs]
        ax.plot(advs, means, color=COLORS[mode], marker="o" if mode=="dcsc" else "s",
                linewidth=2, markersize=8, label=mode)
    ax.set_xlabel("Adversarial Noise (σ_adv)")
    ax.set_ylabel("Final Correction λ")
    ax.set_title("Adaptive Correction: λ Response to Perturbation")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(output_path, dpi=150); plt.close()
    print(f"[Figure] {output_path}")


def generate_table(results, output_path):
    """LaTeX table."""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["adversarial_noise"]][r["control_mode"]+"_p"].append(r["PSNR"])
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{PSNR Under Adversarial Perturbation}" + "\n")
        f.write(r"\label{tab:dcsc_adversarial}" + "\n")
        f.write(r"\begin{tabular}{l" + "c"*len(ADVERSARIAL_VALUES) + "}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & " + " & ".join(f"σ={a:.1f}" for a in ADVERSARIAL_VALUES) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for mode in CONTROL_MODES:
            vals = [f"{np.mean(agg[a].get(mode+'_p',[0])):.1f}" if agg[a].get(mode+"_p") else "—"
                    for a in ADVERSARIAL_VALUES]
            f.write(f"  {mode} & " + " & ".join(vals) + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"[Table] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DCSC Adversarial Robustness")
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--corr-lam", type=float, default=0.7)
    parser.add_argument("--Kp", type=float, default=5.0)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[DCSC] {len(images)} images  adversarial={ADVERSARIAL_VALUES}  Kp={args.Kp}")
    print("[0] Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()
    import lpips
    lpips_fn = None if args.skip_lpips else lpips.LPIPS(net="alex").to(DEVICE)
    v_content = extractor.encode_text("a photo")

    results = run_experiment(pipe, images, extractor, v_content,
                             num_steps=args.steps, corr_lam=args.corr_lam,
                             Kp=args.Kp, lpips_fn=lpips_fn)

    plot_psnr_vs_adversarial(results, str(OUT_DIR / "psnr_vs_adversarial.png"))
    plot_lambda_vs_adversarial(results, str(OUT_DIR / "lambda_vs_adversarial.png"))
    generate_table(results, str(OUT_DIR / "adversarial_table.tex"))

    # Summary
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["adversarial_noise"]][r["control_mode"]+"_p"].append(r["PSNR"])
    print(f"\n{'='*60}")
    print(f"Adversarial Robustness Summary")
    for a in sorted(agg.keys()):
        print(f"  σ_adv={a:.1f}:")
        for mode in CONTROL_MODES:
            vals = agg[a].get(mode+"_p", [])
            if vals:
                print(f"    {mode:15s}: PSNR={np.mean(vals):.1f} ± {np.std(vals):.1f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
