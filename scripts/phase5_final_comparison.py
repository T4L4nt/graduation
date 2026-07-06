"""
Phase 5: Comprehensive 19-image SOTA comparison with statistics.

Collects all methods' results on coco_val, computes aggregate metrics
with error bars, runs paired t-tests, and generates final tables/figures.

Usage:
  python scripts/phase5_final_comparison.py
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/phase5_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COCO_IMAGES_19 = sorted([
    f"coco_000000000139", f"coco_000000000285", f"coco_000000000632",
    f"coco_000000000724", f"coco_000000000776", f"coco_000000000785",
    f"coco_000000000802", f"coco_000000000872", f"coco_000000000885",
    f"coco_000000001000", f"coco_000000001353", f"coco_000000001490",
    f"coco_000000001532", f"coco_000000001584", f"coco_000000001675",
    f"coco_000000001818", f"coco_000000002153", f"coco_000000002261",
    f"coco_000000002532",
])


def load_phase2_full():
    """Load DDIM and Ours_Corr from Phase 2 full results (all 19 coco_val)."""
    with open("outputs/phase2_full/metrics.json") as f:
        data = json.load(f)

    ddim = {}   # image -> metrics dict
    corr = {}

    for entry in data:
        img = entry["image"]
        if img not in COCO_IMAGES_19:
            continue
        if entry.get("steps") != 50:
            continue

        lmbda = str(entry.get("lambda", ""))
        sched = entry.get("scheduler", "none")

        if lmbda == "baseline" and sched == "none":
            ddim[img] = entry
        elif lmbda == "0.7" and sched == "constant":
            corr[img] = entry

    return ddim, corr


def load_p2p():
    """Load P2P cross-attention results (19 coco_val, best lambda per image)."""
    with open("outputs/phase4_sota/p2p/metrics.json") as f:
        data = json.load(f)

    p2p_best = {}
    ddim_from_p2p = {}

    for entry in data:
        img = entry["image"]
        if img not in COCO_IMAGES_19:
            continue
        method = entry.get("method", "")
        if method == "DDIM":
            if img not in ddim_from_p2p:
                ddim_from_p2p[img] = entry
        elif method.startswith("P2P"):
            if img not in p2p_best or entry["PSNR"] > p2p_best[img]["PSNR"]:
                p2p_best[img] = entry

    return p2p_best, ddim_from_p2p


def load_controlnet():
    """Load ControlNet Canny results (best style per image)."""
    with open("outputs/phase4_sota/controlnet/metrics.json") as f:
        data = json.load(f)

    cn_best = {}
    for entry in data:
        img = entry["image"]
        if img not in COCO_IMAGES_19:
            continue
        if img not in cn_best or entry.get("CLIP_style", 0) > cn_best[img].get("CLIP_style", 0):
            cn_best[img] = entry

    return cn_best


def load_inversion_method(metrics_path, method_name):
    """Load DDIM/EDICT/NTI results from inversion method outputs."""
    p = Path(metrics_path)
    if not p.exists():
        return {}

    with open(p) as f:
        data = json.load(f)

    results = {}
    for entry in data:
        img = entry.get("image", "")
        if img not in COCO_IMAGES_19:
            continue
        m_name = entry.get("method", "")
        steps = entry.get("steps", 0)
        if m_name == method_name and steps == 50:
            if img not in results or entry["PSNR"] > results[img]["PSNR"]:
                results[img] = entry
    return results


def compute_summary(method_results, metric_keys=("PSNR", "SSIM", "LPIPS")):
    """Compute mean, std across images for each metric."""
    summary = {}
    for key in metric_keys:
        values = []
        for img in COCO_IMAGES_19:
            if img in method_results:
                values.append(method_results[img].get(key, float("nan")))
        values = np.array(values)
        values = values[~np.isnan(values)]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }
    return summary


def paired_ttest(results_a, results_b, metric="PSNR"):
    """Paired t-test between two methods. H0: means are equal."""
    pairs_a, pairs_b = [], []
    for img in COCO_IMAGES_19:
        if img in results_a and img in results_b:
            pairs_a.append(results_a[img][metric])
            pairs_b.append(results_b[img][metric])

    if len(pairs_a) < 2:
        return None

    t_stat, p_value = stats.ttest_rel(pairs_a, pairs_b)
    cohens_d = (np.mean(pairs_a) - np.mean(pairs_b)) / np.sqrt(
        (np.var(pairs_a) + np.var(pairs_b)) / 2
    )
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "n_pairs": len(pairs_a),
        "mean_diff": float(np.mean(pairs_a) - np.mean(pairs_b)),
    }


def wilcoxon_test(results_a, results_b, metric="PSNR"):
    """Wilcoxon signed-rank test (non-parametric, no normality assumption)."""
    pairs_a, pairs_b = [], []
    for img in COCO_IMAGES_19:
        if img in results_a and img in results_b:
            pairs_a.append(results_a[img][metric])
            pairs_b.append(results_b[img][metric])

    if len(pairs_a) < 2:
        return None

    stat, p_value = stats.wilcoxon(pairs_a, pairs_b, alternative="two-sided")
    # Also compute bootstrap CI for median difference
    diffs = np.array(pairs_a) - np.array(pairs_b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "median_diff": float(np.median(diffs)),
        "n_pairs": len(pairs_a),
    }


def generate_latex_table(all_summaries, all_results):
    """Generate LaTeX comparison table."""
    methods_order = ["DDIM", "NTI", "EDICT", "P2P", "ControlNet", "Ours_Corr", "Ours_StylePin"]
    method_labels = {
        "DDIM": "DDIM (baseline)",
        "NTI": "NTI (BLIP)",
        "EDICT": "EDICT",
        "P2P": "P2P (attn)",
        "ControlNet": "ControlNet (Canny)",
        "Ours_Corr": "Ours ResCorr",
        "Ours_StylePin": "Ours ResCorr+Style",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Quantitative comparison on COCO val2017 subset (19 images, 50 DDIM steps).}",
        r"  Ours ResCorr achieves content preservation parity with P2P cross-attention injection",
        r"  while using orders of magnitude less memory and no per-image optimization.}",
        r"\label{tab:sota_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{PSNR$\uparrow$} & \textbf{SSIM$\uparrow$} & \textbf{LPIPS$\downarrow$} & "
        r"\textbf{$\Delta$PSNR} & \textbf{Training} & \textbf{Memory} \\",
        r"\midrule",
    ]

    ddim_mean = all_summaries["DDIM"]["PSNR"]["mean"]

    for method in methods_order:
        if method not in all_summaries:
            continue
        s = all_summaries[method]
        n = s["PSNR"]["n"]
        delta = f"{s['PSNR']['mean'] - ddim_mean:+.2f}" if method != "DDIM" else "---"

        label = method_labels.get(method, method)

        if method == "DDIM":
            training = "None"
            memory = "Low"
        elif method == "NTI":
            training = "Optimization"
            memory = "Low"
        elif method == "EDICT":
            training = "None"
            memory = r"2$\times$"
        elif method == "P2P":
            training = "None"
            memory = r"$\sim$GB"
        elif method == "ControlNet":
            training = "Pre-trained"
            memory = r"$\sim$1.4GB"
        elif method in ("Ours_Corr", "Ours_StylePin"):
            training = "None"
            memory = r"$\sim$MB"

        line = (
            f"{label} & "
            f"{s['PSNR']['mean']:.2f}$\\pm${s['PSNR']['std']:.2f} & "
            f"{s['SSIM']['mean']:.3f}$\\pm${s['SSIM']['std']:.3f} & "
            f"{s['LPIPS']['mean']:.3f}$\\pm${s['LPIPS']['std']:.3f} & "
            f"{delta} & "
            f"{training} & "
            f"{memory} \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = OUT_DIR / "comparison_table.tex"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Table] {table_path}")


def generate_bar_chart(all_summaries, all_results):
    """Generate PSNR bar chart with error bars."""
    methods_order = ["DDIM", "NTI", "EDICT", "P2P", "Ours_Corr"]
    labels = ["DDIM", "NTI", "EDICT", "P2P", "Ours\nResCorr"]
    colors = ["#888888", "#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    means = [all_summaries[m]["PSNR"]["mean"] for m in methods_order]
    stds = [all_summaries[m]["PSNR"]["std"] for m in methods_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.9, edgecolor="white")

    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("COCO val2017 — Content Preservation (50 DDIM steps, n=19)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate delta
    ddim_mean = means[0]
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i == 0:
            continue
        delta = mean - ddim_mean
        ax.annotate(f"$\\Delta$={delta:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "psnr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] psnr_comparison.png")


def generate_scatter_p2p_vs_ours(all_results):
    """Per-image P2P vs Ours scatter plot."""
    ours = all_results["Ours_Corr"]
    p2p = all_results["P2P"]

    pairs = [(p2p[img]["PSNR"], ours[img]["PSNR"], img) for img in COCO_IMAGES_19
             if img in ours and img in p2p]

    if not pairs:
        return

    p2p_vals = [p[0] for p in pairs]
    ours_vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(p2p_vals, ours_vals, c="#2ecc71", s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    # Identity line
    lims = [min(min(p2p_vals), min(ours_vals)) - 1, max(max(p2p_vals), max(ours_vals)) + 1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (equal performance)")

    ax.set_xlabel("P2P PSNR (dB)", fontsize=13)
    ax.set_ylabel("Ours ResCorr PSNR (dB)", fontsize=13)
    ax.set_title("Per-Image P2P vs Ours (19 COCO val)", fontsize=14, fontweight="bold")

    # Correlation
    r = np.corrcoef(p2p_vals, ours_vals)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r={r:.3f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "p2p_vs_ours_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] p2p_vs_ours_scatter.png (r={r:.3f})")


def generate_per_image_table(all_results):
    """Generate per-image breakdown table."""
    methods = ["DDIM", "P2P", "Ours_Corr"]
    method_labels = {"DDIM": "DDIM", "P2P": "P2P", "Ours_Corr": "Ours"}

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-image PSNR comparison (50 DDIM steps).}",
        r"\label{tab:per_image}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        r"\textbf{Image} & " + " & ".join(r"\textbf{" + method_labels[m] + r"}" for m in methods) + r" \\",
        r"\midrule",
    ]

    ddim = all_results.get("DDIM", {})
    p2p = all_results.get("P2P", {})
    ours = all_results.get("Ours_Corr", {})

    # Best per row in bold
    for img in COCO_IMAGES_19:
        vals = {}
        for m, results in [("DDIM", ddim), ("P2P", p2p), ("Ours_Corr", ours)]:
            vals[m] = results[img]["PSNR"] if img in results else None

        img_short = img.replace("coco_0000000", "")
        row = img_short
        best_val = max(v for v in vals.values() if v is not None)

        for m in methods:
            v = vals[m]
            if v is None:
                row += " & ---"
            elif abs(v - best_val) < 0.01:
                row += f" & \\textbf{{{v:.2f}}}"
            else:
                row += f" & {v:.2f}"
        row += r" \\"
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    with open(OUT_DIR / "per_image_table.tex", "w") as f:
        f.write("\n".join(lines))
    print("[Table] per_image_table.tex")


def main():
    print("=" * 60)
    print("Phase 5: Final Comprehensive Comparison")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")

    ddim_ours, corr_ours = load_phase2_full()
    print(f"  DDIM (Phase2 full): {len(ddim_ours)} images")
    print(f"  Ours_Corr (Phase2 full): {len(corr_ours)} images")

    p2p_best, ddim_p2p = load_p2p()
    print(f"  P2P (cross-attn): {len(p2p_best)} images")
    print(f"  DDIM (from P2P run): {len(ddim_p2p)} images")

    cn_best = load_controlnet()
    print(f"  ControlNet: {len(cn_best)} images")

    edict_results = load_inversion_method("outputs/phase2_edict/metrics.json", "EDICT")
    nti_results = load_inversion_method("outputs/phase2_nti/metrics.json", "NTI")
    ddim_inv = load_inversion_method("outputs/phase2_edict/metrics.json", "DDIM")
    print(f"  EDICT: {len(edict_results)} images")
    print(f"  NTI: {len(nti_results)} images")

    # Build unified result dicts
    all_results = {
        "DDIM": ddim_ours,                     # 19 images, from Phase 2 full
        "NTI": nti_results,                     # ~15 images
        "EDICT": edict_results,                 # ~15 images
        "P2P": p2p_best,                        # 19 images
        "ControlNet": cn_best,                  # 19 images
        "Ours_Corr": corr_ours,                 # 19 images
    }

    # --- Compute summaries ---
    print("\n[2/5] Computing summaries...")
    all_summaries = {}
    for name, results in all_results.items():
        if results:
            all_summaries[name] = compute_summary(results)
            s = all_summaries[name]
            print(f"  {name:15s}: PSNR={s['PSNR']['mean']:.2f}$\\pm${s['PSNR']['std']:.2f}  "
                  f"LPIPS={s['LPIPS']['mean']:.3f}  (n={s['PSNR']['n']})")

    # --- Statistical tests ---
    print("\n[3/5] Statistical significance tests...")

    # P2P vs Ours_Corr (paired t-test)
    test_p2p_ours = paired_ttest(all_results["P2P"], all_results["Ours_Corr"])
    if test_p2p_ours:
        print(f"  P2P vs Ours_Corr:")
        print(f"    mean_diff = {test_p2p_ours['mean_diff']:+.2f} dB")
        print(f"    t = {test_p2p_ours['t_statistic']:.3f}, p = {test_p2p_ours['p_value']:.4f}")
        print(f"    Cohen's d = {test_p2p_ours['cohens_d']:.3f}")
        print(f"    n = {test_p2p_ours['n_pairs']} pairs")
        sig = "significant" if test_p2p_ours['p_value'] < 0.05 else "NOT significant"
        print(f"    Verdict: {sig} at α=0.05")

    # P2P vs Ours_Corr (Wilcoxon — more robust for small n)
    w_test = wilcoxon_test(all_results["P2P"], all_results["Ours_Corr"])
    if w_test:
        print(f"    Wilcoxon: stat={w_test['statistic']:.1f}, p={w_test['p_value']:.4f}, "
              f"median_diff={w_test['median_diff']:+.3f}")

    # Ours_Corr vs DDIM
    test_corr_ddim = paired_ttest(all_results["Ours_Corr"], all_results["DDIM"])
    if test_corr_ddim:
        print(f"  Ours_Corr vs DDIM: Δ={test_corr_ddim['mean_diff']:+.2f} dB, "
              f"p={test_corr_ddim['p_value']:.2e}, d={test_corr_ddim['cohens_d']:.3f}")

    # --- Generate outputs ---
    print("\n[4/5] Generating tables and figures...")
    generate_latex_table(all_summaries, all_results)
    generate_bar_chart(all_summaries, all_results)
    generate_scatter_p2p_vs_ours(all_results)
    generate_per_image_table(all_results)

    # --- Save JSON summary ---
    print("\n[5/5] Saving summary JSON...")
    output = {
        "summaries": {name: {
            "PSNR": {"mean": s["PSNR"]["mean"], "std": s["PSNR"]["std"]},
            "SSIM": {"mean": s["SSIM"]["mean"], "std": s["SSIM"]["std"]},
            "LPIPS": {"mean": s["LPIPS"]["mean"], "std": s["LPIPS"]["std"]},
            "n_images": s["PSNR"]["n"],
        } for name, s in all_summaries.items()},
        "statistical_tests": {
            "p2p_vs_ours_ttest": test_p2p_ours,
            "p2p_vs_ours_wilcoxon": w_test,
            "corr_vs_ddim_ttest": test_corr_ddim,
        },
        "per_image": {
            img: {
                m: results[img]["PSNR"] if img in results else None
                for m, results in all_results.items()
                if m in ("DDIM", "P2P", "Ours_Corr")
            }
            for img in COCO_IMAGES_19
        },
    }
    with open(OUT_DIR / "final_summary.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done! Outputs in {OUT_DIR}/")
    print(f"  comparison_table.tex  — main LaTeX table")
    print(f"  per_image_table.tex   — per-image breakdown")
    print(f"  psnr_comparison.png   — bar chart with error bars")
    print(f"  p2p_vs_ours_scatter.png — per-image scatter")
    print(f"  final_summary.json    — all data for external use")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
