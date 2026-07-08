#!/usr/bin/env python
"""
Phase 6c: FLUX 19-image full validation, text token drift analysis,
text injection ablation, and Euler inversion limitation analysis.

Usage:
    # Full pipeline (all steps)
    python scripts/flux_phase6c_analysis.py --mode all

    # Individual steps
    python scripts/flux_phase6c_analysis.py --mode validate    # 19-image latent corr
    python scripts/flux_phase6c_analysis.py --mode text_drift  # text token analysis
    python scripts/flux_phase6c_analysis.py --mode text_inject # encoder injection
    python scripts/flux_phase6c_analysis.py --mode euler       # Euler vs DDIM
    python scripts/flux_phase6c_analysis.py --mode analyze     # CPU-only: figs+tables
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_common import (
    load_flux_pipeline,
    flux_invert,
    FluxFeatureExtractor,
    compute_block_drift,
    compute_per_token_drift,
    compute_text_image_drift_correlation,
    compute_metrics,
    apply_correction_latent,
    FluxFeatureCorrector,
    run_correction_feature,
    OUTPUT_DIR,
    DATA_DIR,
    N_JOINT_BLOCKS,
    N_SINGLE_BLOCKS,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 19-image coco_val set (same as Phase 5)
COCO_19 = [
    "coco_000000000139", "coco_000000000285", "coco_000000000632",
    "coco_000000000724", "coco_000000000776", "coco_000000000785",
    "coco_000000000802", "coco_000000000872", "coco_000000000885",
    "coco_000000001000", "coco_000000001353", "coco_000000001490",
    "coco_000000001532", "coco_000000001584", "coco_000000001675",
    "coco_000000001818", "coco_000000002153", "coco_000000002261",
    "coco_000000002532",
]

STYLES = {
    "sd15": {"color": "#2196F3", "marker": "o", "label": "SD 1.5 (DDIM)"},
    "sdxl": {"color": "#FF9800", "marker": "s", "label": "SDXL (DDIM)"},
    "dit": {"color": "#4CAF50", "marker": "^", "label": "DiT (v-pred)"},
    "flux": {"color": "#E91E63", "marker": "D", "label": "FLUX (Flow Match)"},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def paired_ttest(vals_a, vals_b):
    """Paired t-test returning t-stat, p-value, Cohen's d."""
    from scipy import stats
    diffs = np.array(vals_a) - np.array(vals_b)
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    cohens_d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-8))
    return {"t_stat": float(t_stat), "p_value": float(p_val), "cohens_d": abs(cohens_d)}


def load_or_run_validation(pipe, image_paths, num_steps=50, lam=0.7, force=False):
    """Run 19-image validation or load cached results."""
    cache_path = OUTPUT_DIR / "full_validation_19.json"

    if cache_path.exists() and not force:
        print(f"Loading cached results from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"Running FLUX latent correction on {len(image_paths)} images...")
    results = {"args": {"num_steps": num_steps, "lam": lam},
               "per_image": [], "summary": {}}

    no_corr_psnr, no_corr_ssim, no_corr_lpips = [], [], []
    corr_psnr, corr_ssim, corr_lpips = [], [], []

    for img_path in tqdm(image_paths, desc="Evaluating"):
        img = Image.open(img_path).convert("RGB")

        out = flux_invert(pipe, img, num_steps=num_steps)
        m_base = compute_metrics(img, out["image_recon"])

        z_corrected = apply_correction_latent(
            out["z_0_recon_raw"], out["z_0"], lam=lam
        )
        z_corrected = z_corrected.to(pipe.device, dtype=pipe.vae.dtype)
        z_corrected = (
            z_corrected / pipe.vae.config.scaling_factor
            + pipe.vae.config.shift_factor
        )
        with torch.no_grad():
            img_corr = pipe.vae.decode(z_corrected, return_dict=False)[0]
            img_corr = pipe.image_processor.postprocess(img_corr, output_type="pil")[0]
        m_corr = compute_metrics(img, img_corr)

        entry = {
            "image": img_path.stem,
            "baseline_PSNR": m_base["PSNR"],
            "baseline_SSIM": m_base["SSIM"],
            "baseline_LPIPS": m_base["LPIPS"],
            "corr_PSNR": m_corr["PSNR"],
            "corr_SSIM": m_corr["SSIM"],
            "corr_LPIPS": m_corr["LPIPS"],
            "delta_PSNR": round(m_corr["PSNR"] - m_base["PSNR"], 3),
        }
        results["per_image"].append(entry)

        no_corr_psnr.append(m_base["PSNR"])
        no_corr_ssim.append(m_base["SSIM"])
        no_corr_lpips.append(m_base["LPIPS"])
        corr_psnr.append(m_corr["PSNR"])
        corr_ssim.append(m_corr["SSIM"])
        corr_lpips.append(m_corr["LPIPS"])

        # Save first 5 reconstructions
        idx = image_paths.index(img_path)
        if idx < 5:
            img.save(OUTPUT_DIR / f"{img_path.stem}_original.png")
            out["image_recon"].save(OUTPUT_DIR / f"{img_path.stem}_recon_nocorr.png")
            img_corr.save(OUTPUT_DIR / f"{img_path.stem}_recon_corr.png")

    # Summary stats
    results["summary"] = {
        "PSNR_nocorr_mean": round(np.mean(no_corr_psnr), 3),
        "PSNR_nocorr_std": round(np.std(no_corr_psnr), 3),
        "PSNR_corr_mean": round(np.mean(corr_psnr), 3),
        "PSNR_corr_std": round(np.std(corr_psnr), 3),
        "SSIM_nocorr_mean": round(np.mean(no_corr_ssim), 4),
        "SSIM_corr_mean": round(np.mean(corr_ssim), 4),
        "LPIPS_nocorr_mean": round(np.mean(no_corr_lpips), 4),
        "LPIPS_corr_mean": round(np.mean(corr_lpips), 4),
        "delta_PSNR": round(np.mean(corr_psnr) - np.mean(no_corr_psnr), 3),
    }

    # Statistical tests
    stats = paired_ttest(corr_psnr, no_corr_psnr)
    results["statistics"] = {
        "PSNR_paired_ttest": stats,
        "description": "One-sided paired t-test: H0 = correction provides no improvement",
    }

    print(f"\n{'='*60}")
    print(f"19-IMAGE FLUX VALIDATION (steps={num_steps}, lambda={lam})")
    print(f"{'='*60}")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")
    print(f"\n  Paired t-test: t={stats['t_stat']:.3f}, p={stats['p_value']:.2e}, "
          f"Cohen's d={stats['cohens_d']:.3f}")

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved: {cache_path}")

    return results


# ---------------------------------------------------------------------------
# Step 3: Text token drift analysis
# ---------------------------------------------------------------------------

def run_text_token_analysis(pipe, image_paths):
    """Per-position text token drift across joint blocks."""
    print(f"\n{'='*60}")
    print("TEXT TOKEN DRIFT ANALYSIS")
    print(f"{'='*60}")

    # Run feature extraction on up to 3 images
    n_analyze = min(3, len(image_paths))
    all_per_block = []
    all_position_means = []

    for img_path in tqdm(image_paths[:n_analyze], desc="Text drift"):
        img = Image.open(img_path).convert("RGB")
        out = flux_invert(pipe, img, num_steps=50, extract_features=True)
        drift = compute_block_drift(out["features_inv"], out["features_recon"])
        token_drift = compute_per_token_drift(out["features_inv"], out["features_recon"])
        all_per_block.append(token_drift["block_means"])
        all_position_means.append(token_drift["position_means"])

        text_img_corr = compute_text_image_drift_correlation(drift)
        token_drift["text_image_correlation"] = text_img_corr

    # Aggregate across images
    joint_names = [f"joint_{i}" for i in range(N_JOINT_BLOCKS)]
    mean_per_block = {}
    for name in joint_names:
        vals = [d.get(name, 0) for d in all_per_block]
        mean_per_block[name] = float(np.mean(vals))

    # Mean per position
    all_positions = sorted(set().union(*[d.keys() for d in all_position_means]),
                           key=lambda x: int(x))
    mean_per_position = {}
    for pos in all_positions:
        vals = [d.get(pos, 0) for d in all_position_means]
        mean_per_position[pos] = float(np.mean(vals))

    results = {
        "per_block_mean_text_drift": mean_per_block,
        "per_position_mean_text_drift": mean_per_position,
        "n_tokens_per_image": len(all_positions),
        "n_images": n_analyze,
    }

    # Find high/low drift positions (potential BOS/Semantic/Padding signatures)
    sorted_positions = sorted(mean_per_position.items(), key=lambda x: x[1], reverse=True)
    results["top5_positions"] = sorted_positions[:5]
    results["bottom5_positions"] = sorted_positions[-5:]

    print(f"\nText token drift statistics ({n_analyze} images):")
    print(f"  Tokens per prompt: {len(all_positions)}")
    print(f"  Mean per-token drift (across blocks): {np.mean(list(mean_per_position.values())):.6f}")
    print(f"  Top positions: {sorted_positions[:5]}")
    print(f"  Bottom positions: {sorted_positions[-5:]}")

    # Text vs image drift correlation per block
    correlations = []
    for d in all_per_block:
        # Compute correlation between text_drift and image_drift across joint blocks
        # We need image drift too - compute from existing data or re-extract
        pass

    # Heatmap: per-block text drift
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: text drift per joint block
    ax = axes[0]
    blocks = list(mean_per_block.keys())
    values = [mean_per_block[b] for b in blocks]
    colors = ["#2196F3" if i < 18 else "#E91E63" for i in range(N_JOINT_BLOCKS)]
    ax.barh(range(len(blocks)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(blocks, fontsize=7)
    ax.set_xlabel("Mean text token drift")
    ax.set_title("Per-block text token drift (joint blocks)")
    ax.invert_yaxis()

    # Right: text drift per token position
    ax = axes[1]
    positions = list(mean_per_position.keys())
    pos_values = [mean_per_position[p] for p in positions]
    pos_ints = [int(p) for p in positions]
    ax.bar(pos_ints, pos_values, color="#9C27B0", width=0.8)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean drift across blocks")
    ax.set_title(f"Per-position text drift ({len(positions)} tokens)")
    ax.axhline(y=np.mean(pos_values), color="red", linestyle="--", alpha=0.5,
               label=f"Mean: {np.mean(pos_values):.4f}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "text_token_drift.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'text_token_drift.png'}")

    out_path = OUTPUT_DIR / "text_token_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Step 4: Text token injection ablation
# ---------------------------------------------------------------------------

def run_text_injection_ablation(pipe, image_paths):
    """3-condition comparison: latent, latent+encoder, encoder-only."""
    print(f"\n{'='*60}")
    print("TEXT INJECTION ABLATION")
    print(f"{'='*60}")

    n_images = min(5, len(image_paths))
    conditions = [
        {"name": "latent_only", "lam_hidden": 0.7, "lam_encoder": 0.0,
         "use_feature": False},
        {"name": "feature_hidden_only", "lam_hidden": 0.7, "lam_encoder": 0.0,
         "use_feature": True},
        {"name": "feature_hidden_plus_text", "lam_hidden": 0.7, "lam_encoder": 0.5,
         "use_feature": True},
        {"name": "feature_text_only", "lam_hidden": 0.0, "lam_encoder": 0.5,
         "use_feature": True},
    ]

    results = {"args": {"n_images": n_images}, "per_image": [], "summary": {}}

    for img_path in tqdm(image_paths[:n_images], desc="Injection ablation"):
        img = Image.open(img_path).convert("RGB")
        entry = {"image": img_path.stem}

        for cond in conditions:
            if cond["use_feature"]:
                r = run_correction_feature(
                    pipe, img, num_steps=50,
                    lam_hidden=cond["lam_hidden"],
                    lam_encoder=cond["lam_encoder"],
                )
                entry[f"{cond['name']}_PSNR"] = r["corr_PSNR"]
                entry[f"{cond['name']}_LPIPS"] = r["corr_LPIPS"]
                entry[f"{cond['name']}_delta"] = r["delta_PSNR"]
            else:
                # Latent correction (baseline)
                out = flux_invert(pipe, img, num_steps=50)
                m_base = compute_metrics(img, out["image_recon"])
                z_corr = apply_correction_latent(
                    out["z_0_recon_raw"], out["z_0"], lam=0.7
                )
                z_corr = z_corr.to(pipe.device, dtype=pipe.vae.dtype)
                z_corr = (
                    z_corr / pipe.vae.config.scaling_factor
                    + pipe.vae.config.shift_factor
                )
                with torch.no_grad():
                    img_corr = pipe.vae.decode(z_corr, return_dict=False)[0]
                    img_corr = pipe.image_processor.postprocess(img_corr, output_type="pil")[0]
                m_corr = compute_metrics(img, img_corr)
                entry["latent_only_PSNR"] = m_corr["PSNR"]
                entry["latent_only_LPIPS"] = m_corr["LPIPS"]
                entry["latent_only_delta"] = round(m_corr["PSNR"] - m_base["PSNR"], 3)
                entry["baseline_PSNR"] = m_base["PSNR"]

        results["per_image"].append(entry)

    # Summary
    for cond in conditions:
        delta_key = f"{cond['name']}_delta"
        deltas = [e[delta_key] for e in results["per_image"]]
        results["summary"][f"{cond['name']}_mean_delta"] = round(np.mean(deltas), 3)
        results["summary"][f"{cond['name']}_std_delta"] = round(np.std(deltas), 3)

    print("\nText injection ablation summary:")
    for cond in conditions:
        name = cond["name"]
        print(f"  {name}: ΔPSNR = {results['summary'][f'{name}_mean_delta']:.3f} "
              f"± {results['summary'][f'{name}_std_delta']:.3f}")

    out_path = OUTPUT_DIR / "text_injection_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Step 6: Euler inversion limitation analysis
# ---------------------------------------------------------------------------

def load_arch_data():
    """Load (baseline_PSNR, delta_PSNR) per image for all four architectures."""
    data = {}

    # FLUX data
    flux_path = OUTPUT_DIR / "full_validation_19.json"
    if flux_path.exists():
        with open(flux_path) as f:
            flux_data = json.load(f)
        data["FLUX (Flow Match)"] = {
            "images": {},
            "meta": {"baseline_mean": flux_data["summary"]["PSNR_nocorr_mean"],
                     "delta_mean": flux_data["summary"]["delta_PSNR"]},
        }
        for e in flux_data["per_image"]:
            data["FLUX (Flow Match)"]["images"][e["image"]] = {
                "baseline": e["baseline_PSNR"], "delta": e["delta_PSNR"],
            }

    # SD 1.5 from Phase 2 (19 images, 50 steps)
    sd15_path = Path("outputs/phase2_full/metrics.json")
    if sd15_path.exists():
        with open(sd15_path) as f:
            sd15_raw = json.load(f)

        sd15_base = {}
        sd15_corr = {}
        for e in sd15_raw:
            img = e.get("image", "")
            if e.get("steps") != 50:
                continue
            lmbda = str(e.get("lambda", ""))
            if lmbda == "baseline":
                sd15_base[img] = e
            elif lmbda == "0.7" and e.get("scheduler") == "constant":
                sd15_corr[img] = e

        common = set(sd15_base) & set(sd15_corr)
        data["SD 1.5 (DDIM)"] = {"images": {}, "meta": {}}
        baselines, deltas = [], []
        for img in sorted(common):
            base = sd15_base[img]["PSNR"]
            corr = sd15_corr[img]["PSNR"]
            data["SD 1.5 (DDIM)"]["images"][img] = {"baseline": base, "delta": corr - base}
            baselines.append(base)
            deltas.append(corr - base)
        data["SD 1.5 (DDIM)"]["meta"] = {
            "baseline_mean": round(np.mean(baselines), 2),
            "delta_mean": round(np.mean(deltas), 2),
        }

    # SDXL from phase2
    sdxl_path = Path("outputs/sdxl_phase2/sdxl_phase2_results")
    if sdxl_path.exists():
        with open(sdxl_path) as f:
            sdxl_raw = json.load(f)
        # Deduplicate: pick best lambda (0.7) per image, top5 group
        sdxl_best = {}
        for e in sdxl_raw:
            if e.get("lambda") == 0.7 and e.get("group") == "top5":
                sdxl_best[e["image"]] = e
        if sdxl_best:
            baselines = [e["baseline_PSNR"] for e in sdxl_best.values()]
            deltas = [e["Δ_PSNR"] for e in sdxl_best.values()]
            data["SDXL (DDIM)"] = {"images": {}, "meta": {
                "baseline_mean": round(np.mean(baselines), 2),
                "delta_mean": round(np.mean(deltas), 2),
            }}
            for img, e in sdxl_best.items():
                data["SDXL (DDIM)"]["images"][img] = {
                    "baseline": e["baseline_PSNR"], "delta": e["Δ_PSNR"],
                }

    # DiT from phase2
    dit_path = Path("outputs/dit_phase2/ablation.json")
    if dit_path.exists():
        with open(dit_path) as f:
            dit_raw = json.load(f)
        dit_best = {}
        for e in dit_raw:
            if e.get("group") != "top5":
                continue
            img = e.get("image", "")
            if img == "average":
                continue
            if img not in dit_best or e["PSNR"] > dit_best[img]["PSNR"]:
                dit_best[img] = e
        if dit_best:
            baselines = [e["PSNR"] - e["delta_psnr"] for e in dit_best.values()]
            deltas = [e["delta_psnr"] for e in dit_best.values()]
            data["DiT (v-pred)"] = {"images": {}, "meta": {
                "baseline_mean": round(np.mean(baselines), 2),
                "delta_mean": round(np.mean(deltas), 2),
            }}
            for img, e in dit_best.items():
                data["DiT (v-pred)"]["images"][img] = {
                    "baseline": e["PSNR"] - e["delta_psnr"],
                    "delta": e["delta_psnr"],
                }

    return data


def run_euler_analysis():
    """Cross-architecture baseline vs delta scatter plot."""
    print(f"\n{'='*60}")
    print("EULER INVERSION LIMITATION ANALYSIS")
    print(f"{'='*60}")

    data = load_arch_data()
    if len(data) < 2:
        print("Not enough architecture data. Run validations first.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for arch_name, arch_data in data.items():
        style_key = None
        for k in STYLES:
            if k in arch_name.lower():
                style_key = k
                break
        if style_key is None:
            style_key = "flux"

        s = STYLES[style_key]
        xs = [v["baseline"] for v in arch_data["images"].values()]
        ys = [v["delta"] for v in arch_data["images"].values()]

        if xs:
            ax.scatter(xs, ys, c=s["color"], marker=s["marker"], s=80,
                       label=s["label"], edgecolors="white", linewidth=0.5, zorder=3)
            # Mark mean
            ax.scatter([arch_data["meta"]["baseline_mean"]],
                       [arch_data["meta"]["delta_mean"]],
                       c=s["color"], marker=s["marker"], s=200,
                       edgecolors="black", linewidth=1.5, zorder=4)

    ax.set_xlabel("Baseline PSNR (no correction) [dB]", fontsize=12)
    ax.set_ylabel("ΔPSNR (correction gain) [dB]", fontsize=12)
    ax.set_title("Correction gain vs. baseline fidelity\n"
                 "Four architectures across two paradigms", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Annotate Euler cost
    if "FLUX (Flow Match)" in data and "SD 1.5 (DDIM)" in data:
        flux_base = data["FLUX (Flow Match)"]["meta"]["baseline_mean"]
        sd15_base = data["SD 1.5 (DDIM)"]["meta"]["baseline_mean"]
        euler_gap = sd15_base - flux_base
        flux_delta = data["FLUX (Flow Match)"]["meta"]["delta_mean"]
        recovery_pct = 100 * flux_delta / euler_gap if euler_gap > 0 else 0

        ax.annotate(
            f"Euler cost: ~{euler_gap:.1f} dB\n"
            f"FLUX correction recovers ~{recovery_pct:.0f}%\n"
            f"of the Euler inversion gap",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "euler_vs_ddim_baseline.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'euler_vs_ddim_baseline.png'}")

    # Print summary
    print("\nCross-architecture summary:")
    print(f"{'Architecture':<25s} {'Baseline':>10s} {'ΔPSNR':>10s} {'N':>6s}")
    print("-" * 55)
    for name, d in data.items():
        n = len(d["images"])
        print(f"{name:<25s} {d['meta']['baseline_mean']:>10.2f} "
              f"{d['meta']['delta_mean']:>10.2f} {n:>6d}")

    out_path = OUTPUT_DIR / "euler_analysis.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure generation (CPU-only)
# ---------------------------------------------------------------------------

def generate_figures(validation_results):
    """Generate per-image bar chart and LaTeX table from 19-image results."""
    per_image = validation_results.get("per_image", [])
    if not per_image:
        print("No per-image data available")
        return

    # Per-image delta PSNR bar chart
    images = [e["image"].replace("coco_", "") for e in per_image]
    deltas = [e["delta_PSNR"] for e in per_image]
    baselines = [e["baseline_PSNR"] for e in per_image]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(images))
    width = 0.35

    bars1 = ax.bar(x - width/2, baselines, width, label="Baseline (no corr)",
                   color="#607D8B", alpha=0.8)
    bars2 = ax.bar(x + width/2, [b + d for b, d in zip(baselines, deltas)],
                   width, label="With correction", color="#E91E63", alpha=0.8)

    ax.set_ylabel("PSNR [dB]")
    ax.set_title(f"FLUX.1-dev Reconstruction Quality (n={len(images)}, 50 steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.axhline(y=np.mean(baselines), color="#607D8B", linestyle="--", alpha=0.3)
    ax.axhline(y=np.mean([b+d for b,d in zip(baselines, deltas)]),
               color="#E91E63", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_image_psnr.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'per_image_psnr.png'}")

    # LaTeX table
    lines = [r"\begin{tabular}{lcccc}",
             r"\toprule",
             r"Image & PSNR$_{\text{nocorr}}$ & PSNR$_{\text{corr}}$ & "
             r"$\Delta$PSNR & LPIPS$_{\text{nocorr}}$ \\",
             r"\midrule"]
    for e in per_image:
        img_short = e["image"].replace("coco_0000000", "").replace("coco_00000000", "")
        lines.append(
            f"{img_short} & {e['baseline_PSNR']:.2f} & {e['corr_PSNR']:.2f} & "
            f"{e['delta_PSNR']:+.2f} & {e['baseline_LPIPS']:.3f} \\\\"
        )
    # Mean row
    lines.append(r"\midrule")
    lines.append(
        f"Mean & {validation_results['summary']['PSNR_nocorr_mean']:.2f} & "
        f"{validation_results['summary']['PSNR_corr_mean']:.2f} & "
        f"{validation_results['summary']['delta_PSNR']:+.2f} & "
        f"{validation_results['summary']['LPIPS_nocorr_mean']:.3f} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = OUTPUT_DIR / "per_image_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 6c: Full Analysis")
    parser.add_argument("--mode", default="all",
                        choices=["all", "validate", "text_drift", "text_inject",
                                 "euler", "analyze", "figures"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--offload-t5", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if cached results exist")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Determine which steps to run
    run_validate = args.mode in ("all", "validate")
    run_text_drift = args.mode in ("all", "text_drift")
    run_text_inject = args.mode in ("all", "text_inject")
    run_euler = args.mode in ("all", "euler")
    run_figures = args.mode in ("all", "analyze", "figures")

    # Load image paths
    image_paths = []
    for name in COCO_19:
        for ext in (".jpg", ".png", ".jpeg"):
            p = DATA_DIR / f"{name}{ext}"
            if p.exists():
                image_paths.append(p)
                break

    if not image_paths:
        print(f"No coco_val images found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images (expected 19)")

    # Load pipeline if GPU steps needed
    pipe = None
    if run_validate or run_text_drift or run_text_inject:
        pipe = load_flux_pipeline(device=args.device, offload_t5=args.offload_t5)

    # Step 2: 19-image validation
    validation_results = None
    if run_validate:
        validation_results = load_or_run_validation(
            pipe, image_paths, num_steps=args.num_steps, lam=args.lam,
            force=args.force,
        )

    # Step 3: Text token drift analysis
    if run_text_drift:
        run_text_token_analysis(pipe, image_paths)

    # Step 4: Text injection ablation
    if run_text_inject:
        run_text_injection_ablation(pipe, image_paths)

    # Step 6: Euler vs DDIM analysis
    if run_euler:
        run_euler_analysis()

    # Generate figures and tables from saved data
    if run_figures:
        cache_path = OUTPUT_DIR / "full_validation_19.json"
        if cache_path.exists():
            with open(cache_path) as f:
                validation_results = json.load(f)
        if validation_results:
            generate_figures(validation_results)
        else:
            print("No validation results found. Run --mode validate first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
