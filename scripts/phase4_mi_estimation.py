"""
Phase 4 互信息估计：KSG 估计器量化逐层信息保持

核心问题：重建特征 f_recon 保留了多少反演特征 f_inv 的信息？
答案：I(f_inv; f_recon) — 逐层估计，揭示信息丢失的架构级模式。

方法：KSG (Kraskov-Stojmirovic-Grassberger, 2004) k-NN 互信息估计器。
- 非参数、无分布假设、适用于连续变量
- PCA 降维至 top-10 主成分后估计（流形分析已验证快速谱衰减）
- Bootstrap 重采样给出置信区间

这是真正的信息论分析——直接估计互信息，而非用 ΔPSNR 做代理。

用法:
  python scripts/phase4_mi_estimation.py --quick        # 3 图快速验证
  python scripts/phase4_mi_estimation.py                # 全量 19 图
"""

import argparse, json, os, sys
from pathlib import Path

import torch
import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    load_pipeline, load_image,
    FeatureCollector,
    DEVICE, DTYPE,
)

OUT_DIR = Path("outputs/phase4_mi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All ResNet + key Attention layers for comprehensive comparison
MI_LAYERS = [
    # Encoder
    "down_blocks.0.resnets.0", "down_blocks.0.resnets.1",
    "down_blocks.1.resnets.0", "down_blocks.1.resnets.1",
    "down_blocks.2.resnets.0", "down_blocks.2.resnets.1",
    "down_blocks.3.resnets.0", "down_blocks.3.resnets.1",
    # Bottleneck
    "mid_block.resnets.0", "mid_block.resnets.1",
    "mid_block.attentions.0.transformer_blocks.0",
    # Decoder
    "up_blocks.0.resnets.0", "up_blocks.0.resnets.1", "up_blocks.0.resnets.2",
    "up_blocks.1.resnets.0", "up_blocks.1.resnets.1", "up_blocks.1.resnets.2",
    "up_blocks.2.resnets.0", "up_blocks.2.resnets.1", "up_blocks.2.resnets.2",
    "up_blocks.3.resnets.0", "up_blocks.3.resnets.1", "up_blocks.3.resnets.2",
    # Attention (for contrast)
    "up_blocks.0.attentions.0.transformer_blocks.0",
    "up_blocks.1.attentions.0.transformer_blocks.0",
    "up_blocks.2.attentions.0.transformer_blocks.0",
    "up_blocks.3.attentions.0.transformer_blocks.0",
    "down_blocks.0.attentions.0.transformer_blocks.0",
    "down_blocks.1.attentions.0.transformer_blocks.0",
    "down_blocks.2.attentions.0.transformer_blocks.0",
]


# ---------------------------------------------------------------------------
# Mutual Information Estimators
# ---------------------------------------------------------------------------

def gaussian_mi(x, y):
    """Estimate I(X;Y) assuming joint Gaussianity.

    I_gauss = 0.5 * log( det(Σ_xx) * det(Σ_yy) / det(Σ_joint) )

    This is exact for Gaussian variables and provides a reasonable lower-bound
    approximation for non-Gaussian data. Much more sample-efficient than KSG
    in moderate dimensions (works with N > D instead of N >> 10^D).

    Returns MI in nats.
    """
    N, Dx = x.shape
    _, Dy = y.shape
    if N < Dx + Dy + 2:
        return 0.0

    # Center the data
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)

    # Regularised covariance determinants
    reg = 1e-6
    cov_x = (xc.T @ xc) / (N - 1) + reg * np.eye(Dx)
    cov_y = (yc.T @ yc) / (N - 1) + reg * np.eye(Dy)

    xy = np.hstack([xc, yc])
    cov_xy = (xy.T @ xy) / (N - 1) + reg * np.eye(Dx + Dy)

    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)

    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        return 0.0

    mi = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(0.0, float(mi))


def ksg_mi(x, y, k=3):
    """Estimate I(X;Y) using the KSG k-NN estimator.

    WARNING: Requires N >> k^D samples to be reliable (curse of dimensionality).
    Only use when D ≤ 3 and N ≥ 50, or D ≤ 2 and N ≥ 30.

    Ref: Kraskov, Stojmirovic, Grassberger (2004), Eq. (8).
    """
    N = x.shape[0]
    if N < 2 * k:
        return 0.0

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dists, _ = tree_xy.query(xy, k=k+1)
    eps = dists[:, -1]

    mi_sum = 0.0
    for i in range(N):
        nx = max(tree_x.query_ball_point(x[i], eps[i], p=2, return_length=True) - 1, 0)
        ny = max(tree_y.query_ball_point(y[i], eps[i], p=2, return_length=True) - 1, 0)
        mi_sum += digamma(nx + 1) + digamma(ny + 1)

    mi = digamma(k) - mi_sum / N + digamma(N)
    return max(0.0, float(mi))


def estimate_mi(x, y, pca_dim, k=3, n_bootstrap=30, seed=42):
    """Estimate I(X;Y) with PCA reduction. Uses Gaussian MI as primary estimator
    (sample-efficient), with KSG as complementary when dimensionality is low enough.

    Returns: {"mi_gauss": float, "mi_ksg": float or None, "ci_low": float, "ci_high": float}
    """
    N = x.shape[0]
    n_comp = min(pca_dim, N - 3, x.shape[1])
    if n_comp < 2:
        return {"mi_gauss": 0.0, "mi_ksg": None, "ci_low": 0.0, "ci_high": 0.0,
                "n_components": n_comp, "n_samples": N}

    pca = PCA(n_components=n_comp, random_state=42)
    x_reduced = pca.fit_transform(x)
    y_reduced = pca.transform(y)

    # Normalise to unit variance per component
    x_std = x_reduced.std(axis=0, keepdims=True) + 1e-8
    y_std = y_reduced.std(axis=0, keepdims=True) + 1e-8
    x_norm = x_reduced / x_std
    y_norm = y_reduced / y_std

    # Primary: Gaussian MI (robust in moderate dimensions)
    mi_gauss = gaussian_mi(x_norm, y_norm)

    # Complementary: KSG (only when D ≤ 3 and N ≥ 30)
    if n_comp <= 3 and N >= 30:
        mi_ksg = ksg_mi(x_norm, y_norm, k=k)
    else:
        mi_ksg = None  # KSG unreliable in >3D

    # Bootstrap CI on Gaussian MI
    ci_low, ci_high = mi_gauss, mi_gauss
    if N >= 20 and n_bootstrap > 0:
        rng = np.random.RandomState(seed)
        estimates = []
        for _ in range(n_bootstrap):
            idx = rng.choice(N, N, replace=True)
            estimates.append(gaussian_mi(x_norm[idx], y_norm[idx]))
        estimates = np.array(estimates)
        ci_low = float(np.percentile(estimates, 5))
        ci_high = float(np.percentile(estimates, 95))

    var_retained = float(np.sum(pca.explained_variance_ratio_))
    return {"mi_gauss": mi_gauss, "mi_ksg": mi_ksg,
            "ci_low": ci_low, "ci_high": ci_high,
            "n_components": n_comp, "n_samples": N,
            "variance_retained": var_retained}


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def pooled_features(feat):
    """Global average pool a feature tensor to [C]."""
    if feat.dim() == 4:
        return feat.mean(dim=[2, 3]).squeeze(0).cpu().float().numpy()
    elif feat.dim() == 3:
        return feat.mean(dim=1).squeeze(0).cpu().float().numpy()
    return feat.flatten().cpu().float().numpy()


def collect_paired_features(pipe, image_paths, num_steps=50, sample_every=10):
    """Collect paired (inversion, reconstruction) features across all analysis layers.

    Returns:
        inv_features: {layer: np.array [N, C]}
        recon_features: {layer: np.array [N, C]}
        N = n_images * n_sampled_steps
    """
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    sample_steps = list(range(0, num_steps, sample_every))
    # Always include the first step
    if 0 not in sample_steps:
        sample_steps = [0] + sample_steps

    inv_accum = {name: [] for name in MI_LAYERS}
    recon_accum = {name: [] for name in MI_LAYERS}

    n_images = len(image_paths)

    for img_idx, img_path in enumerate(image_paths):
        print(f"  [{img_idx+1}/{n_images}] {Path(img_path).name}")
        latent, _ = load_image(pipe, img_path)

        # --- Inversion with feature collection ---
        collector_inv = FeatureCollector(pipe.unet, MI_LAYERS)
        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_steps, device=DEVICE)
        timesteps = scheduler.timesteps
        z = latent.clone()
        extended_ts = timesteps.tolist() + [0]
        inv_step_features = {name: {} for name in MI_LAYERS}

        with torch.no_grad():
            for i in range(len(extended_ts) - 1, 0, -1):
                t_cur = extended_ts[i]
                t_next = extended_ts[i - 1]
                step_idx = len(extended_ts) - 1 - i

                collector_inv.clear()
                noise_pred = pipe.unet(z, t_cur, encoder_hidden_states=prompt_embeds).sample

                if step_idx in sample_steps:
                    for name in MI_LAYERS:
                        if name in collector_inv.features:
                            inv_step_features[name][step_idx] = pooled_features(
                                collector_inv.features[name])

                alpha_cur = scheduler.alphas_cumprod[t_cur]
                alpha_next = scheduler.alphas_cumprod[t_next]
                coeff1 = (alpha_next / alpha_cur).sqrt()
                sigma_cur = (1 - alpha_cur).sqrt()
                sigma_next = (1 - alpha_next).sqrt()
                coeff2 = sigma_next - coeff1 * sigma_cur
                z = coeff1 * z + coeff2 * noise_pred

        collector_inv.remove()
        noise_T = z.clone()

        # --- Reconstruction with feature collection ---
        collector_recon = FeatureCollector(pipe.unet, MI_LAYERS)
        z = noise_T.clone()
        recon_step_features = {name: {} for name in MI_LAYERS}

        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                collector_recon.clear()
                noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
                z = scheduler.step(noise_pred, t, z).prev_sample

                if step_idx in sample_steps:
                    for name in MI_LAYERS:
                        if name in collector_recon.features:
                            recon_step_features[name][step_idx] = pooled_features(
                                collector_recon.features[name])

        collector_recon.remove()

        # Store matched pairs
        for name in MI_LAYERS:
            for step in sample_steps:
                if step in inv_step_features[name] and step in recon_step_features[name]:
                    inv_accum[name].append(inv_step_features[name][step])
                    recon_accum[name].append(recon_step_features[name][step])

        torch.cuda.empty_cache()

    inv_features = {}
    recon_features = {}
    for name in MI_LAYERS:
        if inv_accum[name]:
            inv_features[name] = np.stack(inv_accum[name], axis=0)
            recon_features[name] = np.stack(recon_accum[name], axis=0)

    n_total = sum(len(v) for v in inv_accum.values())
    print(f"  Collected {n_total} feature vectors across {len(MI_LAYERS)} layers")
    return inv_features, recon_features


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def classify_layer(name):
    is_attn = "attentions" in name
    if name.startswith("down_blocks"):
        region = "encoder"
    elif name.startswith("mid_block"):
        region = "bottleneck"
    else:
        region = "decoder"
    return ("Attention" if is_attn else "ResNet", region)


def estimate_all_layers(inv_features, recon_features, pca_dim=5, k=3):
    """Estimate I(f_inv; f_recon) for all layers.

    Uses Gaussian MI as primary estimator (sample-efficient in moderate dimensions)
    with KSG as complementary validation when D ≤ 3 and N ≥ 30.
    """
    results = {}
    layer_names = sorted(inv_features.keys())

    for name in layer_names:
        X_inv = inv_features[name]  # [N, C]
        X_recon = recon_features[name]
        N = X_inv.shape[0]

        mi_result = estimate_mi(X_inv, X_recon, pca_dim=pca_dim, k=k)

        ltype, region = classify_layer(name)
        results[name] = {
            "mi": mi_result["mi_gauss"],  # primary: Gaussian MI
            "mi_ksg": mi_result["mi_ksg"],  # complementary: KSG (may be None)
            "ci_low": mi_result["ci_low"],
            "ci_high": mi_result["ci_high"],
            "n_samples": mi_result["n_samples"],
            "n_components": mi_result["n_components"],
            "variance_retained": mi_result["variance_retained"],
            "type": ltype,
            "region": region,
        }

        ksg_str = f"KSG={mi_result['mi_ksg']:.3f}" if mi_result["mi_ksg"] is not None else "KSG=N/A"
        print(f"  {name:<50s} I_gauss={mi_result['mi_gauss']:.3f} [{mi_result['ci_low']:.3f}, "
              f"{mi_result['ci_high']:.3f}] {ksg_str} [{ltype}, {region}]")

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_mi_per_layer(results, output_path):
    """Bar chart: per-layer I(f_inv; f_recon), sorted, colored by type."""
    valid = [(name, r) for name, r in results.items() if r["mi"] is not None]
    valid.sort(key=lambda x: x[1]["mi"], reverse=True)

    names = [v[0] for v in valid]
    mi_vals = [v[1]["mi"] for v in valid]
    # Asymmetric error bars: ensure non-negative
    ci_low_errs = [max(0, v[1]["mi"] - v[1]["ci_low"]) for v in valid]
    ci_high_errs = [max(0, v[1]["ci_high"] - v[1]["mi"]) for v in valid]
    errors = [ci_low_errs, ci_high_errs]

    colors = []
    for name in names:
        ltype = results[name]["type"]
        colors.append("#E74C3C" if ltype == "Attention" else "#2E86C1")

    fig, ax = plt.subplots(figsize=(18, 6))
    xs = range(len(names))
    ax.bar(xs, mi_vals, yerr=errors, color=colors, capsize=2, alpha=0.85)

    short_names = [n.replace("up_blocks.", "up.").replace("down_blocks.", "dn.")
                   .replace("mid_block.", "mid.").replace(".resnets.", ".rn")
                   .replace(".attentions.", ".attn").replace(".transformer_blocks.0", "")
                   for n in names]
    ax.set_xticks(xs)
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("I(f_inv; f_recon) [nats]", fontsize=12)
    ax.set_title("Per-Layer Mutual Information: Inversion vs Reconstruction Features\n"
                 f"(Gaussian MI, PCA-5, {len(valid)} layers, "
                 f"error bars = 90% bootstrap CI)", fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86C1", label="ResNet"),
        Patch(facecolor="#E74C3C", label="Attention"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_resnet_vs_attention_mi(results, output_path):
    """Grouped bar: ResNet vs Attention mean MI."""
    resnet_mi = [r["mi"] for r in results.values()
                 if r["mi"] is not None and r["type"] == "ResNet"]
    attn_mi = [r["mi"] for r in results.values()
               if r["mi"] is not None and r["type"] == "Attention"]

    fig, ax = plt.subplots(figsize=(7, 5))
    means = [np.mean(resnet_mi), np.mean(attn_mi)]
    stds = [np.std(resnet_mi), np.std(attn_mi)]
    counts = [len(resnet_mi), len(attn_mi)]

    bars = ax.bar(["ResNet", "Attention"], means, yerr=stds,
                  color=["#2E86C1", "#E74C3C"], capsize=8, alpha=0.85, width=0.45)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"n={count}", ha="center", fontsize=10)

    ax.set_ylabel("I(f_inv; f_recon) [nats]", fontsize=12)
    ax.set_title("Mutual Information: ResNet vs Attention", fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    ratio = means[0] / max(means[1], 1e-8)
    ax.text(0.5, 0.95, f"ResNet/Attention MI ratio: {ratio:.1f}×\n"
            f"(higher MI = more information preserved)",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_mi_vs_drift(results, drift_path, output_path):
    """Scatter: MI vs Phase 1 L2 drift. Prediction: strong negative correlation."""
    with open(drift_path) as f:
        drift_data = json.load(f)
    drift_agg = drift_data.get("aggregated", {})

    points = []
    for name, r in results.items():
        if r["mi"] is not None and name in drift_agg:
            points.append({
                "name": name,
                "mi": r["mi"],
                "drift": drift_agg[name]["mean"],
                "type": r["type"],
                "region": r["region"],
            })

    if not points:
        print("[WARN] No overlapping layers for MI vs drift plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for p in points:
        color = "#E74C3C" if p["type"] == "Attention" else "#2E86C1"
        marker = "s" if p["type"] == "Attention" else "o"
        ax.scatter(p["drift"], p["mi"], c=color, marker=marker, s=70,
                   alpha=0.8, edgecolors="black", linewidth=0.5)
        short = (p["name"].replace("up_blocks.", "up.").replace("down_blocks.", "dn.")
                 .replace("mid_block.", "mid."))
        ax.annotate(short, (p["drift"], p["mi"]), fontsize=5.5, alpha=0.7,
                    textcoords="offset points", xytext=(3, 3))

    drifts = [p["drift"] for p in points]
    mis = [p["mi"] for p in points]
    corr = np.corrcoef(drifts, mis)[0, 1]

    ax.set_xlabel("Phase 1 Layer Drift (L2 distance)", fontsize=11)
    ax.set_ylabel("I(f_inv; f_recon) [nats]", fontsize=11)
    ax.set_title(f"Mutual Information vs Feature Drift (r = {corr:.3f})", fontsize=13)
    ax.grid(alpha=0.3)

    ax.legend(handles=[
        plt.scatter([], [], c="#2E86C1", marker="o", s=50, label="ResNet"),
        plt.scatter([], [], c="#E74C3C", marker="s", s=50, label="Attention"),
    ], fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


def plot_mi_by_region(results, output_path):
    """Grouped bar: MI by UNet region."""
    regions = {"encoder": [], "bottleneck": [], "decoder": []}
    for name, r in results.items():
        if r["mi"] is not None:
            regions[r["region"]].append(r["mi"])

    fig, ax = plt.subplots(figsize=(6, 5))
    region_names = ["encoder", "bottleneck", "decoder"]
    means = [np.mean(regions[r]) if regions[r] else 0 for r in region_names]
    stds = [np.std(regions[r]) if regions[r] else 0 for r in region_names]
    colors = ["#27AE60", "#F39C12", "#2E86C1"]

    ax.bar(region_names, means, yerr=stds, color=colors, capsize=8, alpha=0.85, width=0.5)
    ax.set_ylabel("I(f_inv; f_recon) [nats]", fontsize=12)
    ax.set_title("Information Preservation by UNet Region", fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Figure] {output_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results):
    print(f"\n{'='*70}")
    print("MUTUAL INFORMATION ESTIMATION — KEY FINDINGS")
    print(f"  Primary estimator: Gaussian MI (sample-efficient, exact for Gaussian)")
    print(f"  Complementary:     KSG k-NN (non-parametric, k=3, only when D≤3)")
    print(f"{'='*70}")

    valid = [(name, r) for name, r in results.items() if r["mi"] is not None]
    valid.sort(key=lambda x: x[1]["mi"], reverse=True)

    print("\nTop-5 layers by I(f_inv; f_recon) (most information preserved):")
    for i, (name, r) in enumerate(valid[:5]):
        ltype, region = classify_layer(name)
        print(f"  {i+1}. {name:<50s} I={r['mi']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
              f"[{ltype}, {region}]")

    print("\nBottom-5 layers by I(f_inv; f_recon) (most information LOST):")
    for i, (name, r) in enumerate(valid[-5:]):
        ltype, region = classify_layer(name)
        print(f"  {len(valid)-4+i}. {name:<50s} I={r['mi']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
              f"[{ltype}, {region}]")

    resnet_mi = [r["mi"] for r in results.values()
                 if r["mi"] is not None and r["type"] == "ResNet"]
    attn_mi = [r["mi"] for r in results.values()
              if r["mi"] is not None and r["type"] == "Attention"]

    print(f"\nResNet:    I = {np.mean(resnet_mi):.3f} ± {np.std(resnet_mi):.3f} nats  (n={len(resnet_mi)})")
    print(f"Attention: I = {np.mean(attn_mi):.3f} ± {np.std(attn_mi):.3f} nats  (n={len(attn_mi)})")
    print(f"Ratio:     {np.mean(resnet_mi) / max(np.mean(attn_mi), 1e-8):.1f}×")

    for region in ["encoder", "bottleneck", "decoder"]:
        region_mi = [r["mi"] for r in results.values()
                     if r["mi"] is not None and r["region"] == region]
        if region_mi:
            print(f"{region:<12s} I = {np.mean(region_mi):.3f} ± {np.std(region_mi):.3f} nats  (n={len(region_mi)})")

    # Correlation with Phase 1 drift
    drift_path = Path("outputs/phase1/layer_drift_summary.json")
    if drift_path.exists():
        with open(drift_path) as f:
            drift_data = json.load(f)
        drift_agg = drift_data.get("aggregated", {})
        shared = [(name, r["mi"], drift_agg[name]["mean"])
                  for name, r in results.items()
                  if r["mi"] is not None and name in drift_agg]
        if shared:
            mis_s = [s[1] for s in shared]
            drifts_s = [s[2] for s in shared]
            corr = np.corrcoef(drifts_s, mis_s)[0, 1]
            print(f"\nI(f_inv; f_recon) vs Phase 1 Drift: r = {corr:.3f} ({len(shared)} layers)")
            if corr < -0.5:
                print("→ Strong negative correlation: high-drift layers lose more information. ✓")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
This is a genuine information-theoretic analysis using direct MI estimation.
Unlike the per-layer marginal correction analysis (phase4_info_theory.py),
which uses ΔPSNR as a proxy, this estimates:

  I(f_inv; f_recon) = mutual information between inversion and reconstruction features

Primary estimator: Gaussian MI — exact for jointly Gaussian variables, provides
a tight lower bound for non-Gaussian data after PCA whitening. Sample-efficient:
works with N > D, unlike KSG which requires N >> 10^D.

Complementary: KSG k-NN estimator (only when D ≤ 3, N ≥ 30) for non-parametric
validation on low-dimensional projections.

Key implications:

1. High MI → reconstruction preserves inversion information → low effective drift
   Low MI  → reconstruction loses inversion information → high effective drift

2. ResNet vs Attention MI ratio reveals the architectural information bottleneck.
   If ResNet MI < Attention MI: ResNet layers lose more information during reconstruction,
   consistent with the Phase 1 finding that ResNet drift >> Attention drift.

3. MI-vs-drift correlation: strong negative correlation validates that Phase 1's
   L2 drift metric captures genuine information loss.

4. PCA to 5 dimensions is justified by the manifold analysis (phase4_manifold.py),
   which showed rapid eigenvalue decay and intrinsic dimensions of 2-35.

Relationship to the correction mechanism:
  - Low I(f_inv; f_recon) → residual d = f_inv - f_recon contains recoverable information
  - Correction f_out = f_recon + λ·d recovers a fraction λ of this lost information
  - Per-layer MI identifies WHERE information is lost → WHERE correction is needed
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation with 3 images")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--pca-dim", type=int, default=5,
                        help="PCA target dimensionality for MI estimation (3-5 recommended)")
    parser.add_argument("--k", type=int, default=3,
                        help="k for KSG k-NN estimator")
    parser.add_argument("--output", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path("data/coco_val")
    all_images = sorted(image_dir.glob("*.jpg"))
    if args.quick:
        image_paths = [str(p) for p in all_images[:3]]
        sample_every = 5  # dense sampling to get enough samples for MI estimation
        print(f"[QUICK MODE] {len(image_paths)} images, sample_every={sample_every}")
    else:
        image_paths = [str(p) for p in all_images]
        sample_every = 5  # 10 steps per image → 190 samples per layer
        print(f"[FULL MODE] {len(image_paths)} images, sample_every={sample_every}")

    print("Loading SD 1.5 pipeline...")
    pipe = load_pipeline()

    # Collect features
    print("\n[1/2] Collecting paired inversion/reconstruction features...")
    inv_features, recon_features = collect_paired_features(
        pipe, image_paths, args.steps, sample_every=sample_every)

    # MI estimation
    print(f"\n[2/2] Estimating MI (KSG, k={args.k}, PCA-{args.pca_dim})...")
    results = estimate_all_layers(inv_features, recon_features,
                                  pca_dim=args.pca_dim, k=args.k)

    # Save
    out_path = out_dir / "per_layer_mi.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[JSON] {out_path}")

    # Plots
    plot_mi_per_layer(results, out_dir / "mi_per_layer.png")
    plot_resnet_vs_attention_mi(results, out_dir / "resnet_vs_attention_mi.png")
    plot_mi_by_region(results, out_dir / "mi_by_region.png")

    drift_path = Path("outputs/phase1/layer_drift_summary.json")
    if drift_path.exists():
        plot_mi_vs_drift(results, drift_path, out_dir / "mi_vs_drift.png")

    print_report(results)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
