#!/usr/bin/env python
"""
Phase 9 Task 5: Normalization ablation for structural distance.

Compares 5 normalization schemes applied before structural feature extraction:
  1. Percentile-95 (current default)
  2. Percentile-98 (used in formation_vs_drift.py)
  3. Min-max [0,1]
  4. Z-score standardization
  5. L2 normalization (unit vector)

For each scheme, computes the full structural distance matrix across all
architectures (SD 1.5, SDXL, HunyuanDiT, FLUX, SD 3.5, DiT-S/2 eps, DiT-S/2 flow).
Reports whether d(eps, flow) << min cross-architecture holds under all schemes.

Usage:
    python scripts/phase9_task5_norm_ablation.py

Output: outputs/phase9_task5/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_task5"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Structural features (raw drift, no internal normalization)
# ---------------------------------------------------------------------------

def extract_structural_features(drift: np.ndarray) -> np.ndarray:
    L = len(drift)
    d = np.maximum(drift, 0.0)
    d_max = d.max()
    if d_max < 1e-12:
        return np.array([0.5, 0.0, 0.0, 0.0])
    peak_pos = float(np.argmax(d)) / max(L - 1, 1)
    n_peaks = float(np.sum(d > 0.7 * d_max))
    d_sorted = np.sort(d)
    n = len(d_sorted)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * d_sorted)) / (n * np.sum(d_sorted) + 1e-12) - (n + 1) / n
    spread = float(np.std(d) / (np.mean(d) + 1e-12))
    return np.array([peak_pos, n_peaks, gini, spread])


def structural_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.linalg.norm(extract_structural_features(d1) - extract_structural_features(d2)))


# ---------------------------------------------------------------------------
# Normalization schemes
# ---------------------------------------------------------------------------

def normalize_percentile(vals, pct=95):
    """Clip at pct-th percentile, then divide by max."""
    v = np.array(vals, dtype=float)
    vmax = np.percentile(v, pct)
    if vmax <= 0:
        vmax = v.max() or 1.0
    return np.clip(v / vmax, 0, 1.0)


def normalize_minmax(vals):
    """Min-max to [0, 1]."""
    v = np.array(vals, dtype=float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def normalize_zscore(vals):
    """Z-score standardization: (x - mean) / std."""
    v = np.array(vals, dtype=float)
    mu, sigma = v.mean(), v.std()
    if sigma < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sigma


def normalize_l2(vals):
    """L2 normalization to unit vector."""
    v = np.array(vals, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def normalize_layernorm(vals):
    """LayerNorm-style: (x - mean) / std, then optional scaling (here identity)."""
    return normalize_zscore(vals)


NORMALIZERS = {
    "percentile95": lambda v: normalize_percentile(v, 95),
    "percentile98": lambda v: normalize_percentile(v, 98),
    "minmax": normalize_minmax,
    "zscore": normalize_zscore,
    "l2": normalize_l2,
    "layernorm": normalize_layernorm,
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_profiles():
    """Load all drift profiles (raw, un-normalized). Returns {name: drift_vector}."""
    profiles = {}

    # 4 architectures from unified results
    with open(PROJECT_ROOT / "outputs" / "phase6_unified" / "results.json") as f:
        data = json.load(f)
    for arch_name, arch_data in data["architectures"].items():
        layers = arch_data["layers"]
        drift_vec = np.array([l["drift"] for l in layers])
        short = arch_name.split(" (")[0].replace("SD ", "SD").replace(" ", "_")
        profiles[short] = drift_vec

    # SD 3.5
    with open(PROJECT_ROOT / "outputs" / "sd35_phase1" / "layer_drift_summary.json") as f:
        data = json.load(f)
    agg = data["aggregated"]
    layer_names = sorted(agg.keys(), key=lambda x: int(x.split("_")[-1]))
    profiles["SD3.5"] = np.array([agg[k]["mean"] for k in layer_names])

    # DiT-S/2 eps and flow
    with open(PROJECT_ROOT / "outputs" / "train_controlled" / "diagnostics" / "comparison.json") as f:
        data = json.load(f)
    for variant in ["eps", "flow"]:
        d = data["per_layer_drift"][variant]
        layer_names = sorted(d.keys(), key=lambda x: int(x.split(".")[-1]) if "blocks" in x else 0)
        profiles[f"DiT-S/2_{variant}"] = np.array([d[k] for k in layer_names])

    return profiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading raw drift profiles...")
    raw_profiles = load_all_profiles()
    names = sorted(raw_profiles.keys())
    print(f"Architectures: {names}")
    for nm in names:
        print(f"  {nm}: {len(raw_profiles[nm])} layers, range [{raw_profiles[nm].min():.2f}, {raw_profiles[nm].max():.2f}]")

    results = {"architectures": names, "schemes": {}}

    # Compute distance matrix under each normalization
    for scheme_name, norm_fn in NORMALIZERS.items():
        # Normalize all profiles
        norm_profiles = {nm: norm_fn(raw_profiles[nm]) for nm in names}

        n = len(names)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = structural_distance(norm_profiles[names[i]], norm_profiles[names[j]])
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # Extract key comparisons
        dit_key = ("DiT-S/2_eps", "DiT-S/2_flow")
        dit_i = names.index(dit_key[0])
        dit_j = names.index(dit_key[1])
        dit_dist = float(dist_mat[dit_i, dit_j])

        # Cross-architecture distances (exclude DiT-S/2 variants)
        full_idx = [i for i, nm in enumerate(names) if not nm.startswith("DiT-S/2")]
        cross_dists = []
        for a in range(len(full_idx)):
            for b in range(a + 1, len(full_idx)):
                cross_dists.append(float(dist_mat[full_idx[a], full_idx[b]]))
        min_cross = min(cross_dists)

        results["schemes"][scheme_name] = {
            "distance_matrix": {names[i]: {names[j]: float(dist_mat[i, j])
                                           for j in range(n)} for i in range(n)},
            "dit_eps_flow_distance": dit_dist,
            "min_cross_architecture_distance": min_cross,
            "ratio": dit_dist / min_cross if min_cross > 0 else float("inf"),
        }

        print(f"\n{scheme_name}: d(eps,flow)={dit_dist:.4f}, min_cross={min_cross:.4f}, ratio={dit_dist/min_cross:.2f}x")

    # ---- Stability summary ----
    print(f"\n{'='*70}")
    print("Stability of d(eps,flow) / min_cross ratio across normalizations:")
    print(f"{'Scheme':<16} {'d(eps,flow)':>12} {'min_cross':>12} {'Ratio':>8}")
    print("-" * 52)
    ratios = {}
    for scheme_name in NORMALIZERS:
        s = results["schemes"][scheme_name]
        ratios[scheme_name] = s["ratio"]
        print(f"{scheme_name:<16} {s['dit_eps_flow_distance']:>12.4f} {s['min_cross_architecture_distance']:>12.4f} {s['ratio']:>8.2f}x")

    ratio_values = list(ratios.values())
    results["ratio_stats"] = {
        "mean": float(np.mean(ratio_values)),
        "std": float(np.std(ratio_values)),
        "min": float(np.min(ratio_values)),
        "max": float(np.max(ratio_values)),
        "cv": float(np.std(ratio_values) / np.mean(ratio_values)) if np.mean(ratio_values) > 0 else 0,
    }

    # Verdict
    if all(r < 1.0 for r in ratio_values):
        verdict = "ROBUST: Under ALL normalization schemes, d(eps,flow) < min cross-architecture distance."
    elif all(r < 0.5 for r in ratio_values):
        verdict = "STRONG: Under ALL schemes, d(eps,flow) << min cross-architecture distance (ratio < 0.5)."
    elif results["ratio_stats"]["cv"] < 0.3:
        verdict = "STABLE: Ratio is consistent across normalization schemes (CV < 0.3)."
    else:
        verdict = "SENSITIVE: Ratio varies substantially with normalization choice — report range."
    results["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    # Save
    out_path = OUT_DIR / "normalization_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax_idx, (scheme_name, norm_fn) in enumerate(NORMALIZERS.items()):
        ax = axes[ax_idx]
        s = results["schemes"][scheme_name]
        dist_mat = np.array([[s["distance_matrix"][names[i]][names[j]]
                               for j in range(len(names))] for i in range(len(names))])

        im = ax.imshow(dist_mat, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    color = "white" if dist_mat[i, j] > np.median(dist_mat[dist_mat > 0]) else "black"
                    ax.text(j, i, f"{dist_mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color=color)
        ax.set_title(f"{scheme_name}  (ratio={s['ratio']:.2f}x)", fontsize=10)

    # Hide extra subplot if any
    for ax in axes[len(NORMALIZERS):]:
        ax.set_visible(False)

    plt.suptitle("Structural Distance Matrix Under Different Normalization Schemes", fontsize=14)
    plt.tight_layout()
    fig_path = OUT_DIR / "normalization_ablation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
