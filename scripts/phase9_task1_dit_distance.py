#!/usr/bin/env python
"""
Phase 9 Task 1: DiT-S/2 paradigm-pair structural distance vs cross-architecture spectrum.

Computes d(eps, flow | DiT-S/2) and places it in the context of the full
cross-architecture distance matrix (SD 1.5, SDXL, HunyuanDiT, FLUX, SD 3.5).

If d(eps, flow) << min cross-architecture pair, this is strong quantitative
evidence for Claim 1's sub-conclusion: architecture dominates paradigm.

Usage:
    python scripts/phase9_task1_dit_distance.py

Output: outputs/phase9_task1/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_task1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Structural feature extraction (from dit_controlled_shared.py)
# ---------------------------------------------------------------------------

def extract_structural_features(drift: np.ndarray) -> np.ndarray:
    """4 features: peak_position(relative), n_peaks, gini_concentration, spread."""
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
# Data loaders
# ---------------------------------------------------------------------------

def load_unified_drift():
    """Load aggregated drift profiles from phase6_unified results."""
    with open(PROJECT_ROOT / "outputs" / "phase6_unified" / "results.json") as f:
        data = json.load(f)
    profiles = {}
    for arch_name, arch_data in data["architectures"].items():
        layers = arch_data["layers"]
        drift_vec = np.array([l["drift"] for l in layers])
        # Use short names
        short = arch_name.split(" (")[0].replace("SD ", "SD").replace(" ", "_")
        profiles[short] = {
            "drift": drift_vec,
            "n_layers": len(drift_vec),
            "full_name": arch_name,
        }
    return profiles


def load_dit_controlled_drift():
    """Load DiT-S/2 eps and flow drift profiles."""
    with open(PROJECT_ROOT / "outputs" / "train_controlled" / "diagnostics" / "comparison.json") as f:
        data = json.load(f)
    per_layer = data["per_layer_drift"]
    profiles = {}
    for variant in ["eps", "flow"]:
        d = per_layer[variant]
        layer_names = sorted(d.keys(), key=lambda x: int(x.split(".")[-1]) if "blocks" in x else 0)
        drift_vec = np.array([d[k] for k in layer_names])
        profiles[f"DiT-S/2_{variant}"] = {
            "drift": drift_vec,
            "n_layers": len(drift_vec),
            "full_name": f"DiT-S/2 ({variant}-prediction)",
            "layer_names": layer_names,
        }
    return profiles


def load_sd35_drift():
    """Load SD 3.5 aggregated drift profile."""
    with open(PROJECT_ROOT / "outputs" / "sd35_phase1" / "layer_drift_summary.json") as f:
        data = json.load(f)
    agg = data["aggregated"]
    # Sort by block index
    layer_names = sorted(agg.keys(), key=lambda x: int(x.split("_")[-1]))
    drift_vec = np.array([agg[k]["mean"] for k in layer_names])
    return {
        "SD3.5": {
            "drift": drift_vec,
            "n_layers": len(drift_vec),
            "full_name": "SD 3.5 (MM-DiT, Rectified Flow)",
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading drift profiles...")
    profiles = load_unified_drift()
    profiles.update(load_sd35_drift())
    profiles.update(load_dit_controlled_drift())

    names = list(profiles.keys())
    n = len(names)
    print(f"Loaded {n} profiles: {names}")

    # Compute pairwise structural distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = structural_distance(profiles[names[i]]["drift"],
                                    profiles[names[j]]["drift"])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Results
    results = {
        "architectures": [profiles[nm]["full_name"] for nm in names],
        "short_names": names,
        "n_layers": [profiles[nm]["n_layers"] for nm in names],
        "structural_distance_matrix": {names[i]: {names[j]: float(dist_matrix[i, j])
                                                   for j in range(n)} for i in range(n)},
    }

    # Find key comparisons
    dit_eps_flow_dist = dist_matrix[names.index("DiT-S/2_eps"), names.index("DiT-S/2_flow")]
    results["dit_eps_flow_distance"] = float(dit_eps_flow_dist)

    # Cross-architecture distances (excluding DiT-S/2 variants)
    full_arch_names = [nm for nm in names if not nm.startswith("DiT-S/2")]
    full_indices = [names.index(nm) for nm in full_arch_names]
    cross_arch_dists = []
    for i_idx, i in enumerate(full_indices):
        for j in full_indices[i_idx + 1:]:
            cross_arch_dists.append({
                "pair": f"{names[i]} vs {names[j]}",
                "distance": float(dist_matrix[i, j]),
            })
    cross_arch_dists.sort(key=lambda x: x["distance"])
    results["cross_architecture_distances"] = cross_arch_dists
    results["min_cross_arch_distance"] = cross_arch_dists[0]["distance"]
    results["ratio_dit_to_min_cross"] = float(dit_eps_flow_dist / cross_arch_dists[0]["distance"])

    # Print summary
    print(f"\n{'='*60}")
    print(f"DiT-S/2 d(eps, flow) = {dit_eps_flow_dist:.4f}")
    print(f"Min cross-architecture d   = {cross_arch_dists[0]['distance']:.4f} ({cross_arch_dists[0]['pair']})")
    print(f"Ratio = {results['ratio_dit_to_min_cross']:.2f}x")
    print(f"\nFull distance ranking:")
    print(f"{'Pair':<35} {'d':>8}")
    print(f"{'-'*45}")
    # Insert DiT-S/2 at appropriate position
    all_dists = cross_arch_dists + [{"pair": "DiT-S/2 eps vs flow", "distance": float(dit_eps_flow_dist)}]
    all_dists.sort(key=lambda x: x["distance"])
    for entry in all_dists:
        marker = " <-- SAME ARCH, DIFF PARADIGM" if "DiT-S/2" in entry["pair"] else ""
        print(f"{entry['pair']:<35} {entry['distance']:>8.4f}{marker}")

    # Save
    out_path = OUT_DIR / "distance_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Distance matrix heatmap
    im = ax1.imshow(dist_matrix, cmap="YlOrRd", aspect="equal")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j:
                color = "white" if dist_matrix[i, j] > 0.8 else "black"
                ax1.text(j, i, f"{dist_matrix[i, j]:.3f}", ha="center", va="center",
                         fontsize=8, color=color)
    ax1.set_title("Structural Distance Matrix\n(4-feature Euclidean)", fontsize=12)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="d")

    # Panel B: Bar chart sorted by distance
    bar_labels = [e["pair"] for e in all_dists]
    bar_values = [e["distance"] for e in all_dists]
    colors = ["#d62728" if "DiT-S/2" in lbl else "#1f77b4" for lbl in bar_labels]
    bars = ax2.barh(range(len(bar_labels)), bar_values, color=colors)
    ax2.set_yticks(range(len(bar_labels)))
    ax2.set_yticklabels(bar_labels, fontsize=8)
    ax2.set_xlabel("Structural Distance d")
    ax2.axvline(x=dit_eps_flow_dist, color="#d62728", linestyle="--", alpha=0.5,
                label=f"d(eps,flow)={dit_eps_flow_dist:.3f}")
    ax2.legend(fontsize=9)
    ax2.set_title("Ranked Pairwise Structural Distances", fontsize=12)
    ax2.invert_yaxis()

    plt.tight_layout()
    fig_path = OUT_DIR / "distance_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # ---- Interpretation ----
    if results["ratio_dit_to_min_cross"] < 0.5:
        verdict = "STRONG SUPPORT: d(eps,flow) is much smaller than any cross-architecture pair."
    elif results["ratio_dit_to_min_cross"] < 1.0:
        verdict = "MODERATE SUPPORT: d(eps,flow) is smaller than the closest cross-architecture pair."
    else:
        verdict = "WEAK / NO SUPPORT: d(eps,flow) is comparable to or larger than some cross-architecture distances."
    results["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    # Update JSON with verdict
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
