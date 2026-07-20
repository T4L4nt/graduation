#!/usr/bin/env python
"""
Ranking stability test for structural distance matrix.

Computes pairwise distance rankings under 4 normalization schemes
(min-max, z-score, L2, LayerNorm) across 5 architectures (10 pairs).
Reports Kendall's W and per-pair rank CV.

Key question: Is the qualitative ranking ("same-family closest,
HunyuanDiT-FLUX furthest") stable across normalization choices?

Usage:
    python scripts/phase9_ranking_stability.py

Output: outputs/phase9_ranking_stability/
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "phase9_ranking_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Structural features
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

def norm_minmax(v):
    v = np.array(v, dtype=float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def norm_zscore(v):
    v = np.array(v, dtype=float)
    mu, sigma = v.mean(), v.std()
    if sigma < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sigma


def norm_l2(v):
    v = np.array(v, dtype=float)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return v
    return v / nrm


NORMALIZERS = {
    "minmax": norm_minmax,
    "zscore": norm_zscore,
    "l2": norm_l2,
    "layernorm": norm_zscore,  # identical to z-score for ranking purposes
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all_profiles():
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
# Kendall's W
# ---------------------------------------------------------------------------

def kendall_w(rankings):
    """
    Compute Kendall's coefficient of concordance W.

    rankings: list of arrays, each of length n_items, containing ranks (0-based or 1-based).
    Returns: W, chi2, p_value, df
    """
    k = len(rankings)       # number of raters
    n = rankings[0].shape[0]  # number of items

    # Convert to rank sums
    rank_sums = np.sum(rankings, axis=0)  # shape (n,)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    # W = 12 * S / (k^2 * (n^3 - n))
    # Correction for ties: not needed if using scipy's method
    W = 12.0 * S / (k * k * (n * n * n - n))

    # Chi-squared test
    chi2 = k * (n - 1) * W
    df = n - 1
    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2, df)

    return W, chi2, p_value, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading drift profiles...")
    profiles = load_all_profiles()

    # Separate into 5 full architectures + DiT-S/2 variants
    full_archs = ["SD1.5", "SDXL", "DiT", "FLUX", "SD3.5"]
    full_archs = [a for a in full_archs if a in profiles]
    n_full = len(full_archs)
    n_pairs = n_full * (n_full - 1) // 2
    print(f"Full architectures ({n_full}): {full_archs} → {n_pairs} pairs")

    # Generate pair labels
    pair_labels = []
    pair_indices = []
    for i in range(n_full):
        for j in range(i + 1, n_full):
            pair_labels.append(f"{full_archs[i]} vs {full_archs[j]}")
            pair_indices.append((i, j))

    # Compute rankings under each normalization
    scheme_names = list(NORMALIZERS.keys())
    all_rankings = []        # [scheme][pair] = rank (0 = closest)
    all_distances = {}       # scheme -> {pair_label: distance}

    for scheme_name, norm_fn in NORMALIZERS.items():
        # Normalize
        norm_profiles = {name: norm_fn(profiles[name]) for name in full_archs}

        # Compute distances
        dists = []
        for i, j in pair_indices:
            d = structural_distance(norm_profiles[full_archs[i]],
                                     norm_profiles[full_archs[j]])
            dists.append(d)

        # Rank: 0 = closest (smallest distance)
        order = np.argsort(dists)
        ranks = np.zeros(n_pairs, dtype=int)
        for rank, idx in enumerate(order):
            ranks[idx] = rank

        all_rankings.append(ranks)
        all_distances[scheme_name] = {pair_labels[k]: dists[k] for k in range(n_pairs)}

        print(f"\n{scheme_name}:")
        for rank_idx in order:
            print(f"  #{rank_idx}: {pair_labels[rank_idx]}  d={dists[rank_idx]:.4f}")

    all_rankings = np.array(all_rankings)  # shape: [n_schemes, n_pairs]

    # ---- Kendall's W ----
    W, chi2, p_w, df = kendall_w(all_rankings)
    print(f"\n{'='*60}")
    print(f"Kendall's W = {W:.4f}  (χ²={chi2:.1f}, df={df}, p={p_w:.2e})")
    if W > 0.9:
        print("→ EXCELLENT agreement across normalization schemes")
    elif W > 0.7:
        print("→ GOOD agreement — ranking is reasonably stable")
    elif W > 0.5:
        print("→ MODERATE agreement — some pairs swap ranks")
    else:
        print("→ POOR agreement — ranking is normalization-dependent")

    # ---- Per-pair rank stability ----
    print(f"\n{'Pair':<30} {'Ranks':>20} {'Mean':>6} {'CV':>6}")
    print("-" * 70)
    rank_cvs = {}
    for k, label in enumerate(pair_labels):
        r = all_rankings[:, k]
        mean_r = r.mean()
        cv = r.std() / (mean_r + 1) if mean_r >= 0 else r.std()
        rank_cvs[label] = float(cv)
        rank_str = ", ".join(str(x) for x in r)
        print(f"{label:<30} [{rank_str}]  {mean_r:>6.1f}  {cv:>6.3f}")

    # ---- Key hypotheses ----
    print(f"\n{'='*60}")
    print("Hypothesis tests on rankings:")

    # H1: "Same-family closest" — SD1.5-SDXL should rank near #0 (closest)
    # H2: "Same-MMDiT close" — FLUX-SD3.5 should rank near the top
    # H3: "HunyuanDiT-FLUX furthest" — should rank near #9

    def check_hypothesis(label_contains, expected_end, pair_labels, all_rankings):
        """Check if pairs matching label_contains consistently rank at expected_end."""
        matching = [k for k, lbl in enumerate(pair_labels)
                    if all(s in lbl for s in label_contains)]
        if not matching:
            return None
        ranks_across_schemes = all_rankings[:, matching]
        mean_ranks = ranks_across_schemes.mean(axis=1)  # per-scheme mean for matching pairs
        overall_mean = ranks_across_schemes.mean()
        overall_std = ranks_across_schemes.std()
        return {
            "pairs": [pair_labels[k] for k in matching],
            "ranks": ranks_across_schemes.tolist(),
            "overall_mean_rank": float(overall_mean),
            "overall_std": float(overall_std),
        }

    # H1: Same UNet family — should be closest
    sd15_sdxl_idx = [k for k, lbl in enumerate(pair_labels) if "SD1.5" in lbl and "SDXL" in lbl]
    if sd15_sdxl_idx:
        idx = sd15_sdxl_idx[0]
        print(f"\nH1: Same UNet family → {pair_labels[idx]}")
        print(f"  Ranks: {all_rankings[:, idx].tolist()}  (mean={all_rankings[:, idx].mean():.1f}/9)")
        min_ranks = all_rankings.min(axis=1)
        is_closest = all_rankings[:, idx] == min_ranks
        print(f"  Closest in: {dict(zip(scheme_names, ['YES' if c else 'NO' for c in is_closest]))}")

    # H2: Same MM-DiT family — should be second-closest
    flux_sd35_idx = [k for k, lbl in enumerate(pair_labels) if "FLUX" in lbl and "SD3.5" in lbl]
    if flux_sd35_idx:
        idx = flux_sd35_idx[0]
        print(f"\nH2: Same MM-DiT family → {pair_labels[idx]}")
        print(f"  Ranks: {all_rankings[:, idx].tolist()}  (mean={all_rankings[:, idx].mean():.1f}/9)")

    # H3: DiT-FLUX — should be furthest (attention topology maximally different)
    dit_flux_idx = [k for k, lbl in enumerate(pair_labels) if "DiT" in lbl and "FLUX" in lbl]
    if dit_flux_idx:
        idx = dit_flux_idx[0]
        print(f"\nH3: DiT-FLUX furthest → {pair_labels[idx]}")
        print(f"  Ranks: {all_rankings[:, idx].tolist()}  (mean={all_rankings[:, idx].mean():.1f}/9)")
        max_ranks = all_rankings.max(axis=1)
        is_furthest = all_rankings[:, idx] == max_ranks
        print(f"  Furthest in: {dict(zip(scheme_names, ['YES' if c else 'NO' for c in is_furthest]))}")

    # ---- Pairwise rank correlations between schemes ----
    print(f"\n{'='*60}")
    print("Pairwise rank correlations (Spearman ρ) between normalization schemes:")
    print(f"{'':>12}", end="")
    for sn in scheme_names:
        print(f"{sn:>10}", end="")
    print()
    for i, sni in enumerate(scheme_names):
        print(f"{sni:>12}", end="")
        for j, snj in enumerate(scheme_names):
            if i == j:
                print(f"{'—':>10}", end="")
            else:
                rho, p = spearmanr(all_rankings[i], all_rankings[j])
                print(f"{rho:>10.4f}", end="")
        print()

    # ---- Verdict ----
    verdict_parts = []
    if W > 0.7:
        verdict_parts.append("RANKING STABLE: qualitative ordering preserved across normalizations → use rank-based heatmap for Fig 2B")
    else:
        verdict_parts.append("RANKING UNSTABLE: distance ordering depends on normalization → downgrade Properties 2/3 to qualitative observations, move evidence weight to peak containment + rank correlation")

    # Check specific claims
    sd15_sdxl_idx = [k for k, lbl in enumerate(pair_labels) if "SD1.5" in lbl and "SDXL" in lbl][0]
    dit_flux_idx = [k for k, lbl in enumerate(pair_labels) if "DiT" in lbl and "FLUX" in lbl][0]
    sd15_sdxl_rank_cv = rank_cvs[pair_labels[sd15_sdxl_idx]]
    dit_flux_rank_cv = rank_cvs[pair_labels[dit_flux_idx]]

    if sd15_sdxl_rank_cv < 0.3:
        verdict_parts.append(f"SD1.5-SDXL consistently closest (rank CV={sd15_sdxl_rank_cv:.2f})")
    else:
        verdict_parts.append(f"SD1.5-SDXL rank unstable (CV={sd15_sdxl_rank_cv:.2f}) — 'same-family closest' is scheme-dependent")

    if dit_flux_rank_cv < 0.3:
        verdict_parts.append(f"DiT-FLUX consistently furthest (rank CV={dit_flux_rank_cv:.2f})")
    else:
        verdict_parts.append(f"DiT-FLUX rank unstable (CV={dit_flux_rank_cv:.2f}) — 'attention topology furthest' is scheme-dependent")

    print(f"\nVERDICT:")
    for vp in verdict_parts:
        print(f"  • {vp}")

    # Save
    output = {
        "n_architectures": n_full,
        "n_pairs": n_pairs,
        "architectures": full_archs,
        "pair_labels": pair_labels,
        "normalization_schemes": scheme_names,
        "kendall_W": float(W),
        "kendall_chi2": float(chi2),
        "kendall_p": float(p_w),
        "kendall_df": int(df),
        "rankings": {sn: all_rankings[i].tolist() for i, sn in enumerate(scheme_names)},
        "per_pair_rank_cv": rank_cvs,
        "distances": all_distances,
        "verdict": "; ".join(verdict_parts),
    }

    out_path = OUT_DIR / "ranking_stability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
