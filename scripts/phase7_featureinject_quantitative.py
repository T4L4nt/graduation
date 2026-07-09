#!/usr/bin/env python
"""
Phase 7d: Quantitative comparison with FeatureInject ("One Size Does Not Fit All").

Computes:
  1. Jaccard overlap: |drift top-K ∩ formation zone| / |top-K ∪ FZ|
  2. Out-of-zone ratio: fraction of top-K drift layers outside FZ
  3. Drift peak position relative to formation zone boundaries
  4. Point-biserial correlation: drift magnitude vs binary (in/out FZ)
  5. LaTeX table for the paper

Pure CPU. Uses pre-computed drift data.

Usage:
    python scripts/phase7_featureinject_quantitative.py
"""

import json
from pathlib import Path

import numpy as np
from phase7_arch_topo_mapping import load_drift_data, normalize_drift

OUT_DIR = Path("outputs/phase7_editing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Formation zones from FeatureInject paper, per architecture.
# Each tuple = (normalized_start, normalized_end) in [0, 1].
FORMATION = {
    "SD 1.5": [(0.35, 0.55)],
    "SDXL":    [(0.40, 0.62)],
    "DiT":     [(0.55, 0.78)],
    "FLUX":    [(0.93, 1.0)],
}

INFERRED = {"DiT"}  # HunyuanDiT zone inferred from SD3.5 analog


def compute_jaccard(topk_indices, formation_mask):
    """Jaccard index between drift top-K set and formation-zone set."""
    topk_set = set(topk_indices)
    fz_set = set(np.where(formation_mask)[0])
    intersection = topk_set & fz_set
    union = topk_set | fz_set
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


def compute_out_of_zone_ratio(topk_indices, formation_mask):
    """Fraction of top-K drift layers that fall OUTSIDE the formation zone."""
    if len(topk_indices) == 0:
        return 0.0
    outside = sum(1 for i in topk_indices if not formation_mask[i])
    return outside / len(topk_indices)


def point_biserial_r(values, binary_mask):
    """Point-biserial correlation between continuous values and binary mask."""
    g1 = values[binary_mask]
    g0 = values[~binary_mask]
    if len(g1) < 2 or len(g0) < 2:
        return float("nan"), float("nan")
    n1, n0 = len(g1), len(g0)
    n = n1 + n0
    m1, m0 = g1.mean(), g0.mean()
    s = values.std(ddof=0)
    if s == 0:
        return 0.0, 1.0
    r_pb = ((m1 - m0) / s) * np.sqrt(n1 * n0 / (n * (n - 1)))
    # t-statistic for testing r_pb == 0
    if abs(r_pb) >= 1.0:
        return r_pb, 0.0
    t = r_pb * np.sqrt((n - 2) / (1 - r_pb ** 2))
    from scipy.stats import t as tdist
    df = n - 2
    p = 2 * tdist.sf(abs(t), df)
    return r_pb, p


def get_formation_mask(n_layers, zones):
    """Create boolean mask: True if layer depth falls in any formation zone."""
    depths = np.linspace(0, 1, n_layers)
    mask = np.zeros(n_layers, dtype=bool)
    for zs, ze in zones:
        mask |= (depths >= zs) & (depths <= ze)
    return mask


def main():
    print("Loading drift data...")
    data = load_drift_data()
    arch_names = ["SD 1.5", "SDXL", "DiT", "FLUX"]

    results = {}

    for arch in arch_names:
        drift = data[arch]["drift"]
        layers = data[arch]["layers"]
        n = len(drift)
        drift_norm = normalize_drift(drift, pct=98)
        depths = np.linspace(0, 1, n)

        zones = FORMATION[arch]
        fz_mask = get_formation_mask(n, zones)

        # Top-K indices sorted by drift magnitude (descending)
        ranked = np.argsort(-drift)
        top5 = ranked[:5]
        top10 = ranked[:10]
        top_k_formula = max(3, min(10, int(n * 0.2)))  # ~20% of layers
        top20pct = ranked[:top_k_formula]

        # Jaccard overlap
        j5 = compute_jaccard(top5, fz_mask)
        j10 = compute_jaccard(top10, fz_mask)
        j20p = compute_jaccard(top20pct, fz_mask)

        # Out-of-zone ratio
        o5 = compute_out_of_zone_ratio(top5, fz_mask)
        o10 = compute_out_of_zone_ratio(top10, fz_mask)
        o20p = compute_out_of_zone_ratio(top20pct, fz_mask)

        # Drift peak position
        peak_idx = int(np.argmax(drift))
        peak_depth = depths[peak_idx]
        peak_layer = layers[peak_idx]

        # Peak in/out of formation zone
        peak_in_fz = fz_mask[peak_idx]

        # Point-biserial: drift magnitude vs formation zone membership
        r_pb, p_pb = point_biserial_r(drift, fz_mask)

        # Fraction of layers in formation zone
        fz_fraction = fz_mask.sum() / n

        # FZ mean drift vs non-FZ mean drift
        fz_mean_drift = drift[fz_mask].mean()
        nonfz_mean_drift = drift[~fz_mask].mean()
        drift_ratio = fz_mean_drift / max(nonfz_mean_drift, 1e-10)

        results[arch] = {
            "n_layers": n,
            "fz_fraction": fz_fraction,
            "jaccard_k5": j5,
            "jaccard_k10": j10,
            "jaccard_20pct": j20p,
            "out_of_zone_k5": o5,
            "out_of_zone_k10": o10,
            "out_of_zone_20pct": o20p,
            "peak_depth": peak_depth,
            "peak_layer": peak_layer,
            "peak_in_fz": peak_in_fz,
            "r_pb": r_pb,
            "p_pb": p_pb,
            "fz_mean_drift": fz_mean_drift,
            "nonfz_mean_drift": nonfz_mean_drift,
            "drift_ratio": drift_ratio,
            "inferred": arch in INFERRED,
            "formation_zones": zones,
        }

    # ---- Print results ----
    print("\n" + "=" * 90)
    print("Quantitative FeatureInject vs. Ours Comparison")
    print("=" * 90)

    for arch in arch_names:
        r = results[arch]
        inferred = " (inferred from SD3.5)" if r["inferred"] else ""
        print(f"\n{'─' * 70}")
        print(f"  {arch}{inferred}")
        print(f"{'─' * 70}")
        print(f"  Layers: {r['n_layers']}  |  Formation zone fraction: {r['fz_fraction']:.2f}")
        print(f"  Peak drift: layer {r['peak_layer']} at depth {r['peak_depth']:.3f}")
        print(f"  Peak in formation zone: {r['peak_in_fz']}")
        print(f"  Jaccard (k=5):  {r['jaccard_k5']:.3f}")
        print(f"  Jaccard (k=10): {r['jaccard_k10']:.3f}")
        print(f"  Jaccard (20%):  {r['jaccard_20pct']:.3f}")
        print(f"  Out-of-zone ratio (k=5):  {r['out_of_zone_k5']:.3f}")
        print(f"  Out-of-zone ratio (k=10): {r['out_of_zone_k10']:.3f}")
        print(f"  Out-of-zone ratio (20%):  {r['out_of_zone_20pct']:.3f}")
        print(f"  Point-biserial r: {r['r_pb']:.4f}  (p={r['p_pb']:.4f})")
        print(f"  Mean drift: FZ={r['fz_mean_drift']:.6f}  non-FZ={r['nonfz_mean_drift']:.6f}  "
              f"ratio={r['drift_ratio']:.2f}×")

    # ---- LaTeX table ----
    latex_lines = []
    latex_lines.append(r"% Auto-generated by scripts/phase7_featureinject_quantitative.py")
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"  \centering")
    latex_lines.append(r"  \caption{Quantitative comparison: drift concentration vs.")
    latex_lines.append(r"    FeatureInject semantic formation zones.")
    latex_lines.append(r"    Low Jaccard and high out-of-zone ratios indicate")
    latex_lines.append(r"    that inversion-reconstruction drift is not a by-product")
    latex_lines.append(r"    of forward semantic formation.}")
    latex_lines.append(r"  \label{tab:featureinject_quant}")
    latex_lines.append(r"  \small")
    latex_lines.append(r"  \begin{tabular}{lccccccc}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Architecture & Layers & FZ frac. & Peak depth & Peak in FZ? "
                       r"& Jaccard$_{k=5}$ & Jaccard$_{k=10}$ & Out-of-zone$_{k=5}$ \\")
    latex_lines.append(r"    \midrule")
    for arch in arch_names:
        r = results[arch]
        inferred_mark = r"$^\dagger$" if r["inferred"] else ""
        peak_in = "Yes" if r["peak_in_fz"] else r"\textbf{No}"
        arch_display = arch.replace("DiT", "HunyuanDiT")
        latex_lines.append(
            f"    {arch_display}{inferred_mark} & {r['n_layers']} & {r['fz_fraction']:.2f} & "
            f"{r['peak_depth']:.3f} & {peak_in} & "
            f"{r['jaccard_k5']:.3f} & {r['jaccard_k10']:.3f} & {r['out_of_zone_k5']:.2f} \\\\"
        )
    latex_lines.append(r"    \midrule")
    latex_lines.append(r"    \multicolumn{8}{l}{$^\dagger$ Formation zone inferred from SD3.5 "
                       r"(FeatureInject did not study HunyuanDiT).} \\")
    latex_lines.append(r"    \multicolumn{8}{l}{FZ frac. = fraction of layers inside the "
                       r"FeatureInject formation zone.} \\")
    latex_lines.append(r"    \multicolumn{8}{l}{Jaccard$_{k=K}$ = "
                       r"$|\text{top-}K\text{ drift} \cap \text{FZ}| \,/\, "
                       r"|\text{top-}K \cup \text{FZ}|$.} \\")
    latex_lines.append(r"    \multicolumn{8}{l}{Out-of-zone$_{k=5}$ = fraction of top-5 drift "
                       r"layers outside the formation zone.} \\")
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_path = OUT_DIR / "featureinject_quantitative.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"\n\nLaTeX table saved: {latex_path}")

    # ---- Detailed per-architecture table ----
    latex_lines2 = []
    latex_lines2.append(r"% Detailed version with point-biserial correlation and drift ratios")
    latex_lines2.append(r"\begin{table}[t]")
    latex_lines2.append(r"  \centering")
    latex_lines2.append(r"  \caption{Detailed quantitative comparison including point-biserial "
                        r"correlation between drift magnitude and formation zone membership. "
                        r"A non-significant or negative correlation indicates drift is not "
                        r"concentrated within the formation zone.}")
    latex_lines2.append(r"  \label{tab:featureinject_detailed}")
    latex_lines2.append(r"  \small")
    latex_lines2.append(r"  \begin{tabular}{lccccc}")
    latex_lines2.append(r"    \toprule")
    latex_lines2.append(r"    Architecture & Jaccard$_{20\%}$ & Out-of-zone$_{20\%}$ "
                        r"& $r_{pb}$ & $p$ & Drift ratio \\")
    latex_lines2.append(r"    \midrule")
    for arch in arch_names:
        r = results[arch]
        inferred_mark = r"$^\dagger$" if r["inferred"] else ""
        arch_display = arch.replace("DiT", "HunyuanDiT")
        p_str = f"{r['p_pb']:.4f}" if not np.isnan(r['p_pb']) else "N/A"
        r_str = f"{r['r_pb']:.3f}" if not np.isnan(r['r_pb']) else "N/A"
        latex_lines2.append(
            f"    {arch_display}{inferred_mark} & {r['jaccard_20pct']:.3f} & "
            f"{r['out_of_zone_20pct']:.2f} & {r_str} & {p_str} & "
            f"{r['drift_ratio']:.2f}$\\times$ \\\\"
        )
    latex_lines2.append(r"    \bottomrule")
    latex_lines2.append(r"  \end{tabular}")
    latex_lines2.append(r"\end{table}")

    latex_path2 = OUT_DIR / "featureinject_detailed.tex"
    with open(latex_path2, "w") as f:
        f.write("\n".join(latex_lines2))
    print(f"Detailed LaTeX table saved: {latex_path2}")

    # ---- Summary JSON ----
    json_path = OUT_DIR / "featureinject_quantitative.json"
    # Convert numpy values for JSON serialization
    results_json = {}
    for arch, r in results.items():
        results_json[arch] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                bool(v) if isinstance(v, np.bool_) else v)
            for k, v in r.items()
        }
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ---- Interpretative summary ----
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for arch in arch_names:
        r = results[arch]
        if r["out_of_zone_k5"] >= 0.6:
            status = "STRONG DIVERGENCE"
        elif r["out_of_zone_k5"] >= 0.2:
            status = "PARTIAL DIVERGENCE"
        else:
            status = "OVERLAP"
        print(f"  {arch:8s}  {status:20s}  "
              f"out-of-zone={r['out_of_zone_k5']:.2f}  "
              f"r_pb={r['r_pb']:.3f}  "
              f"drift_ratio={r['drift_ratio']:.2f}×")


if __name__ == "__main__":
    main()
