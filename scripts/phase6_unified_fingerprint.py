#!/usr/bin/env python
"""
Phase 6: Four-architecture unified drift fingerprint comparison.

Loads drift data from SD 1.5, SDXL, DiT, and FLUX, then generates:
  1. Four-panel drift heatmap with per-architecture normalization
  2. Architecture similarity matrix (Pearson r, cosine, Spearman rho)
  3. Overlay plot of normalized drift profiles

Pure CPU script — uses pre-computed JSON data.

Usage:
    python scripts/phase6_unified_fingerprint.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.interpolate import interp1d

OUT_DIR = Path("outputs/phase6_unified")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sd15_drift():
    """Load SD 1.5 38-layer aggregated drift."""
    path = Path("outputs/phase1/layer_drift_summary.json")
    with open(path) as f:
        data = json.load(f)

    aggregated = data["aggregated"]
    layers = sorted(aggregated.keys(),
                    key=lambda x: (x.split(".")[0], len(x), x))
    drift_vals = np.array([aggregated[l]["mean"] for l in layers])

    # Section markers
    sections = {
        "Encoder": (0, 12),
        "Bottleneck": (12, 14),
        "Decoder": (14, 38),
    }
    labels = [l.replace("down_blocks", "down").replace("up_blocks", "up")
              .replace("mid_block", "mid").replace("attentions", "attn")
              .replace("transformer_blocks", "tr")
              .replace("resnets", "res")
              for l in layers]

    return drift_vals, labels, sections, "SD 1.5 (UNet, DDIM)"


def load_sdxl_drift():
    """Load SDXL 28-layer drift (mean across images)."""
    path = Path("outputs/sdxl_phase1/layer_drift_summary.json")
    with open(path) as f:
        data = json.load(f)

    per_image = data["per_image"]
    layers = sorted(per_image.keys(),
                    key=lambda x: (x.split(".")[0], len(x), x))
    drift_vals = []
    for l in layers:
        vals = [float(v) for v in per_image[l].values()]
        drift_vals.append(np.mean(vals))
    drift_vals = np.array(drift_vals)

    sections = {
        "Encoder": (0, 9),
        "Bottleneck": (9, 11),
        "Decoder": (11, 28),
    }
    labels = [l.replace("down_blocks", "down").replace("up_blocks", "up")
              .replace("mid_block", "mid").replace("attentions", "attn")
              .replace("transformer_blocks", "tr")
              .replace("resnets", "res")
              for l in layers]

    return drift_vals, labels, sections, "SDXL (UNet, DDIM)"


def load_dit_drift():
    """Load DiT 40-block drift (mean across images)."""
    path = Path("outputs/dit_phase1/layer_drift_summary.json")
    with open(path) as f:
        data = json.load(f)

    per_image = data["per_image"]
    block_names = sorted(per_image.keys(),
                         key=lambda x: int(x.replace("blocks.", "")))
    drift_vals = []
    for b in block_names:
        vals = [float(v) for v in per_image[b].values()]
        drift_vals.append(np.mean(vals))
    drift_vals = np.array(drift_vals)

    # Section: bottom (0-13), transition (14-27), top (28-39) per Phase 1
    sections = {
        "Bottom (0-13)": (0, 14),
        "Transition (14-27)": (14, 28),
        "Top (28-39)": (28, 40),
    }
    labels = [f"b{i}" for i in range(40)]

    return drift_vals, labels, sections, "DiT (Transformer, v-pred)"


def load_flux_drift():
    """Load FLUX 57-block drift from diagnosis output."""
    path = Path("outputs/phase6_flux/diagnosis_summary.json")
    with open(path) as f:
        data = json.load(f)

    drift_data = data["drift"]

    # Build ordered list: joint_0..joint_18 then single_0..single_37
    joint_names = [f"joint_{i}" for i in range(19)]
    single_names = [f"single_{i}" for i in range(38)]
    ordered = joint_names + single_names

    drift_vals = np.array([drift_data[n]["hidden_drift"] for n in ordered])
    sections = {
        "Joint (text+image)": (0, 19),
        "Single (image only)": (19, 57),
    }
    labels = [f"j{i}" for i in range(19)] + [f"s{i}" for i in range(38)]

    return drift_vals, labels, sections, "FLUX (Transformer, Flow Match)"


# ---------------------------------------------------------------------------
# Normalization and interpolation
# ---------------------------------------------------------------------------

def normalize_drift(vals, pct=95):
    """Normalize drift to [0, 1] using percentile cap."""
    vmax = np.percentile(vals, pct)
    if vmax <= 0:
        vmax = vals.max() or 1.0
    return np.clip(vals / vmax, 0, 1.0)


def interpolate_to_length(vals, target_length=57):
    """Interpolate drift vector to target length."""
    x_orig = np.linspace(0, 1, len(vals))
    x_new = np.linspace(0, 1, target_length)
    return interp1d(x_orig, vals, kind="linear")(x_new)


# ---------------------------------------------------------------------------
# Figure 1: Four-panel drift heatmap
# ---------------------------------------------------------------------------

def plot_four_panel_heatmap(all_data):
    """1x4 panel vertical heatmap comparing drift fingerprints."""
    n_archs = len(all_data)
    fig, axes = plt.subplots(1, n_archs, figsize=(4 * n_archs, 14),
                              gridspec_kw={"width_ratios": [1] * n_archs})

    blues = LinearSegmentedColormap.from_list("drift_blues",
        [(0.0, "#f7fbff"), (0.3, "#6baed6"), (0.7, "#2171b5"), (1.0, "#08306b")])

    for ax_idx, (drift_raw, labels, sections, title) in enumerate(all_data):
        ax = axes[ax_idx]
        drift_norm = normalize_drift(drift_raw, pct=95)

        # Vertical heatmap: reshape to (N, 1)
        drift_2d = drift_norm.reshape(-1, 1)
        im = ax.imshow(drift_2d, aspect="auto", cmap=blues, origin="upper")

        # Section labels on left
        for sec_name, (start, end) in sections.items():
            mid = (start + end) / 2
            ax.text(-0.15, mid, sec_name, transform=ax.get_yaxis_transform(),
                    fontsize=8, ha="right", va="center", rotation=90,
                    fontweight="bold")

        # Section divider lines
        for sec_name, (start, end) in sections.items():
            if start > 0:
                ax.axhline(y=start - 0.5, color="white", linewidth=1.5, alpha=0.7)
            if end < len(labels):
                ax.axhline(y=end - 0.5, color="white", linewidth=1.5, alpha=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks(range(0, len(labels), max(1, len(labels) // 10)))
        ax.set_yticklabels([labels[i] for i in
                            range(0, len(labels), max(1, len(labels) // 10))],
                           fontsize=6)
        ax.invert_yaxis()

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized drift (per-architecture 95th pct)", fontsize=10)

    fig.suptitle("Cross-Architecture Feature Drift Fingerprints",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    fig.savefig(OUT_DIR / "four_arch_fingerprint.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'four_arch_fingerprint.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Architecture similarity matrix
# ---------------------------------------------------------------------------

def compute_arch_similarity(all_data):
    """Pairwise similarity between architecture drift profiles."""
    arch_names = [d[3] for d in all_data]
    n = len(arch_names)

    # Interpolate all to common length
    max_len = max(len(d[0]) for d in all_data)
    interp_drifts = {}
    for drift_raw, _, _, name in all_data:
        interp_drifts[name] = interpolate_to_length(
            normalize_drift(drift_raw, pct=95), max_len
        )

    pearson = np.zeros((n, n))
    cosine = np.zeros((n, n))
    spearman = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = interp_drifts[arch_names[i]]
            b = interp_drifts[arch_names[j]]
            pearson[i, j] = np.corrcoef(a, b)[0, 1]
            cosine[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            spearman[i, j] = stats.spearmanr(a, b)[0]

    return arch_names, {"Pearson r": pearson, "Cosine": cosine,
                        "Spearman ρ": spearman}


def plot_similarity_matrix(arch_names, similarity):
    """3-panel similarity matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric_name, matrix) in zip(axes, similarity.items()):
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")

        # Annotate cells
        for i in range(len(arch_names)):
            for j in range(len(arch_names)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(range(len(arch_names)))
        ax.set_yticks(range(len(arch_names)))
        ax.set_xticklabels(arch_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(arch_names, fontsize=8)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Architecture Drift Profile Similarity", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "arch_similarity_matrix.png", dpi=200)
    plt.close()
    print(f"Saved: {OUT_DIR / 'arch_similarity_matrix.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Drift profile overlay
# ---------------------------------------------------------------------------

def plot_drift_profile_overlay(all_data):
    """Overlay normalized drift profiles for all architectures."""
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    styles = ["-", "--", "-.", ":"]

    for idx, (drift_raw, labels, sections, name) in enumerate(all_data):
        drift_norm = normalize_drift(drift_raw, pct=95)
        x = np.linspace(0, 1, len(drift_norm))
        ax.plot(x, drift_norm, color=colors[idx % len(colors)],
                linestyle=styles[idx % len(styles)], linewidth=1.8,
                label=name, alpha=0.85)

    ax.set_xlabel("Normalized depth (0 = input, 1 = output)", fontsize=12)
    ax.set_ylabel("Normalized drift (per-architecture 95th pct)", fontsize=12)
    ax.set_title("Drift Profile Comparison Across Architectures", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "drift_profile_overlay.png", dpi=200)
    plt.close()
    print(f"Saved: {OUT_DIR / 'drift_profile_overlay.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading architecture drift data...")
    all_data = [
        load_sd15_drift(),
        load_sdxl_drift(),
        load_dit_drift(),
        load_flux_drift(),
    ]

    for drift_vals, labels, sections, name in all_data:
        print(f"  {name}: {len(drift_vals)} layers, "
              f"drift range [{drift_vals.min():.4f}, {drift_vals.max():.4f}]")

    print("\nGenerating four-panel heatmap...")
    plot_four_panel_heatmap(all_data)

    print("\nComputing architecture similarity...")
    arch_names, similarity = compute_arch_similarity(all_data)
    plot_similarity_matrix(arch_names, similarity)

    # Print key comparisons
    print("\nKey pairwise similarities (Spearman ρ primary, Pearson r reference):")
    for i in range(len(arch_names)):
        for j in range(i + 1, len(arch_names)):
            r = similarity["Pearson r"][i, j]
            s = similarity["Spearman ρ"][i, j]
            print(f"  {arch_names[i]:30s} vs {arch_names[j]:30s}  "
                  f"ρ={s:.3f}, r={r:.3f}")

    # Highlight architecture determinism finding
    dit_idx = [i for i, n in enumerate(arch_names) if "DiT" in n]
    flux_idx = [i for i, n in enumerate(arch_names) if "FLUX" in n]
    if dit_idx and flux_idx:
        r_transformer = similarity["Pearson r"][dit_idx[0], flux_idx[0]]
        print(f"\n  >>> FLUX vs DiT (both Transformer, diff paradigm): r = {r_transformer:.3f}")
        if r_transformer > 0.5:
            print("  => Backbone dominates drift pattern (Transformer > paradigm)")
        else:
            print("  => Architecture details dominate (joint/single split != pure DiT)")

    print("\nGenerating drift profile overlay...")
    plot_drift_profile_overlay(all_data)

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
