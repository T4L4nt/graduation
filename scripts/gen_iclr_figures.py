"""
Generate ICLR 2027 main-paper figures.

5 Figures + 2 Tables for 8-page main text.

Figure 2: Architecture Fingerprint — definition + 5-arch overlay + similarity
Figure 3: Topology → Fingerprint mapping (data evidence)
Figure 5: Diagnosis-guided correction + editing examples

(Figures 1 and 4 are conceptual — drawn in draw.io)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent palette
C_ARCH = {  # per-architecture
    "SD 1.5": "#3498DB", "SDXL": "#E74C3C",
    "HunyuanDiT": "#2ECC71", "FLUX": "#F39C12", "SD 3.5": "#9B59B6",
}
C_DOWN, C_MID, C_UP, C_INT = "#27AE60", "#F39C12", "#2E86C1", "#E74C3C"
C_NOISE = "#E67E22"
C_DARK = "#2C3E50"

# ===========================================================================
# Unified data loader
# ===========================================================================

def load_drift_profile(path, arch_name=""):
    """Load drift profile in architectural (depth) order.

    Handles three data formats:
      Format 1: {"aggregated": {layer: {"mean": ...}}}        — SD 1.5, SD 3.5
      Format 2: {"per_image": {layer: {img: val}}}            — SDXL, DiT
      Format 3: {"drift": {"joint_0": {"hidden_drift": ...}}} — FLUX

    CRITICAL: full_ranking is sorted by drift magnitude (not depth) and is
    NOT suitable as a drift profile. Use per_image/aggregated instead.
    """
    with open(path) as f:
        data = json.load(f)

    # Format 1: aggregated
    if "aggregated" in data:
        agg = data["aggregated"]
        k0 = list(agg.keys())[0]
        if isinstance(agg[k0], dict) and "mean" in agg[k0]:
            layers = sorted(agg.keys(),
                          key=lambda x: (x.split(".")[0], len(x), x))
            vals = np.array([agg[l]["mean"] for l in layers])
            return layers, vals

    # Format 2: per_image (SDXL, DiT) — sort by architectural depth
    if "per_image" in data:
        per_image = data["per_image"]
        # SDXL: layers like "down_blocks.0.resnets.0"
        # DiT: layers like "blocks.0"
        # Sort: DiT blocks by numeric index; others by section then name
        k0 = list(per_image.keys())[0]
        if k0.startswith("blocks."):
            # DiT: sort by block number
            layers = sorted(per_image.keys(),
                          key=lambda x: int(x.replace("blocks.", "")))
        else:
            # SDXL / UNet: sort by section (down→mid→up) then name
            layers = sorted(per_image.keys(),
                          key=lambda x: (x.split(".")[0], len(x), x))
        vals = np.array([np.mean([float(v) for v in per_image[l].values()])
                        for l in layers])
        return layers, vals

    # Format 3: FLUX drift
    if "drift" in data:
        drift_data = data["drift"]
        k0 = list(drift_data.keys())[0]
        if isinstance(drift_data[k0], dict) and "hidden_drift" in drift_data[k0]:
            joint_names = [f"joint_{i}" for i in range(19)]
            single_names = [f"single_{i}" for i in range(38)]
            ordered = joint_names + single_names
            vals = np.array([drift_data[n]["hidden_drift"] for n in ordered])
            return ordered, vals

    raise ValueError(f"Cannot parse {arch_name}: keys={list(data.keys())}")


def load_all_architectures():
    paths = {
        "SD 1.5": "outputs/phase1/layer_drift_summary.json",
        "SDXL": "outputs/sdxl_phase1/layer_drift_summary.json",
        "HunyuanDiT": "outputs/dit_phase1/layer_drift_summary.json",
        "FLUX": "outputs/phase6_flux/diagnosis_summary.json",
        "SD 3.5": "outputs/sd35_phase1/layer_drift_summary.json",
    }
    profiles = {}
    for name, p in paths.items():
        try:
            _, vals = load_drift_profile(p, name)
            profiles[name] = vals
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
    return profiles


# ===========================================================================
# Figure 2: Architecture Fingerprint
# ===========================================================================

def fig2_fingerprint():
    """Figure 2: (a) 5-arch overlay (b) structural distance matrix (no interpolation).

    Panel B uses architecture-level structural features (peak position, modality,
    drift concentration, spread) computed directly from raw layer counts — no
    interpolation to a common length. Avoids the artifact where interpolating
    28-point SDXL to 57 points (51% synthetic) inflates similarity with DiT.
    """
    profiles = load_all_architectures()
    if len(profiles) < 3:
        print("[Fig 2] Not enough architectures loaded, skipping")
        return

    # Interpolate to unified length (Panel A: qualitative visual only)
    unified_len = 57
    unified = {}
    for name, vals in profiles.items():
        v = vals / (vals.max() + 1e-8)
        if len(v) != unified_len:
            x_old = np.linspace(0, 1, len(v))
            x_new = np.linspace(0, 1, unified_len)
            v = np.interp(x_new, x_old, v)
        unified[name] = v

    # Structural features (NO interpolation — raw layer counts)
    def extract_features(vals):
        """Extract 4D structural feature vector from raw drift values."""
        vn = vals / (vals.max() + 1e-8)
        n = len(vn)
        peak_idx = np.argmax(vn)
        peak_pos = peak_idx / max(1, n - 1)

        # Count significant local maxima (>0.3 normalized)
        peaks = sum(1 for i in range(1, n - 1)
                    if vn[i] > vn[i - 1] and vn[i] > vn[i + 1] and vn[i] > 0.3)
        n_peaks = min(peaks, 9)  # cap for display

        # Drift concentration: fraction of layers above 0.5
        concentration = (vn > 0.5).sum() / n

        # Spread: normalized span of above-noise (>0.1) layers
        above = np.where(vn > 0.1)[0]
        spread = (above[-1] - above[0]) / n if len(above) > 0 else 0.0

        return np.array([peak_pos, n_peaks / 10.0, concentration, spread])

    names_list = list(profiles.keys())
    n = len(names_list)
    feat_vecs = {name: extract_features(profiles[name]) for name in names_list}

    # Structural distance matrix (Euclidean, 0 = identical)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(
                feat_vecs[names_list[i]] - feat_vecs[names_list[j]])

    # ---- Figure layout ----
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    # --- Panel A: 5-architecture overlay (qualitative, interpolated) ---
    ax = fig.add_subplot(gs[0])
    for name, vals in unified.items():
        x = np.arange(len(vals))
        ax.plot(x, vals, color=C_ARCH[name], linewidth=2.2, alpha=0.9, label=name)

    for name, vals in unified.items():
        pi = np.argmax(vals)
        ax.annotate(name, (pi, vals[pi]), color=C_ARCH[name], fontsize=8,
                   fontweight="bold", xytext=(0, 6), textcoords="offset points",
                   ha="center")

    ax.set_xlabel("Layer Index (interpolated, qualitative)", fontsize=11, color=C_DARK)
    ax.set_ylabel("Normalized Drift Phi(M)", fontsize=11, color=C_DARK)
    ax.set_title("Architecture Fingerprints: 4 Backbones, 2 Paradigms",
                fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(alpha=0.25)
    ax.text(0.02, -0.12,
            "(qualitative overlay -- interpolation for visual alignment only)",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")

    # --- Panel B: Structural distance matrix ---
    ax2 = fig.add_subplot(gs[1])
    # Use Greens colormap (dark = close, light = far)
    im = ax2.imshow(dist, cmap="YlOrRd", vmin=0, vmax=max(1.0, dist.max()),
                    aspect="equal")

    short_names = [n.replace("HunyuanDiT", "H-DiT") for n in names_list]
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_title("Structural Distance (Euclidean)\nlower = more similar",
                 fontsize=12, fontweight="bold", color=C_DARK)

    for i in range(n):
        for j in range(n):
            val = dist[i, j]
            # Black text on light cells, white on dark
            c = "white" if val > 0.4 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # Caption with actual closest/farthest pairs
    # Find closest and farthest off-diagonal pairs
    triu_idx = np.triu_indices(n, k=1)
    closest = np.argmin(dist[triu_idx])
    farthest = np.argmax(dist[triu_idx])
    i_c, j_c = triu_idx[0][closest], triu_idx[1][closest]
    i_f, j_f = triu_idx[0][farthest], triu_idx[1][farthest]
    fig.text(0.5, -0.03,
             f"Panel B: Structural distance from 4 raw-layer-count features "
             f"(peak position, modality, concentration, spread) -- NO interpolation. "
             f"Closest: {short_names[i_c]}--{short_names[j_c]} d={dist[i_c,j_c]:.3f}; "
             f"Farthest: {short_names[i_f]}--{short_names[j_f]} d={dist[i_f,j_f]:.3f}.",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.suptitle("Figure 2: Architecture Fingerprint -- Phi(M) is reproducible, architecture-specific",
                fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    fig.savefig(OUT_DIR / "fig2_fingerprint.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_fingerprint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 2] -> outputs/figures/fig2_fingerprint.pdf")


# ===========================================================================
# Figure 3: Topology → Fingerprint mapping evidence
# ===========================================================================

def fig3_topology_mapping():
    """Figure 3: Architecture topology determines drift location.
    Shows per-architecture drift with architecture graph annotation.
    """
    profiles = load_all_architectures()
    if len(profiles) < 3:
        print("[Fig 3] Not enough architectures, skipping")
        return

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

    arch_info = [
        ("SD 1.5", "UNet\nPeak: decoder"),
        ("SDXL", "UNet (large)\nPeak: mid_block"),
        ("HunyuanDiT", "Transformer\nPeak: transition"),
        ("FLUX", "MM-DiT\nPeak: joint→single"),
        ("SD 3.5", "MM-DiT-X\nPeak: output compression"),
    ]

    for ax, (name, info) in zip(axes, arch_info):
        if name in profiles:
            vals = profiles[name]
            vals_n = vals / (vals.max() + 1e-8)
            x = np.arange(len(vals_n))
            ax.fill_between(x, 0, vals_n, color=C_ARCH[name], alpha=0.3)
            ax.plot(x, vals_n, color=C_ARCH[name], linewidth=1.8)

            # Mark peak
            pi = np.argmax(vals_n)
            ax.axvline(x=pi, color="#E74C3C", linewidth=1, linestyle="--", alpha=0.7)

        ax.set_title(f"{name}\n{info}", fontsize=9, fontweight="bold", color=C_DARK)
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("Φ" if ax == axes[0] else "", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 3: Topology → Fingerprint — Drift Peaks at Architecture-specific Bottlenecks",
                fontsize=13, fontweight="bold", color=C_DARK)
    fig.savefig(OUT_DIR / "fig3_topology.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_topology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 3] → outputs/figures/fig3_topology.pdf")


# ===========================================================================
# Figure 5: Diagnosis-guided correction + editing
# ===========================================================================

def fig5_application():
    """Figure 5: Correction mechanism + editing results."""
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

    # --- Panel A: Why simple correction works ---
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5, 9.5, "Diagnosis → Intervention", fontsize=11, fontweight="bold",
           color=C_DARK, ha="center")
    ax.text(5, 8.5, "Φ(M) → Peak layer → One λ", fontsize=10, color=C_DARK, ha="center")

    ax.text(5, 7, "random5 ≈ top5", fontsize=10, fontweight="bold", color=C_UP, ha="center")
    ax.text(5, 6.3, "ΔPSNR < 0.3 dB", fontsize=9, color=C_DARK, ha="center")

    ax.text(5, 5.3, "λ ∈ {0.3, 0.5, 0.7}", fontsize=10, fontweight="bold", color=C_UP, ha="center")
    ax.text(5, 4.6, "PSNR range < 0.08 dB", fontsize=9, color=C_DARK, ha="center")

    ax.text(5, 3.3, "P2P vs Ours", fontsize=10, fontweight="bold", color="#2ECC71", ha="center")
    ax.text(5, 2.6, "Cohen's d = 0.033", fontsize=9, color=C_DARK, ha="center")
    ax.text(5, 2.0, "MB vs GB memory", fontsize=9, color=C_DARK, ha="center")

    ax.text(5, 0.8, "Diagnosis makes simple", fontsize=11, fontweight="bold",
           color="#F1C40F", ha="center")
    ax.text(5, 0.2, "correction sufficient", fontsize=11, fontweight="bold",
           color="#F1C40F", ha="center")

    # --- Panel B: Cross-architecture ---
    ax = fig.add_subplot(gs[1])
    archs = ["SD 1.5", "SDXL", "HunyuanDiT", "FLUX"]
    delta_psnr = [2.75, 5.23, 5.65, 3.94]
    colors_bar = [C_ARCH[a] for a in archs]
    ax.barh(range(len(archs)), delta_psnr, color=colors_bar, height=0.5,
           edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs, fontsize=10)
    ax.set_xlabel("ΔPSNR (dB)", fontsize=10, color=C_DARK)
    ax.set_title("Correction across Architectures", fontsize=11, fontweight="bold", color=C_DARK)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(delta_psnr):
        ax.text(v + 0.1, i, f"+{v:.1f}", fontsize=9, va="center", fontweight="bold")

    # --- Panel C: Editing results ---
    ax = fig.add_subplot(gs[2])
    try:
        with open("outputs/phase8_iclr_editing/summary.json") as f:
            e_data = json.load(f)
        pc = e_data["per_condition"]
        conditions = ["original", "cut_a", "noise_a"]
        labels = ["Original", "Cut A", "Noise A"]
        metrics_plot = ["lpips_consistency", "ssim_preservation", "clip_dir_sim"]
        metric_labels = ["LPIPS↓", "SSIM↑", "CLIP-Dir↑"]
        x = np.arange(len(metrics_plot))
        width = 0.25
        for i, (cond, label) in enumerate(zip(conditions, labels)):
            vals = [pc[cond][m]["mean"] for m in metrics_plot]
            colors_cond = [C_UP, C_INT, C_NOISE]
            ax.bar(x + i * width, vals, width, color=colors_cond[i],
                  label=label, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_title("Editing: Structure ↑, Direction → 0", fontsize=11, fontweight="bold", color=C_DARK)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"[Editing data unavailable]\n{e}", ha="center", va="center",
               transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")

    fig.suptitle("Figure 5: Diagnosis-guided Correction — Simple because Diagnosis is Sufficient",
                fontsize=14, fontweight="bold", color=C_DARK)
    fig.savefig(OUT_DIR / "fig5_application.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_application.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 5] → outputs/figures/fig5_application.pdf")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating ICLR main-paper figures (2, 3, 5)...")
    fig2_fingerprint()
    fig3_topology_mapping()
    fig5_application()
    print(f"\nDone. Output: {OUT_DIR.resolve()}\n")
    print("Figures 1 and 4 (conceptual) → draw.io")
