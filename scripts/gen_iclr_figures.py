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
    with open(path) as f:
        data = json.load(f)

    # Format 1: Phase 1 — {"aggregated": {layer: {"mean": ...}}}
    if "aggregated" in data:
        agg = data["aggregated"]
        k0 = list(agg.keys())[0]
        if isinstance(agg[k0], dict) and "mean" in agg[k0]:
            names = list(agg.keys())
            vals = np.array([agg[n]["mean"] for n in names])
            return names, vals

    # Format 2: SDXL / DiT — {"full_ranking": [{"layer": ..., "mean_drift": ...}]}
    if "full_ranking" in data:
        ranking = data["full_ranking"]
        r0 = ranking[0]
        mk = "mean" if "mean" in r0 else "mean_drift"
        names = [r["layer"] for r in ranking]
        vals = np.array([r[mk] for r in ranking])
        return names, vals

    # Format 3: FLUX — {"drift": {"joint_0": {"hidden_drift": ...}, ...}}
    if "drift" in data:
        drift_data = data["drift"]
        k0 = list(drift_data.keys())[0]
        if isinstance(drift_data[k0], dict) and "hidden_drift" in drift_data[k0]:
            names = list(drift_data.keys())
            vals = np.array([drift_data[n]["hidden_drift"] for n in names])
            return names, vals

    raise ValueError(f"Cannot parse {arch_name}")


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
    """Figure 2: (a) Definition sketch (b) 5-arch overlay (c) similarity matrix."""
    profiles = load_all_architectures()
    if len(profiles) < 3:
        print("[Fig 2] Not enough architectures loaded, skipping")
        return

    # Interpolate to unified length
    unified_len = 57
    unified = {}
    for name, vals in profiles.items():
        v = vals / (vals.max() + 1e-8)
        if len(v) != unified_len:
            x_old = np.linspace(0, 1, len(v))
            x_new = np.linspace(0, 1, unified_len)
            v = np.interp(x_new, x_old, v)
        unified[name] = v

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3)

    # --- Panel A: 5-architecture overlay ---
    ax = fig.add_subplot(gs[0])
    for name, vals in unified.items():
        x = np.arange(len(vals))
        ax.plot(x, vals, color=C_ARCH[name], linewidth=2.2, alpha=0.9, label=name)

    # Mark peaks
    for name, vals in unified.items():
        pi = np.argmax(vals)
        ax.annotate(name, (pi, vals[pi]), color=C_ARCH[name], fontsize=8,
                   fontweight="bold", xytext=(0, 6), textcoords="offset points", ha="center")

    ax.set_xlabel("Layer Index (interpolated)", fontsize=11, color=C_DARK)
    ax.set_ylabel("Normalized Drift Φ(M)", fontsize=11, color=C_DARK)
    ax.set_title("Architecture Fingerprints: 5 Backbones, 2 Paradigms",
                fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(alpha=0.25)

    # --- Panel B: Similarity matrix ---
    ax2 = fig.add_subplot(gs[1])
    names_list = list(unified.keys())
    n = len(names_list)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = np.corrcoef(unified[names_list[i]], unified[names_list[j]])[0, 1]

    im = ax2.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(names_list, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(names_list, fontsize=9)
    ax2.set_title("Pairwise Pearson r", fontsize=12, fontweight="bold", color=C_DARK)

    for i in range(n):
        for j in range(n):
            c = "white" if abs(sim[i, j]) < 0.55 else "black"
            ax2.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    fig.suptitle("Figure 2: Architecture Fingerprint — Φ(M) is reproducible, architecture-specific",
                fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    fig.savefig(OUT_DIR / "fig2_fingerprint.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_fingerprint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 2] → outputs/figures/fig2_fingerprint.pdf")


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
    delta_psnr = [2.75, 5.37, 5.65, 3.94]
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
