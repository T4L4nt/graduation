#!/usr/bin/env python
"""
Phase 7b: Architecture topology → drift fingerprint mapping.

Creates a visualization linking each architecture's structural topology
to its observed drift fingerprint. This formalizes the "architecture signature"
claim by showing WHY each fingerprint looks the way it does.

Pure CPU script. Uses pre-computed drift data.

Usage:
    python scripts/phase7_arch_topo_mapping.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

OUT_DIR = Path("outputs/phase7_editing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Architecture-color mapping
ARCH_COLORS = {
    "SD 1.5": "#2196F3",
    "SDXL": "#FF9800",
    "DiT": "#4CAF50",
    "FLUX": "#E91E63",
}


def load_drift_data():
    """Load drift vectors for all four architectures."""
    data = {}

    # SD 1.5
    with open("outputs/phase1/layer_drift_summary.json") as f:
        d = json.load(f)
    data["SD 1.5"] = {
        "drift": np.array([v["mean"] for v in d["aggregated"].values()]),
        "layers": list(d["aggregated"].keys()),
    }

    # SDXL
    with open("outputs/sdxl_phase1/layer_drift_summary.json") as f:
        d = json.load(f)
    layers = sorted(d["per_image"].keys(), key=lambda x: (x.split(".")[0], len(x), x))
    drift_sdxl = np.array([np.mean(list(d["per_image"][l].values())) for l in layers])
    data["SDXL"] = {"drift": drift_sdxl, "layers": layers}

    # DiT
    with open("outputs/dit_phase1/layer_drift_summary.json") as f:
        d = json.load(f)
    block_names = sorted(d["per_image"].keys(), key=lambda x: int(x.replace("blocks.", "")))
    drift_dit = np.array([np.mean(list(d["per_image"][b].values())) for b in block_names])
    data["DiT"] = {"drift": drift_dit, "layers": block_names}

    # FLUX
    with open("outputs/phase6_flux/diagnosis_summary.json") as f:
        d = json.load(f)
    joint = [f"joint_{i}" for i in range(19)]
    single = [f"single_{i}" for i in range(38)]
    drift_flux = np.array([d["drift"][n]["hidden_drift"] for n in joint + single])
    data["FLUX"] = {"drift": drift_flux, "layers": joint + single}

    return data


def normalize_drift(vals, pct=95):
    vmax = np.percentile(vals, pct)
    if vmax <= 0:
        vmax = vals.max() or 1.0
    return np.clip(vals / vmax, 0, 1.0)


def plot_arch_schematic(ax, arch_name, x, y, w, h):
    """Draw a simplified schematic of each architecture's topology."""
    if arch_name == "SD 1.5":
        # UNet encoder-decoder with skip connections
        # Encoder blocks
        for i in range(4):
            ey = y + h - (i + 1) * h / 5
            rect = FancyBboxPatch((x + w * 0.05, ey), w * 0.35, h / 6,
                                   boxstyle="round,pad=0.02",
                                   facecolor="#BBDEFB", edgecolor="#2196F3", linewidth=1)
            ax.add_patch(rect)
            if i == 3:
                ax.text(x + w * 0.22, ey + h / 12, "Encoder\nResNet", fontsize=6,
                        ha="center", va="center", color="#1565C0", fontweight="bold")
        # Bottleneck
        by = y + h * 0.35
        rect = FancyBboxPatch((x + w * 0.05, by), w * 0.35, h / 8,
                               boxstyle="round,pad=0.02",
                               facecolor="#90CAF9", edgecolor="#1976D2", linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w * 0.22, by + h / 16, "Bottleneck", fontsize=6,
                ha="center", va="center", color="#1565C0")
        # Decoder blocks (where drift concentrates)
        for i in range(4):
            dy = y + (i + 1) * h / 5
            is_hot = (i == 0)  # top decoder = most drift
            face = "#FF5252" if is_hot else "#BBDEFB"
            edge = "#D32F2F" if is_hot else "#2196F3"
            rect = FancyBboxPatch((x + w * 0.55, dy), w * 0.35, h / 6,
                                   boxstyle="round,pad=0.02",
                                   facecolor=face, edgecolor=edge, linewidth=1.5 if is_hot else 1)
            ax.add_patch(rect)
            if i == 3:
                ax.text(x + w * 0.72, dy + h / 12, "Decoder\nResNet ★", fontsize=6,
                        ha="center", va="center", color="#B71C1C" if is_hot else "#1565C0",
                        fontweight="bold")
        # Skip connections
        for i in range(4):
            ey = y + h - (i + 1) * h / 5 + h / 12
            dy = y + (i + 1) * h / 5 + h / 12
            ax.plot([x + w * 0.40, x + w * 0.55], [ey, dy],
                    color="#78909C", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.text(x + w * 0.48, y + h / 2, "skip", fontsize=5, color="#78909C",
                rotation=90, va="center", ha="center")

    elif arch_name == "SDXL":
        # Larger UNet, mid_block bottleneck
        for i in range(3):
            ey = y + h - (i + 1) * h / 5
            rect = FancyBboxPatch((x + w * 0.05, ey), w * 0.35, h / 7,
                                   boxstyle="round,pad=0.02",
                                   facecolor="#FFE0B2", edgecolor="#FF9800", linewidth=1)
            ax.add_patch(rect)
        # Hot mid_block
        rect = FancyBboxPatch((x + w * 0.15, y + h * 0.38), w * 0.65, h / 6,
                               boxstyle="round,pad=0.02",
                               facecolor="#FF5252", edgecolor="#D32F2F", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w * 0.48, y + h * 0.38 + h / 12, "mid_block ★", fontsize=7,
                ha="center", va="center", color="#B71C1C", fontweight="bold")
        for i in range(3):
            dy = y + (i + 1) * h / 5
            rect = FancyBboxPatch((x + w * 0.55, dy), w * 0.35, h / 7,
                                   boxstyle="round,pad=0.02",
                                   facecolor="#FFE0B2", edgecolor="#FF9800", linewidth=1)
            ax.add_patch(rect)
        ax.text(x + w * 0.22, y + h * 0.85, "Encoder", fontsize=6, ha="center", color="#E65100")
        ax.text(x + w * 0.72, y + h * 0.85, "Decoder", fontsize=6, ha="center", color="#E65100")

    elif arch_name == "DiT":
        # 40 transformer blocks with residual streams
        for i in range(40):
            block_y = y + h - (i + 1) * h / 41
            in_transition = (14 <= i <= 27)
            face = "#FF5252" if in_transition else "#C8E6C9"
            edge = "#D32F2F" if in_transition else "#4CAF50"
            rect = FancyBboxPatch((x + w * 0.1, block_y), w * 0.75, h / 45,
                                   boxstyle="round,pad=0.01",
                                   facecolor=face, edgecolor=edge,
                                   linewidth=1 if in_transition else 0.3)
            ax.add_patch(rect)
        # Residual stream arrows
        ax.annotate("", xy=(x + w * 0.9, y + h * 0.05), xytext=(x + w * 0.9, y + h * 0.95),
                    arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.5))
        ax.text(x + w * 0.95, y + h / 2, "residual", fontsize=6, color="#78909C",
                rotation=90, va="center")
        # Transition zone annotation
        ax.annotate("blocks\n11-21 ★", xy=(x + w * 0.45, y + h * 0.48),
                    fontsize=6, color="#B71C1C", fontweight="bold", ha="center")
        ax.text(x + w * 0.45, y + h * 0.95, "40 blocks, AdaLN", fontsize=6,
                ha="center", color="#2E7D32")

    elif arch_name == "FLUX":
        # 19 joint + 38 single blocks with dual-stream indicator
        # Joint blocks
        for i in range(19):
            jy = y + h - (i + 1) * h / 60
            is_last = (i == 18)
            face = "#FF5252" if is_last else "#E1BEE7"
            edge = "#D32F2F" if is_last else "#9C27B0"
            rect = FancyBboxPatch((x + w * 0.05, jy), w * 0.85, h / 65,
                                   boxstyle="round,pad=0.01",
                                   facecolor=face, edgecolor=edge,
                                   linewidth=1.5 if is_last else 0.3)
            ax.add_patch(rect)
        # Divider
        ax.axhline(y=y + h * 0.62, color="white", linewidth=2)
        ax.text(x + w * 0.45, y + h * 0.64, "── joint→single ──", fontsize=5,
                ha="center", color="#666", fontstyle="italic")
        # Single blocks
        for i in range(38):
            sy = y + h * 0.60 - (i + 1) * h / 65
            is_hot = (i <= 6) or (i >= 34)
            face = "#FF5252" if is_hot else "#BBDEFB"
            edge = "#D32F2F" if is_hot else "#2196F3"
            rect = FancyBboxPatch((x + w * 0.05, sy), w * 0.85, h / 70,
                                   boxstyle="round,pad=0.01",
                                   facecolor=face, edgecolor=edge,
                                   linewidth=1 if is_hot else 0.2)
            ax.add_patch(rect)
        # Annotations
        ax.text(x + w * 0.45, y + h * 0.92, "Joint blocks (19)\ntext+image ★joint_18",
                fontsize=5.5, ha="center", color="#4A148C", fontweight="bold")
        ax.text(x + w * 0.45, y + h * 0.25, "Single blocks (38)\nimage only ★early+late",
                fontsize=5.5, ha="center", color="#0D47A1", fontweight="bold")
        # Dual-stream indicator
        ax.text(x + w * 0.45, y + h * 0.97, "MM-DiT dual-stream", fontsize=5,
                ha="center", color="#666")

    ax.set_xlim(x - 0.05, x + w + 0.2)
    ax.set_ylim(y - 0.02, y + h + 0.02)
    ax.axis("off")


def plot_drift_heatmap(ax, arch_name, drift_vals, x, y, w, h):
    """Plot normalized drift heatmap alongside the schematic."""
    drift_norm = normalize_drift(drift_vals, pct=95)
    drift_2d = drift_norm.reshape(-1, 1)

    blues = plt.cm.Blues
    ax.imshow(drift_2d, aspect="auto", cmap=blues, origin="upper",
              extent=[x, x + w, y, y + h])
    ax.set_xlim(x, x + w)
    ax.set_ylim(y, y + h)
    ax.axis("off")


def create_figure(data):
    """Create the main figure: 4 rows × 3 columns."""
    arch_names = ["SD 1.5", "SDXL", "DiT", "FLUX"]
    arch_labels = [
        "SD 1.5 (UNet + DDIM)",
        "SDXL (UNet + DDIM)",
        "DiT (Transformer + v-pred)",
        "FLUX (MM-DiT + Flow Match)",
    ]

    fig = plt.figure(figsize=(22, 14))

    # Column layout: [Schematic | Drift Heatmap | Mechanism Text]
    gs = fig.add_gridspec(4, 3, width_ratios=[1.2, 0.8, 2.5],
                          height_ratios=[1, 1, 1, 1],
                          hspace=0.4, wspace=0.15)

    mechanisms = {
        "SD 1.5": (
            "UNet encoder-decoder\n"
            "with skip connections",
            "Skip connections propagate\n"
            "correction signals forward,\n"
            "accumulating drift in the\n"
            "decoder ResNet blocks\n"
            "(end of information chain).",
            ["Skip propagation: λdℓ", "Decoder = accumulation endpoint",
             "ResNet >> Attention (5×)", "Top: up_blocks.2.resnets.0"],
        ),
        "SDXL": (
            "Larger UNet with\n"
            "redistributed encoder/decoder",
            "Scaling shifts the information\n"
            "bottleneck to mid_block. Larger\n"
            "encoder/decoder distribute\n"
            "processing, concentrating\n"
            "information loss at the funnel.",
            ["Scaling → bottleneck shift", "mid_block = information funnel",
             "Encoder/decoder distribute load", "Top: mid_block.resnets.1"],
        ),
        "DiT": (
            "40-block transformer\n"
            "with AdaLN + residual streams",
            "No UNet-style skip connections.\n"
            "Information flows sequentially\n"
            "through residual streams.\n"
            "Drift peaks at representational\n"
            "transition (blocks 11–21).",
            ["Sequential residual flow", "Transition zone = representation shift",
             "No cross-layer shortcuts", "Top: blocks.20"],
        ),
        "FLUX": (
            "57-block MM-DiT\n"
            "dual-stream transformer",
            "Dual-stream attention (joint blocks)\n"
            "stabilizes features by mixing\n"
            "text+image. Removing this at the\n"
            "joint→single boundary triggers\n"
            "drift spike. Last joint block is\n"
            "the cross-modal handoff point.",
            ["Dual-stream = stabilization", "Architecture boundary = drift source",
             "joint_18 = cross-modal handoff", "Early single = text context loss"],
        ),
    }

    for row, arch_name in enumerate(arch_names):
        drift = data[arch_name]["drift"]

        # Column 1: Architecture schematic
        ax_schem = fig.add_subplot(gs[row, 0])
        plot_arch_schematic(ax_schem, arch_name, 0, 0, 1, 1)
        ax_schem.set_title(arch_labels[row], fontsize=9, fontweight="bold",
                           color=ARCH_COLORS[arch_name], loc="left", pad=2)

        # Column 2: Drift heatmap
        ax_drift = fig.add_subplot(gs[row, 1])
        plot_drift_heatmap(ax_drift, arch_name, drift, 0, 0, 1, 1)
        n_layers = len(drift)
        ax_drift.set_title(f"Drift ({n_layers} layers)", fontsize=8, color="#555",
                           loc="right", pad=2)

        # Column 3: Mechanism explanation
        ax_mech = fig.add_subplot(gs[row, 2])
        ax_mech.axis("off")

        topo, mechanism, bullets = mechanisms[arch_name]

        # Topology description
        ax_mech.text(0.02, 0.92, "TOPOLOGY", fontsize=7, fontweight="bold",
                     color="#333", va="top", fontfamily="monospace")
        ax_mech.text(0.02, 0.82, topo, fontsize=8, va="top", color=ARCH_COLORS[arch_name],
                     fontweight="bold")

        # Why arrow
        ax_mech.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.75),
                         arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
        ax_mech.text(0.52, 0.72, "WHY", fontsize=7, fontweight="bold", color="#555",
                     fontfamily="monospace")

        # Mechanism
        ax_mech.text(0.02, 0.58, "MECHANISM", fontsize=7, fontweight="bold",
                     color="#333", va="top", fontfamily="monospace")
        ax_mech.text(0.02, 0.42, mechanism, fontsize=8, va="top", color="#333")

        # Key evidence bullets
        ax_mech.text(0.02, 0.22, "EVIDENCE", fontsize=7, fontweight="bold",
                     color="#333", va="top", fontfamily="monospace")
        for i, bullet in enumerate(bullets):
            ax_mech.text(0.05, 0.14 - i * 0.07, f"• {bullet}", fontsize=7.5,
                         va="top", color="#555")

    # Title
    fig.suptitle("Architecture Topology → Drift Fingerprint: A Predictive Mapping",
                 fontsize=14, fontweight="bold", y=0.99)

    # Footer: predictive framework
    fig.text(0.5, 0.01,
             "Predictive Framework: Given a diffusion architecture, drift concentration is "
             "determined by (1) information bottleneck topology, (2) skip/residual connection "
             "structure, and (3) cross-modal interaction boundaries.\n"
             "UNet family: drift at information sink (decoder / mid_block). "
             "Transformer family: drift at representational transitions (mid-blocks / block-type boundaries).",
             fontsize=7, ha="center", color="#888", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT_DIR / "arch_topo_fingerprint_mapping.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'arch_topo_fingerprint_mapping.png'}")


def main():
    print("Loading drift data...")
    data = load_drift_data()
    for name, d in data.items():
        print(f"  {name}: {len(d['drift'])} layers")

    print("\nCreating architecture topology → fingerprint mapping figure...")
    create_figure(data)
    print("Done.")


if __name__ == "__main__":
    main()
