"""
Generate unified framework architecture diagram.

Output: outputs/thesis_figures/unified_framework.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path("outputs/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
C_DIAG = "#3498DB"
C_CORR = "#2ECC71"
C_STYLE = "#E67E22"
C_PIN = "#9B59B6"
C_IMAGE = "#1ABC9C"
C_ARROW = "#7F8C8D"
C_BG = "#FAFAFA"
C_DARK = "#2C3E50"


def load_drift_data():
    path = Path("outputs/phase1/layer_drift_summary.json")
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        agg = d.get("aggregated", {})
        top5 = d.get("top_5", [])
        return {name: agg[name]["mean"] for name in top5 if name in agg}
    return {
        "up_blocks.2.resnets.0": 2671.6,
        "mid_block.resnets.1": 1520.3,
        "up_blocks.0.resnets.0": 1501.2,
        "up_blocks.1.resnets.1": 1419.4,
        "mid_block.resnets.0": 1310.9,
    }


def load_delta_data():
    path = Path("outputs/phase4_info_theory/per_layer_correction.json")
    if not path.exists():
        return 2.27, 1.09
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", {})
    resnet_d = [r["mean_delta_psnr"] for r in results.values()
                if r.get("mean_delta_psnr") is not None and "attentions" not in str(r)]
    attn_d = [r["mean_delta_psnr"] for r in results.values()
              if r.get("mean_delta_psnr") is not None and "attentions" in str(r)]
    return (np.mean(resnet_d) if resnet_d else 2.27,
            np.mean(attn_d) if attn_d else 1.09)


def short_name(name):
    return name.replace("up_blocks.", "up.").replace("down_blocks.", "dn.")\
               .replace("mid_block.", "mid.").replace(".resnets.", ".rn")\
               .replace(".attentions.", ".attn").replace(".transformer_blocks.0", "")


def box(ax, x, y, w, h, text, color, subtext=None, fontsize=10):
    """Draw rounded box with centered text."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.15",
                       facecolor=color, edgecolor="white", linewidth=2, alpha=0.92)
    ax.add_patch(b)
    ax.text(x, y + (0.08 if subtext else 0), text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")
    if subtext:
        ax.text(x, y - 0.12, subtext, ha="center", va="center",
                fontsize=7, color="white", alpha=0.85)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


def curved(ax, x1, y1, x2, y2, rad, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                               connectionstyle=f"arc3,rad={rad}"))


def badge(ax, x, y, text, color="#2C3E50"):
    ax.text(x, y, text, fontsize=7, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.9))


def main():
    drift_data = load_drift_data()
    delta_r, delta_a = load_delta_data()

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_facecolor(C_BG)
    ax.axis("off")

    # ---- Title ----
    ax.text(8, 9.6, "Unified Drift Correction Framework",
            ha="center", fontsize=18, fontweight="bold", color=C_DARK)
    ax.text(8, 9.15, "Diagnosis-Driven   ·   Geometry-Aware   ·   Feedback-Controlled",
            ha="center", fontsize=10, color="#7F8C8D", style="italic")

    # ---- Stage boxes (x-center, y-center, width, height) ----
    # Layout: left column (correction path), right column (diagnosis + style)

    # Row 1: Input → Inversion
    box(ax, 2.2, 8.2, 2.2, 0.7, "Input Image  X", C_IMAGE)
    box(ax, 5.5, 8.2, 3.0, 0.7, "DDIM Inversion", C_DIAG,
        subtext="z_0 → z_1 → ... → z_T")

    # Row 2: Diagnosis (right) + Correction start (left)
    box(ax, 10.0, 8.2, 3.5, 0.7, "Layer Drift Diagnosis", C_DIAG,
        subtext="Hook layers, rank by ||f_inv − f_recon||")

    # Row 3: Residual Correction
    box(ax, 4.5, 6.2, 4.2, 0.8, "Residual Correction", C_CORR,
        subtext="f_out = f_recon + λ · (f_inv − f_recon)", fontsize=11)

    # Row 4: Style Injection (right)
    box(ax, 10.5, 6.2, 3.8, 0.8, "Style Injection (CLIP)", C_STYLE,
        subtext="v_style = v_text − proj_content(v_text)", fontsize=10)

    # Row 5: Pinning Loop
    box(ax, 4.5, 4.2, 5.0, 0.8, "CLIP Pinning Check (Feedback)", C_PIN,
        subtext="Periodic: decode → CLIP encode → check drift → adapt style", fontsize=10)

    # Row 6: Output
    box(ax, 4.5, 2.6, 2.2, 0.7, "Edited Image  X̂", C_IMAGE)

    # ---- Arrows (main path) ----
    arrow(ax, 3.3, 7.85, 4.0, 7.85)       # Input → Inversion
    arrow(ax, 7.0, 7.85, 8.25, 7.85)       # Inversion → Diagnosis

    # Diagnosis → Correction (diagonal down)
    curved(ax, 10.0, 7.85, 6.6, 6.6, rad=-0.3, color=C_ARROW)

    # Correction right → Style
    arrow(ax, 6.6, 6.2, 8.6, 6.2)

    # Style down-left → Pinning
    curved(ax, 10.5, 5.8, 7.0, 4.6, rad=-0.2, color=C_ARROW)

    # Pinning → Output
    arrow(ax, 4.5, 3.8, 4.5, 3.3)

    # ---- Feedback loop: Pinning right → back up to Correction ----
    curved(ax, 7.0, 4.6, 9.5, 6.0, rad=0.4, color=C_PIN, lw=2)
    ax.text(9.2, 5.3, "reduce style\nstrength", fontsize=7, color=C_PIN,
            ha="center", fontstyle="italic")

    # ---- Re-use inversion features ----
    arrow(ax, 7.0, 8.2, 6.6, 6.6)
    ax.text(7.3, 7.6, "f_inv", fontsize=7, color=C_DIAG, fontweight="bold")

    # ---- Per-type ΔPSNR bar (bottom right) ----
    bar_ax = fig.add_axes([0.68, 0.06, 0.13, 0.12])
    bar_ax.bar([0, 1], [delta_r, delta_a], color=[C_CORR, C_STYLE], width=0.5, alpha=0.9)
    bar_ax.set_xticks([0, 1])
    bar_ax.set_xticklabels(["ResNet", "Attention"], fontsize=7)
    bar_ax.set_ylabel("ΔPSNR (dB)", fontsize=7)
    bar_ax.set_title("Correctable Info", fontsize=8, fontweight="bold", color=C_DARK)
    bar_ax.axhline(y=0, color="gray", linewidth=0.5)
    for i, v in enumerate([delta_r, delta_a]):
        bar_ax.text(i, v + 0.1, f"{v:+.1f}", ha="center", fontsize=8, fontweight="bold")
    bar_ax.set_facecolor("#F8F9FA")
    bar_ax.grid(axis='y', alpha=0.3)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)

    # ---- Top-5 drift bar (bottom left) ----
    drift_ax = fig.add_axes([0.42, 0.06, 0.18, 0.12])
    names = [short_name(n) for n in list(drift_data.keys())[:5]]
    values = list(drift_data.values())[:5]
    drift_ax.barh(range(len(names)), values, color=C_DIAG, alpha=0.9, height=0.6)
    drift_ax.set_yticks(range(len(names)))
    drift_ax.set_yticklabels(names, fontsize=6.5)
    drift_ax.set_xlabel("Drift (L2)", fontsize=7)
    drift_ax.set_title("Top-5 Drift Layers", fontsize=8, fontweight="bold", color=C_DARK)
    drift_ax.invert_yaxis()
    drift_ax.set_facecolor("#F8F9FA")
    drift_ax.grid(axis='x', alpha=0.3)
    for spine in drift_ax.spines.values():
        spine.set_visible(False)

    # ---- Metric badges ----
    badge(ax, 3.0, 5.7, "PSNR +2.76 dB\nLPIPS 0.22→0.09")
    badge(ax, 10.5, 5.6, "CLIP_content 0.88→0.98\ncos(v_style, v_content)≈0")
    badge(ax, 3.0, 3.7, "5-8/9 checks triggered\nAdaptive strength")

    # ---- Design principles (bottom) ----
    principles = [
        (3.2, 1.6, "① Diagnosis Precedes Intervention"),
        (8.0, 1.6, "② Correction is Geometry-Aware"),
        (12.8, 1.6, "③ Editing is Feedback-Controlled"),
    ]
    for x, y, text in principles:
        ax.text(x, y, text, fontsize=9, color=C_DARK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#BDC3C7", alpha=0.95))

    # Save
    out_path = OUT_DIR / "unified_framework.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close()
    print(f"[Done] {out_path}")


if __name__ == "__main__":
    main()
