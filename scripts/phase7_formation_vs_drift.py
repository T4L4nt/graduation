#!/usr/bin/env python
"""
Phase 7c: Formation (FeatureInject) vs Drift (Ours) comparison.

Overlays where semantic representations FORM (per FeatureInject,
"One Size Does Not Fit All", OpenReview id=slCmiGEX1D) against where
inversion-reconstruction feature drift CONCENTRATES (our measurements),
for all four architectures.

Core argument: these two quantities are correlated (same architecture
drives both) but NOT identical. FeatureInject analyzes the FORWARD
generation trajectory and never inverts; our object of study
(inversion-reconstruction consistency) does not exist in their work.
The tallest drift peaks sit outside the formation zones (UNet decoder
terminus; FLUX early-single + last-joint), refuting the claim that
drift is a mere by-product of representation formation.

Pure CPU. Reuses pre-computed drift data via phase7_arch_topo_mapping.

Usage:
    python scripts/phase7_formation_vs_drift.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from phase7_arch_topo_mapping import load_drift_data, normalize_drift, ARCH_COLORS

OUT_DIR = Path("outputs/phase7_editing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FeatureInject formation zones, expressed in normalized layer depth [0, 1].
# Values transcribed from the paper's per-architecture claims
# (OpenReview id=slCmiGEX1D). `inferred` marks a zone extrapolated from an
# analogous DiT (SD3.5), since FeatureInject studied SD3.5+FLUX while we use
# HunyuanDiT+FLUX. FLUX is the one directly comparable model.
FORMATION = {
    "SD 1.5": dict(
        zones=[(0.35, 0.55)],
        label="formation: late-enc / early-dec (bottleneck)",
        verdict="DIVERGE",
        note="global-max drift at decoder ResNet (up_blocks.2),\noutside the formation zone",
        inferred=False,
    ),
    "SDXL": dict(
        zones=[(0.40, 0.62)],
        label="formation: bottleneck centrality",
        verdict="OVERLAP",
        note="both concentrate at the bottleneck —\nhonestly explained by the info-funnel mechanism",
        inferred=False,
    ),
    "DiT": dict(
        zones=[(0.55, 0.78)],
        label="formation: mid-late blocks (SD3.5 analog)",
        verdict="PARTIAL",
        note="ours peaks earlier (transition blocks 11-21);\nHunyuanDiT not directly studied by FeatureInject",
        inferred=True,
    ),
    "FLUX": dict(
        zones=[(0.93, 1.0)],
        label="formation: final transformer block",
        verdict="DIVERGE",
        note="ours: early single blocks + joint_18 (last joint).\nDirect FLUX-vs-FLUX comparison",
        inferred=False,
    ),
}

VERDICT_COLOR = {"DIVERGE": "#2E7D32", "OVERLAP": "#757575", "PARTIAL": "#EF6C00"}

ARCH_LABELS = {
    "SD 1.5": "SD 1.5  (UNet + DDIM)",
    "SDXL": "SDXL  (UNet + DDIM)",
    "DiT": "HunyuanDiT  (Transformer + v-pred)",
    "FLUX": "FLUX  (MM-DiT + Flow Matching)",
}


def main():
    print("Loading drift data...")
    data = load_drift_data()
    arch_names = ["SD 1.5", "SDXL", "DiT", "FLUX"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    fig.subplots_adjust(hspace=0.55, top=0.90, bottom=0.11, left=0.08, right=0.82)

    for ax, arch in zip(axes, arch_names):
        drift = normalize_drift(data[arch]["drift"], pct=98)
        n = len(drift)
        x = np.linspace(0, 1, n)
        color = ARCH_COLORS[arch]

        # Our measured drift profile.
        ax.fill_between(x, 0, drift, color=color, alpha=0.22, zorder=1)
        ax.plot(x, drift, color=color, lw=1.6, zorder=2,
                label="Ours: inversion-reconstruction drift")

        # Global-max drift marker (the crux of the argument).
        imax = int(np.argmax(drift))
        ax.plot(x[imax], drift[imax], "o", color=color, ms=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.annotate("drift peak", xy=(x[imax], drift[imax]),
                    xytext=(x[imax], min(drift[imax] + 0.22, 1.08)),
                    fontsize=8, color=color, ha="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

        # FeatureInject formation zone(s).
        spec = FORMATION[arch]
        for (zs, ze) in spec["zones"]:
            hatch = "//" if spec["inferred"] else None
            ax.axvspan(zs, ze, facecolor="#B0BEC5", alpha=0.35, zorder=0,
                       hatch=hatch, edgecolor="#607D8B", lw=0.5,
                       label="FeatureInject: semantic formation zone")

        # Verdict badge.
        v = spec["verdict"]
        ax.text(1.015, 0.72, v, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color="white", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=VERDICT_COLOR[v],
                          edgecolor="none"))
        ax.text(1.015, 0.30, spec["note"], transform=ax.transAxes, fontsize=7.2,
                color="#444", ha="left", va="center")

        ax.set_title(ARCH_LABELS[arch], fontsize=10.5, fontweight="bold",
                     color=color, loc="left", pad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("norm. drift", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.005, -0.30, spec["label"], transform=ax.transAxes,
                fontsize=7.5, color="#546E7A", style="italic")
        if arch == "FLUX":
            ax.set_xlabel("normalized layer depth  (input → output)", fontsize=9)

        # Single shared legend from the first axis.
        if arch == "SD 1.5":
            handles, labels = ax.get_legend_handles_labels()
            seen, h2, l2 = set(), [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l); h2.append(h); l2.append(l)
            ax.legend(h2, l2, fontsize=8, loc="upper right",
                      framealpha=0.9, ncol=1)

    fig.suptitle(
        "Where semantics FORM (FeatureInject) vs. where inversion DRIFTS (Ours)",
        fontsize=14, fontweight="bold", y=0.965)
    fig.text(0.5, 0.955,
             "Correlated (shared architecture) but not identical. FeatureInject analyzes forward "
             "generation and never inverts — inversion-reconstruction consistency is our object, absent from theirs.",
             fontsize=8.5, ha="center", color="#555", style="italic")
    fig.text(0.5, 0.035,
             "Hatched band = zone inferred from an analogous DiT (SD3.5); all others transcribed directly. "
             "DIVERGE: tallest drift peak lies outside the formation zone.",
             fontsize=7, ha="center", color="#888", style="italic")

    out = OUT_DIR / "formation_vs_drift_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
