"""
P0b: Boundary model spectrum — one figure showing all within-architecture variants.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = [
    ("SD 1.4", 0.0106, "continue training", "#1f77b4"),
    ("LCM LoRA", 0.0211, "distillation LoRA", "#ff7f0e"),
    ("Rand text", 0.0339, "Gaussian encoder", "#2ca02c"),
    ("RV", 0.0425, "full fine-tune", "#d62728"),
]

ref = {
    "noise_p95": 0.0163,
    "min_inter": 0.092,
    "cross_arch_range": (0.092, 0.618),
}

fig, ax = plt.subplots(figsize=(8, 4))

x = np.arange(len(data))
vals = [d[1] for d in data]
colors = [d[3] for d in data]
labels = [d[0] for d in data]
tags = [d[2] for d in data]

bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

# Noise floor band
ax.axhspan(0, ref["noise_p95"], alpha=0.08, color="green")
ax.axhline(y=ref["noise_p95"], color="green", linestyle="--", linewidth=1)
ax.text(len(data)-0.5, ref["noise_p95"]+0.003, f"noise floor p95 = {ref['noise_p95']:.3f}",
        fontsize=9, color="green", ha="right")

# Inter-arch region
ax.axhspan(ref["cross_arch_range"][0], ref["cross_arch_range"][1], alpha=0.06, color="red")
ax.axhline(y=ref["min_inter"], color="red", linestyle="-.", linewidth=1)
ax.text(len(data)-0.5, ref["min_inter"]+0.005, f"min inter-arch = {ref['min_inter']:.3f}",
        fontsize=9, color="red", ha="right")

# Value labels
for bar, val, tag in zip(bars, vals, tags):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"  {val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, val/2, tag,
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("D_s(vs SD 1.5)", fontsize=12)
ax.set_title("C1: Within-Architecture Fingerprint Distance Spectrum\n"
             "SD 1.5 UNet variants, 19 images, 50-step DDIM", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.12)
ax.grid(axis="y", alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", alpha=0.15, label=f"Noise floor (p95={ref['noise_p95']:.3f})"),
    Patch(facecolor="red", alpha=0.08, label=f"Cross-arch range [{ref['cross_arch_range'][0]:.3f}, {ref['cross_arch_range'][1]:.3f}]"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

plt.tight_layout()
fp = "outputs/p0b_cross_checkpoint/fig_boundary_spectrum.png"
plt.savefig(fp, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {fp}")
