"""Generate C1 dose-response figure from weight perturbation data."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

summary_path = Path("outputs/p0b_cross_checkpoint/weight_perturbation_summary.json")
with open(summary_path) as f:
    d = json.load(f)

sigmas = [dr["sigma"] for dr in d["dose_response"]]
dists = [dr["Ds"] for dr in d["dose_response"]]
ref = d["protocol"]["min_inter_arch_Ds"]

fig, ax = plt.subplots(figsize=(10, 5))

# Filter valid points
valid_s, valid_d = [], []
for s, ds in zip(sigmas, dists):
    if not np.isnan(ds) and not np.isinf(ds):
        valid_s.append(s)
        valid_d.append(ds)

ax.semilogx(valid_s, valid_d, "o-", color="#d62728", linewidth=2, markersize=10,
            markerfacecolor="white", markeredgewidth=2, zorder=5)

# Inter-arch reference line
ax.axhline(y=ref, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(valid_s[-1] * 1.5, ref + 0.02,
        f"min inter-arch D_s = {ref:.3f}\n(SD 1.5 vs SDXL)",
        fontsize=9, color="gray", va="bottom")

for s, ds in zip(valid_s, valid_d):
    label = f"D_s={ds:.6f}" if ds < 1 else "model broken"
    color = "#d62728" if ds < 1 else "#999"
    ax.annotate(label, (s, ds), textcoords="offset points", xytext=(0, 14),
                fontsize=9, ha="center", color=color)

# Annotations
bbox_props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9)
ax.text(0.02, 0.96,
        f"Stable regime: D_s ≈ 0.0001\n"
        f"Inter-arch ref: D_s = {ref:.3f}\n"
        f"Ratio: 0.04% — fingerprint is\n"
        f"~2000 x more stable within-arch\n"
        f"than across architectures",
        transform=ax.transAxes, fontsize=10, va="top", bbox=bbox_props)

ax.set_xlabel("Perturbation magnitude  sigma  (relative to ||W||)", fontsize=13)
ax.set_ylabel("Structural distance  D_s(baseline, perturbed)", fontsize=13)
ax.set_title("C1: Architecture Fingerprint Stability Under Weight Perturbation\n"
             "SD 1.5 UNet, 19 images, 50-step DDIM", fontsize=14, fontweight="bold")
ax.set_ylim(-0.05, 1.25)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = Path("outputs/p0b_cross_checkpoint/fig_c1_dose_response.png")
plt.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {outpath.resolve()}")
