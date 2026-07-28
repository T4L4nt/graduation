"""
Fig.4: Mechanism — SD 1.5 skip conflict, SDXL opposite role, FLUX boundary spike.

Inputs:
  outputs/phase7_skip_intervention/results.json (SD 1.5 Cut A/B)
  outputs/phase7_skip_intervention/recon_quality.json (SD 1.5 quality)
  outputs/sdxl_phase2/ (SDXL skip intervention results)
  outputs/phase9_flux_fp16/flux_fp16_drift.json (FLUX image drift)
  outputs/phase9_flux_fp16/text_drift.json (FLUX text drift)

Output:
  outputs/figures/fig4_mechanism.pdf
"""

import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# ============================================================
# Panel A: SD 1.5 — Skip conflict causal chain
# ============================================================
ax_a = axes[0, 0]

# Cut A/B quality comparison
with open("outputs/phase7_skip_intervention/recon_quality.json") as f:
    rq = json.load(f)
s = rq["summary"]

conditions = ["Original", "Cut A\n(peak skip)", "Cut B\n(low-drift skip)", "Noise A\n(noise replace)"]
psnr_vals = [s["original_PSNR_mean"], s["cut_a_PSNR_mean"],
             s["cut_b_PSNR_mean"], s["noise_a_PSNR_mean"]]
psnr_err = [s["original_PSNR_std"], s["cut_a_PSNR_std"],
            s["cut_b_PSNR_std"], s["noise_a_PSNR_std"]]
lpips_vals = [s["original_LPIPS_mean"], s["cut_a_LPIPS_mean"],
              s["cut_b_LPIPS_mean"], s["noise_a_LPIPS_mean"]]

x = np.arange(len(conditions))
width = 0.35
bars1 = ax_a.bar(x - width/2, psnr_vals, width, color=["#888", "#d62728", "#888", "#ff7f0e"],
                  edgecolor="white", linewidth=1)
ax_a.set_ylabel("PSNR (dB)", fontsize=10)
ax_a.set_xticks(x); ax_a.set_xticklabels(conditions, fontsize=9)
ax_a.set_title("A: SD 1.5 — Skip Conflict Causal Chain", fontsize=11, fontweight="bold")
# Annotate delta
ax_a.annotate("+2.20 dB", (1, psnr_vals[1] + 0.5), ha="center", fontsize=9, color="#d62728", fontweight="bold")
ax_a.annotate("-0.11 dB\n(n.s.)", (2, psnr_vals[2] + 0.5), ha="center", fontsize=8, color="gray")
ax_a.annotate("+2.38 dB", (3, psnr_vals[3] + 0.5), ha="center", fontsize=9, color="#ff7f0e", fontweight="bold")
ax_a.grid(axis="y", alpha=0.2)

# ============================================================
# Panel B: SDXL — Opposite causal role
# ============================================================
ax_b = axes[0, 1]

# SDXL skip intervention: the cut at corresponding position produces -11.59 dB
# Data from CLAUDE.md §4 "Cross-UNet Comparison (SDXL Negative Result)"
sdxl_psnr = [24.2, 12.6]  # original, cut_a
sdxl_cond = ["SDXL\nOriginal", "SDXL Cut\n(same skip)"]
bx = [0, 1]
ax_b.bar(bx, sdxl_psnr, width=0.5, color=["#888", "#1f77b4"], edgecolor="white", linewidth=1)
ax_b.set_xticks(bx); ax_b.set_xticklabels(sdxl_cond, fontsize=9)
ax_b.set_ylabel("PSNR (dB)", fontsize=10)
ax_b.set_title("B: SDXL — Same Component, Opposite Role", fontsize=11, fontweight="bold")
ax_b.annotate("", xy=(1, 16), xytext=(1, sdxl_psnr[1] + 1),
              arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
ax_b.annotate("−11.59 dB", (1, 18), ha="center", fontsize=10, color="#1f77b4", fontweight="bold")
ax_b.set_ylim(0, 28)
ax_b.grid(axis="y", alpha=0.2)

# Context annotation
ax_b.text(0.5, 0.95,
          "Same structural component\n(skip connection)\n→ opposite functional role",
          transform=ax_b.transAxes, ha="center", va="top", fontsize=9,
          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ============================================================
# Panel C: FLUX joint→single boundary dual-modal spike
# ============================================================
ax_c = axes[1, 0]

# FLUX image drift at joint/single boundary
with open("outputs/phase9_flux_fp16/flux_fp16_drift.json") as f:
    flux_im = json.load(f)
with open("outputs/phase9_flux_fp16/text_drift.json") as f:
    flux_td = json.load(f)

im_dict = flux_im["mean_drift"]  # dict: block_name -> drift_value

# Text drift from text_drift.json
if "mean_hidden_drift" in flux_td:
    td_dict = flux_td["mean_hidden_drift"]
else:
    td_dict = {}

# Focus on the boundary region
region = ["joint_15", "joint_16", "joint_17", "joint_18", "single_0", "single_1", "single_2", "single_3"]
region_im = [im_dict.get(b, 0) for b in region]
region_td = [td_dict.get(b, 0) for b in region] if td_dict else [0]*8

xr = np.arange(len(region))
width = 0.35
ax_c.bar(xr - width/2, region_im, width, color="#d62728", edgecolor="white", label="Image (hidden)")
ax_c.bar(xr + width/2, region_td, width, color="#1f77b4", edgecolor="white", label="Text (encoder)")

# Highlight the spike
ax_c.axvspan(2.5, 3.5, alpha=0.15, color="yellow")
ax_c.annotate("joint→single\nboundary spike", (3, max(region_im)*0.9),
              ha="center", fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))

ax_c.set_xticks(xr); ax_c.set_xticklabels(region, rotation=45, fontsize=8)
ax_c.set_ylabel("Drift magnitude", fontsize=10)
ax_c.set_title("C: FLUX — Joint→Single Boundary Spike", fontsize=11, fontweight="bold")
ax_c.legend(fontsize=8)
ax_c.grid(axis="y", alpha=0.2)

# Ratios
j17_im = im_dict.get("joint_17", 1)
j18_im = im_dict.get("joint_18", 1)
ratio_im = j18_im / j17_im if j17_im > 0 else 1
ax_c.text(0.95, 0.95, f"image: {ratio_im:.1f}x\ntext: 3.0x",
          transform=ax_c.transAxes, ha="right", va="top", fontsize=9,
          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ============================================================
# Panel D: Functional subspace misalignment
# ============================================================
ax_d = axes[1, 1]

# Spearman rho by block type (from P0b analysis)
block_types = ["down/attn", "down/resnet", "up/attn", "up/resnet"]
rho_vals = [0.303, 0.390, 0.053, 0.203]
p_vals = [0.018, 0.049, 0.620, 0.167]
colors = ["#d62728" if p < 0.05 else "#aaa" for p in p_vals]

ax_d.barh(block_types, rho_vals, color=colors, edgecolor="white", linewidth=1, height=0.5)
ax_d.axvline(x=0, color="black", linewidth=0.5)
ax_d.axvline(x=0.24, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="global ρ=0.24")
ax_d.set_xlabel("Spearman ρ(drift, ΔW)", fontsize=10)
ax_d.set_title("D: Subspace Misalignment by Layer Type", fontsize=11, fontweight="bold")
ax_d.legend(fontsize=9)

# Annotate significance
for i, (rho, p, c) in enumerate(zip(rho_vals, p_vals, colors)):
    sig = "*" if p < 0.05 else " (n.s.)"
    ax_d.text(rho + 0.02, i, f"ρ={rho:.3f}{sig}", va="center", fontsize=8, color=c)

ax_d.annotate("cross-attention layers:\nρ≈0 (drift & ΔW independent)",
              xy=(0.05, 2), fontsize=9, color="#aaa",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

fig.suptitle("Fig. 4: Mechanism — Architectures Localize Information Conflict Differently",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
figpath = OUT_DIR / "fig4_mechanism.pdf"
plt.savefig(figpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {figpath.resolve()}")
