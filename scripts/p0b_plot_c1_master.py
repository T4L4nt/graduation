"""
Generate the master C1 dose-response figure with all data points:
  - v4 gaussian perturbation curve (D_total, D_pp, D_shape, D_mag vs epsilon)
  - SD 1.4 → SD 1.5 checkpoint D_s (with ΔW annotation)
  - RV → SD 1.5 checkpoint D_s (with ΔW annotation)
  - Noise floor band (median ± p95)
  - Cross-architecture reference range (min D_s to max D_s)
  - Random init reference point
"""

import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/p0b_cross_checkpoint")

# --- Load v4 dose-response data ---
with open(OUT_DIR / "weight_perturbation_v4_summary.json") as f:
    v4 = json.load(f)

epsilons = sorted(set(r["epsilon"] for r in v4["results"]))
nf = v4["noise_floor"]
ref = v4["reference"]

# Aggregate per epsilon
eps_data = {}
for eps in epsilons:
    eps_rs = [r for r in v4["results"] if abs(r["epsilon"] - eps) < 1e-12 and r.get("status") != "crashed"]
    if not eps_rs:
        continue
    eps_data[eps] = {
        "D_total": float(np.mean([r["D_total"] for r in eps_rs if "D_total" in r and not np.isnan(r["D_total"])])),
        "D_pp": float(np.mean([r["D_peak_pos"] for r in eps_rs])),
        "D_shape": float(np.mean([r["D_shape"] for r in eps_rs])),
        "D_mag": float(np.mean([r["D_mag"] for r in eps_rs])),
        "UNet": float(np.mean([r["UNet_norm"] for r in eps_rs if not np.isnan(r.get("UNet_norm", float("nan")))])),
        "PSNR": float(np.mean([r["PSNR"] for r in eps_rs if not np.isnan(r.get("PSNR", float("nan")))])),
    }

# --- Checkpoint data ---
checkpoints = {
    "SD1.4→SD1.5": {"D_total": 0.0106, "D_pp": 0.000, "D_shape": 0.004, "D_mag": 0.005,
                      "ΔW": 0.209, "label": "continue training", "color": "#2ca02c"},
    "RV→SD1.5": {"D_total": 0.0425, "D_pp": 0.000, "D_shape": 0.036, "D_mag": 0.023,
                 "ΔW": 0.139, "label": "full fine-tune", "color": "#ff7f0e"},
}

# --- Cross-architecture reference (from v2 matrix) ---
cross_arch = {
    "min": 0.092,  # FLUX-SD3.5
    "max": 0.618,  # DiT-FLUX
}

# --- Random init ---
random_init = {"D_total": 0.699}

# =====================================================================
# FIGURE
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# --- Gaussian dose-response curves ---
eps_vals = np.array(sorted(eps_data.keys()))
d_total = np.array([eps_data[e]["D_total"] for e in eps_vals])
d_pp = np.array([eps_data[e]["D_pp"] for e in eps_vals])
d_shape = np.array([eps_data[e]["D_shape"] for e in eps_vals])
d_mag = np.array([eps_data[e]["D_mag"] for e in eps_vals])

# Only plot valid (non-NaN) points
valid = ~np.isnan(d_total)
ax.semilogx(eps_vals[valid], d_total[valid], "o-", color="#d62728", linewidth=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2, label="D_total (v4 gaussian)")
ax.semilogx(eps_vals[valid], d_shape[valid], "s--", color="#9467bd", linewidth=1.5,
            markersize=5, alpha=0.8, label="D_shape")
ax.semilogx(eps_vals[valid], d_mag[valid], "^--", color="#8c564b", linewidth=1.5,
            markersize=5, alpha=0.8, label="D_mag")

# D_pp line at 0
ax.axhline(y=0, color="green", linestyle=":", linewidth=1, alpha=0.5)
ax.annotate("D_pp ≡ 0", (1.5e-4, 0.003), fontsize=9, color="green")

# --- Noise floor band ---
nf_median = nf["noise_D_total"] if "noise_D_total" in nf else 0.0071
nf_p95 = nf["noise_D_total_p95"] if "noise_D_total_p95" in nf else 0.0163
ax.axhspan(0, nf_p95, alpha=0.08, color="gray")
ax.axhline(y=nf_p95, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.annotate(f"noise floor p95 = {nf_p95:.3f}", (eps_vals[valid][0], nf_p95 + 0.008),
            fontsize=8, color="gray")

# --- Cross-architecture range ---
ax.axhspan(cross_arch["min"], cross_arch["max"], alpha=0.06, color="red")
ax.axhline(y=cross_arch["min"], color="red", linestyle="-.", linewidth=1, alpha=0.6)
ax.annotate(f"min inter-arch D_s = {cross_arch['min']:.3f}\n(FLUX-SD3.5)",
            (eps_vals[valid][-1], cross_arch["min"] + 0.02), fontsize=8, color="red", ha="right")
ax.axhline(y=cross_arch["max"], color="red", linestyle="-.", linewidth=1, alpha=0.4)
ax.annotate(f"max inter-arch = {cross_arch['max']:.3f}",
            (eps_vals[valid][-1], cross_arch["max"] - 0.03), fontsize=8, color="red", ha="right")

# --- Checkpoint data points (plotted at their ΔW on the x-axis) ---
for name, cp in checkpoints.items():
    x_pos = cp["ΔW"]
    ax.plot(x_pos, cp["D_total"], "D", color=cp["color"], markersize=12,
            markeredgewidth=1.5, markerfacecolor="white", zorder=10)
    ax.annotate(f"{name}\nD_s={cp['D_total']:.4f}\nΔW={cp['ΔW']:.3f}\n({cp['label']})",
                (x_pos, cp["D_total"]), textcoords="offset points",
                xytext=(15, 15), fontsize=8, color=cp["color"],
                arrowprops=dict(arrowstyle="->", color=cp["color"], alpha=0.5))

# --- Random init ---
ax.annotate(f"random init\nD_s={random_init['D_total']:.3f}",
            (1e-6, random_init["D_total"]), fontsize=8, color="purple",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# --- Stable regime annotation ---
ax.axvspan(eps_vals[valid][0], eps_vals[valid][7], alpha=0.05, color="green")
ax.annotate("STABLE\nREGIME\n(D_total < noise floor)\nε ≤ 1e-4",
            (3e-5, 0.04), fontsize=9, color="green", fontweight="bold",
            ha="center", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.2))

# --- ΔW→D_s disconnect annotation ---
bbox_props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.95)
ax.text(0.02, 0.98,
        f"Gaussian perturbation: stable to ε ≤ 1e-4\n"
        f"Real ΔW: 0.139–0.209 (2000× beyond gaussian limit)\n"
        f"Real D_s: 0.011–0.043 (< min inter-arch 0.092)\n"
        f"→ Weight-space L2 distance does not\n"
        f"  predict fingerprint distance",
        transform=ax.transAxes, fontsize=9, va="top", bbox=bbox_props)

ax.set_xlabel("Perturbation / weight-change magnitude  (gaussian: ε, checkpoint: ‖ΔW‖/‖W‖)", fontsize=12)
ax.set_ylabel("Structural distance  D_s (v2 metric, continuous only)", fontsize=12)
ax.set_title("C1: Architecture Fingerprint Stability — Complete Dose-Response\n"
             "SD 1.5 UNet, 19 images, 50-step DDIM", fontsize=14, fontweight="bold")
ax.set_ylim(-0.02, 0.75)
ax.grid(True, alpha=0.2)
ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

plt.tight_layout()
figpath = OUT_DIR / "fig_c1_master_dose_response.png"
plt.savefig(figpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {figpath.resolve()}")
