"""
Fig.2 — Two drift morphologies and the resolution limits of the fingerprint.
Panel A: normalized drift profiles of 7 architectures, colored by class.
Panel B: D_total heatmap of 21 pairs, class-clustered ordering.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

STORE = {
    "SD1.5":        ("outputs/phase1/layer_drift_summary_104img.json",               "interior"),
    "SDXL":         ("outputs/sdxl_phase1/layer_drift_summary_p_multi_104.json",     "interior"),
    "H-DiT":        ("outputs/dit_phase1/layer_drift_summary_p_multi_104.json",      "interior"),
    "FLUX":         ("outputs/phase9_flux_fp16/flux_p_multi_104.json",               "ramp"),
    "SD3.5":        ("outputs/sd35_phase1/layer_drift_summary_p_multi_104.json",     "ramp"),
    "PixArt-Sigma": ("outputs/p0b_cross_checkpoint/pixart_p_multi_104.json",         "ramp"),
    "PixArt-alpha": ("outputs/p0b_cross_checkpoint/pixart_alpha_p_multi_104.json",   "ramp"),
}

def load_profile(path):
    with open(path) as f:
        d = json.load(f)
    canon = d["canonical_layers"]
    prof = np.array([d["profile"][ln] for ln in canon], dtype=np.float64)
    pn = (prof - prof.min()) / (prof.max() - prof.min()) if prof.max() > prof.min() else prof
    return pn

def load_features(path):
    with open(path) as f:
        return json.load(f)["features"]

profiles, features = {}, {}
for name, (path, cls) in STORE.items():
    profiles[name] = load_profile(path)
    features[name] = load_features(path)

def dist(fa, fb):
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"],
                            fa["spread"]-fb["spread"]])
    return float(np.linalg.norm([d_pp, d_mag]))

# Class-clustered ordering: interior (by pp) then ramp (by pp)
interior_names = sorted([n for n, (_, c) in STORE.items() if c == "interior"],
                        key=lambda n: features[n]["peak_position"])
ramp_names = sorted([n for n, (_, c) in STORE.items() if c == "ramp"],
                    key=lambda n: features[n]["peak_position"])
order = interior_names + ramp_names
print("Order:", order)

# ── Figure ──
fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# Panel A: normalized profiles
warm = ["#d62728", "#ff7f0e", "#e377c2"]   # interior: red/orange/pink
cool = ["#1f77b4", "#17becf", "#2ca02c", "#9467bd"]  # ramp: blue/cyan/green/purple
for i, name in enumerate(interior_names):
    pn = profiles[name]
    x = np.linspace(0, 1, len(pn))
    label = f"{name} (pp={features[name]['peak_position']:.2f})"
    axA.plot(x, pn, color=warm[i], lw=2, label=label, alpha=0.9)
for i, name in enumerate(ramp_names):
    pn = profiles[name]
    x = np.linspace(0, 1, len(pn))
    label = f"{name} (pp={features[name]['peak_position']:.2f})"
    axA.plot(x, pn, color=cool[i], lw=2, label=label, alpha=0.9)
    # † marker at terminal peak
    axA.annotate("†", xy=(1.0, 1.0), xytext=(0.985, 0.93),
                 fontsize=12, color=cool[i], fontweight="bold",
                 annotation_clip=False)

axA.set_xlabel("normalized layer position", fontsize=11)
axA.set_ylabel("normalized drift", fontsize=11)
axA.set_title("(A) Drift profiles by morphology class (P-multi)", fontsize=12, fontweight="bold")
axA.legend(fontsize=8, loc="upper left", framealpha=0.9)
axA.grid(True, alpha=0.25, linestyle="--")

# Panel B: D_total heatmap
n = len(order)
D = np.zeros((n, n))
for i, na in enumerate(order):
    for j, nb in enumerate(order):
        D[i, j] = dist(features[na], features[nb])

im = axB.imshow(D, cmap="YlOrRd", vmin=0, vmax=0.7)
axB.set_xticks(range(n)); axB.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
axB.set_yticks(range(n)); axB.set_yticklabels(order, fontsize=9)
axB.set_title("(B) Structural distance D$_{total}$ (class-clustered)", fontsize=12, fontweight="bold")

# Class block outlines
ni = len(interior_names)
axB.add_patch(Rectangle((-0.5, -0.5), ni, ni, fill=False, edgecolor="#d62728",
                        lw=2.5, linestyle="--"))
axB.add_patch(Rectangle((ni-0.5, ni-0.5), n-ni, n-ni, fill=False, edgecolor="#1f77b4",
                        lw=2.5, linestyle="--"))

# Annotate values
for i in range(n):
    for j in range(n):
        if i != j:
            axB.text(j, i, f"{D[i,j]:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if D[i,j] > 0.35 else "black")

cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
cbar.set_label("D$_{total}$", fontsize=11)

# Class labels below
axA.text(0.5, -0.13, "red/orange: interior-localized   |   blue/green: terminal-accumulating († = terminal-censored)",
         transform=axA.transAxes, fontsize=9, ha="center", color="#444")

plt.tight_layout()
out = "/home/hiaskc/Talant/graduation/fig2_morphology.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

# Print the 21-pair matrix for reference
print("\n21-pair D_total matrix:")
for i, na in enumerate(order):
    for j, nb in enumerate(order):
        if j > i:
            print(f"  {na:>14s}-{nb:<14s}: {D[i,j]:.4f}")
