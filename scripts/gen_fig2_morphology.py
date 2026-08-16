"""
Fig.2 — Two drift morphologies and the resolution limits of the fingerprint.
Panel A: normalized drift profiles of 7 architectures, colored by class.
Panel B: morphology-aware distance heatmap (21 pairs, class-clustered).
  - Ramp-class block shows D_mag (D_pp is degenerate by construction).
  - Other cells show D_total = sqrt(D_pp^2 + D_mag^2).
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORE = {
    "SD1.5":        ("outputs/phase1/layer_drift_summary_104img.json",               "interior", "#a50f45"),
    "SDXL":         ("outputs/sdxl_phase1/layer_drift_summary_p_multi_104.json",     "interior", "#d62728"),
    "H-DiT":        ("outputs/dit_phase1/layer_drift_summary_p_multi_104.json",      "interior", "#ff7f0e"),
    "FLUX":         ("outputs/phase9_flux_fp16/flux_p_multi_104.json",               "ramp",     "#08306b"),
    "SD3.5":        ("outputs/sd35_phase1/layer_drift_summary_p_multi_104.json",     "ramp",     "#1f77b4"),
    "PixArt-Σ":     ("outputs/p0b_cross_checkpoint/pixart_p_multi_104.json",         "ramp",     "#17becf"),
    "PixArt-α":     ("outputs/p0b_cross_checkpoint/pixart_alpha_p_multi_104.json",   "ramp",     "#2ca02c"),
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

profiles, features, colors = {}, {}, {}
for name, (path, cls, col) in STORE.items():
    profiles[name] = load_profile(path)
    features[name] = load_features(path)
    colors[name] = col

def d_pp(fa, fb): return abs(fa["peak_position"] - fb["peak_position"])
def d_mag(fa, fb):
    return float(np.linalg.norm([fa["concentration"]-fb["concentration"],
                                 fa["spread"]-fb["spread"]]))
def d_total(fa, fb): return float(np.linalg.norm([d_pp(fa, fb), d_mag(fa, fb)]))

interior_names = sorted([n for n, (_, c, _) in STORE.items() if c == "interior"],
                        key=lambda n: features[n]["peak_position"])
ramp_names = sorted([n for n, (_, c, _) in STORE.items() if c == "ramp"],
                    key=lambda n: features[n]["peak_position"])
order = interior_names + ramp_names
ni = len(interior_names)
n = len(order)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ── Panel A ──
for name in interior_names:
    pn = profiles[name]
    x = np.linspace(0, 1, len(pn))
    axA.plot(x, pn, color=colors[name], lw=2,
             label=f"{name} (pp={features[name]['peak_position']:.2f})", alpha=0.9)
for name in ramp_names:
    pn = profiles[name]
    x = np.linspace(0, 1, len(pn))
    axA.plot(x, pn, color=colors[name], lw=2,
             label=f"{name} (pp={features[name]['peak_position']:.2f})†", alpha=0.9)

axA.set_xlabel("normalized layer position", fontsize=11)
axA.set_ylabel("normalized drift", fontsize=11)
axA.set_title("(A) Drift profiles by morphology class (P-multi)", fontsize=12, fontweight="bold")
axA.legend(fontsize=8, loc="upper left", framealpha=0.9)
axA.grid(True, alpha=0.25, linestyle="--")

# ── Panel B ──
D = np.zeros((n, n))
for i, na in enumerate(order):
    for j, nb in enumerate(order):
        D[i, j] = d_total(features[na], features[nb])

im = axB.imshow(D, cmap="YlOrRd", vmin=0, vmax=0.6)
axB.set_xticks(range(n)); axB.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
axB.set_yticks(range(n)); axB.set_yticklabels(order, fontsize=9)
axB.set_title("(B) Morphology-aware distance matrix", fontsize=12, fontweight="bold")

# Class separator lines (no boxes)
axB.axhline(y=ni-0.5, color="black", lw=1.5)
axB.axvline(x=ni-0.5, color="black", lw=1.5)
# Class labels above the x-axis
axB.text((ni-1)/2, -1.35, "interior-localized", ha="center", fontsize=10,
         color="#a50f45", fontweight="bold")
axB.text(ni + (n-ni-1)/2, -1.35, "terminal-accumulating (†)", ha="center", fontsize=10,
         color="#08306b", fontweight="bold")

# Cell annotations: ramp block shows D_mag (3 decimals), others D_total (2 decimals)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        in_ramp_block = (i >= ni) and (j >= ni)
        val = d_mag(features[order[i]], features[order[j]]) if in_ramp_block else D[i, j]
        txt = f"{val:.3f}" if in_ramp_block else f"{val:.2f}"
        axB.text(j, i, txt, ha="center", va="center", fontsize=8,
                 color="white" if val > 0.3 else "black")

# Ramp block label
axB.text(n-0.5, ni-0.45, "D$_{mag}$\n(D$_{pp}$ degenerate)",
         ha="right", va="top", fontsize=7.5, color="#08306b")

cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
cbar.set_label("D$_{total}$", fontsize=11)

fig.text(0.5, 0.015,
         "Figure 2. (A) Normalized drift profiles under P-multi (104 images each), ordered by morphology class, then by peak position. "
         "Class membership is assigned from profile morphology (peak position and monotonicity), not from a distance threshold. "
         "† = terminal-censored: the peak lies on the last hooked layer, so post-peak falloff is unverifiable. "
         "(B) Structural distance between all 21 pairs. Within the terminal-accumulating block (lower right), cells report D$_{mag}$ at three decimals, "
         "because D$_{pp}$ is degenerate there by construction; all other cells report D$_{total}$ = √(D$_{pp}$² + D$_{mag}$²).",
         fontsize=8.5, ha="center", wrap=True)

plt.tight_layout(rect=[0, 0.045, 1, 1])
out = "/home/hiaskc/Talant/graduation/fig2_morphology.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

print("\nRamp-block D_mag (3 decimals):")
for i in range(ni, n):
    for j in range(ni, n):
        if j > i:
            print(f"  {order[i]:>12s}-{order[j]:<12s}: {d_mag(features[order[i]], features[order[j]]):.4f}")
