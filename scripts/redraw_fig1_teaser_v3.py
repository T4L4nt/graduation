"""Redraw Figure 1 as a single-column-height, two-panel teaser.

Panel A: SD1.5 base + 4 variants (SD1.4/RV/LCM/RandText) overlaid,
annotated with D_pp = 0, D_mag <= 0.023.
Panel B: six architecture profiles colored by morphology class
(warm = inner-layer peak, cool = terminal ramp), peak labels with
canonical P-multi@104 values.

All profiles come from the canonical P-multi@104 store.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        d = json.load(f)
    layers = d["canonical_layers"]
    vals = [d["profile"][l] for l in layers]
    x = [i / len(vals) for i in range(len(vals))]
    y = [v / max(vals) for v in vals]
    return x, y


fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(6.8, 2.55), gridspec_kw={"width_ratios": [1, 1.12]}
)

# ---- Panel A: lineage invariance ----
A_CURVES = [
    ("SD1.5 (base)", "outputs/phase1/layer_drift_summary_104img.json",
     "#0d3356", 2.0),
    ("SD1.4", "outputs/p0b_cross_checkpoint/band1_phase1/SD1.4_p_multi_104.json",
     "#14406e", 1.2),
    ("RV", "outputs/p0b_cross_checkpoint/band1_phase1/RealisticVision_p_multi_104.json",
     "#2b6a8f", 1.2),
    ("LCM", "outputs/p0b_cross_checkpoint/band1_phase1/LCM-LoRA_p_multi_104.json",
     "#5b9fc0", 1.2),
    ("RandText", "outputs/p0b_cross_checkpoint/band1_phase1/RandText_p_multi_104.json",
     "#8fc1dd", 1.2),
]
for name, rel, color, lw in A_CURVES:
    x, y = load(rel)
    axA.plot(x, y, color=color, lw=lw, zorder=2)

axA.text(0.02, 1.13, "SD1.5 base + 4 variants\n"
         "(SD1.4 · RV · LCM · RandText)\n"
         "D_pp = 0,  D_mag ≤ 0.023",
         ha="left", va="top", fontsize=7.5, color="#0d3356",
         bbox=_BBOX, zorder=6)

axA.set_xlim(-0.02, 1.05)
axA.set_ylim(-0.08, 1.30)
axA.set_xlabel("normalized layer position", fontsize=8)
axA.set_ylabel("normalized drift", fontsize=8)
axA.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
axA.set_yticks([0.0, 0.5, 1.0])
axA.tick_params(labelsize=7)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.set_title("(A) Same architecture, different weights\n"
              "→ same fingerprint",
              fontsize=8.5, fontweight="bold", loc="left", pad=3)

# ---- Panel B: morphology dichotomy ----
B_CURVES = [
    ("SD1.5", "outputs/phase1/layer_drift_summary_104img.json",
     "#8c2d1b", 1.4, dict(pp=0.684, label="SD1.5 0.684", pos=(0.705, 1.22))),
    ("SDXL", "outputs/sdxl_phase1/layer_drift_summary_p_multi_104.json",
     "#c0392b", 1.1, dict(pp=0.429, label="SDXL 0.429", pos=(0.429, 0.78))),
    ("H-DiT", "outputs/dit_phase1/layer_drift_summary_p_multi_104.json",
     "#e67e22", 1.1, dict(pp=0.500, label="H-DiT 0.500", pos=(0.500, 1.12))),
    ("FLUX", "outputs/phase9_flux_fp16/flux_p_multi_104.json",
     "#1b4f72", 1.1, dict(pp=0.983, label="FLUX 0.983†", pos=(0.983, 1.10))),
    ("PixArt-Σ", "outputs/p0b_cross_checkpoint/pixart_p_multi_104.json",
     "#2e86c1", 1.1, dict(pp=0.964, label="PixArt-Σ 0.964†", pos=(0.964, 1.24))),
    ("SD3.5", "outputs/sd35_phase1/layer_drift_summary_p_multi_104.json",
     "#7fb3d5", 1.1, dict(pp=0.958, label="SD3.5 0.958†", pos=(0.958, 0.80))),
]
for name, rel, color, lw, ann in B_CURVES:
    x, y = load(rel)
    axB.plot(x, y, color=color, lw=lw, zorder=2)
    axB.scatter([ann["pp"]], [1.0], s=22, color=color, zorder=4)
    axB.annotate(ann["label"], (ann["pp"], 1.0), xytext=ann["pos"],
                 ha="center", va="center", fontsize=6.5,
                 color=color, bbox=_BBOX, zorder=6)

axB.set_xlim(-0.02, 1.13)
axB.set_ylim(-0.08, 1.32)
axB.set_xlabel("normalized layer position", fontsize=8)
axB.set_yticks([0.0, 0.5, 1.0])
axB.set_yticklabels([])
axB.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
axB.tick_params(labelsize=7)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.set_title("(B) Different architectures → two morphologies,\n"
              "not predicted by textbook labels",
              fontsize=8.5, fontweight="bold", loc="left", pad=3)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "fig1_teaser_v2.pdf"), dpi=200)
print("saved fig1_teaser_v2.pdf")
