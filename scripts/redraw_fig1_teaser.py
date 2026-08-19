"""Redraw Figure 1 (concept/motivation) with canonical P-multi values only.

Panel A: six real normalized drift profiles from the canonical P-multi@104
store, colored by morphology class; PixArt-alpha as a lineage probe.
Panel B: true distance magnitudes with the bootstrap noise floor.

Only canonical values are used (design constraint from review):
peaks SD 1.5 U2.R0 (0.684), SDXL mid_block.resnets.1 (0.429),
H-DiT blocks.20 (0.500), FLUX single_37 (0.983), PixArt-Sigma T27 (0.964),
SD 3.5 block_23 (0.958); distances D_mag in [0.0045, 0.0232] (lineage),
D_s in [0.30, 0.59] (between classes), untrained D_pp = 0.658,
noise floor p95 = 0.0042.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["axes.unicode_minus"] = False
_pref = ["Songti SC", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in _pref if f in _avail] + ["sans-serif"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (model label, profile json (relative to ROOT), color, peak annotation)
MODELS = [
    ("SD 1.5", "outputs/phase1/layer_drift_summary_104img.json",
     "#14406e", dict(peak="U2.R0", pp=0.684, xytext=(0, 16), ha="center")),
    ("SDXL", "outputs/sdxl_phase1/layer_drift_summary_p_multi_104.json",
     "#2b6a8f", dict(peak="mid_block.resnets.1", pp=0.429,
                    xytext=(16, 12), ha="right")),
    ("H-DiT", "outputs/dit_phase1/layer_drift_summary_p_multi_104.json",
     "#5b9fc0", dict(peak="blocks.20", pp=0.500, xytext=(-4, -30),
                    ha="center")),
    ("FLUX", "outputs/phase9_flux_fp16/flux_p_multi_104.json",
     "#7a3b12", dict(peak="single_37", pp=0.983, censored=True,
                    stack=(1.00, 1.36))),
    ("PixArt-Σ", "outputs/p0b_cross_checkpoint/pixart_p_multi_104.json",
     "#a05a2c", dict(peak="T27", pp=0.964, censored=True,
                    stack=(1.00, 1.07))),
    ("SD 3.5", "outputs/sd35_phase1/layer_drift_summary_p_multi_104.json",
     "#d08a52", dict(peak="block_23", pp=0.958, censored=True,
                    stack=(1.00, 0.76))),
]
PROBE = ("PixArt-α", "outputs/p0b_cross_checkpoint/pixart_alpha_p_multi_104.json",
         "#8c4a1a")

# distance magnitudes (canonical)
lineage_lo, lineage_hi = 0.0045, 0.0232       # D_mag, SD1.5 family, P-multi
noise_p95 = 0.0042                            # bootstrap noise floor p95
cross_lo, cross_hi = 0.30, 0.59               # D_s between morphology classes
untrained = 0.658                             # D_pp random-init vs trained


def load_profile(rel):
    with open(os.path.join(ROOT, rel)) as f:
        d = json.load(f)
    layers = d["canonical_layers"]
    vals = [d["profile"][l] for l in layers]
    x = [i / len(vals) for i in range(len(vals))]
    y = [v / max(vals) for v in vals]
    return x, y


fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]}
)

# ---- Panel A: real normalized drift profiles ----
axA.axvspan(0.80, 1.02, color="#f2e6d8", zorder=0)
axA.axvspan(-0.05, 0.80, color="#e2eef4", zorder=0)
axA.text(0.05, 1.55, "localized", ha="left", fontsize=11,
         color="#2b6a8f", fontweight="bold")
axA.text(0.84, 1.55, "accumulated", ha="left", fontsize=11,
         color="#a05a2c", fontweight="bold")

_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2)

for name, rel, color, ann in MODELS:
    x, y = load_profile(rel)
    (line,) = axA.plot(x, y, color=color, lw=1.6, zorder=2)
    pp = ann["pp"]
    marker_y = y[round(pp * len(y))]
    axA.scatter([pp], [marker_y], s=55, color=color, zorder=4)
    dagger = "†" if ann.get("censored") else ""
    label = f"{name}\n{ann['peak']} ({pp:.3f}){dagger}"
    if ann.get("stack"):
        axA.annotate(label, (pp, marker_y), xytext=ann["stack"],
                     ha="left", va="center", fontsize=8.5, color=color,
                     bbox=_BBOX, zorder=6,
                     arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                     alpha=0.7))
    else:
        axA.annotate(label, (pp, marker_y), textcoords="offset points",
                     xytext=ann["xytext"], ha=ann.get("ha", "center"),
                     fontsize=8.5, color=color, bbox=_BBOX, zorder=6)

x, y = load_profile(PROBE[1])
axA.plot(x, y, color=PROBE[2], lw=1.5, ls=(0, (4, 2)), zorder=3)
pp = 0.964
axA.scatter([pp], [y[round(pp * len(y))]], s=95, facecolor="none",
            edgecolor=PROBE[2], linewidth=1.4, zorder=5)
axA.annotate("PixArt-α (lineage probe)\nT27 (0.964)†", (pp, 1.0),
             xytext=(1.00, 0.49), ha="left", va="center",
             fontsize=8, color=PROBE[2], style="italic",
             bbox=_BBOX, zorder=6,
             arrowprops=dict(arrowstyle="-", color=PROBE[2], lw=0.8,
                             alpha=0.7))

axA.set_xlim(-0.02, 1.48)
axA.set_ylim(-0.35, 1.72)
axA.set_xlabel("normalized layer position (P-multi@104, canonical layer order)",
               fontsize=9.5)
axA.set_ylabel("normalized mean drift (max = 1)", fontsize=9.5)
axA.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_yticks([0.0, 0.5, 1.0])
for s in axA.spines.values():
    s.set_visible(False)
axA.set_title("(A) Architecture fingerprint space — canonical drift profiles",
              fontsize=11, fontweight="bold", loc="left")

# ---- Panel B: distance magnitudes (real values, log scale) ----
axB.axvline(noise_p95, ymin=0, ymax=0.86, color="gray", ls="--", lw=1,
            zorder=2)
axB.text(noise_p95 * 1.15, 2.46, "noise floor\np95 = 0.0042", ha="left",
         fontsize=8, color="gray")

bars = [
    ("within-lineage\nD_mag", lineage_lo, lineage_hi, "#2b6a8f", "right"),
    ("between\nmorphology classes\nD_s", cross_lo, cross_hi, "#a05a2c",
     "above"),
    ("untrained vs trained\nD_pp", untrained, None, "#7a7a7a", "right"),
]
for i, (label, lo, hi, color, place) in enumerate(bars):
    y = 2.4 - i
    if hi is not None:
        axB.barh(y, hi - lo, left=lo, height=0.5, color=color, alpha=0.85,
                 zorder=3)
        if place == "above":
            axB.text((lo * hi) ** 0.5, y + 0.33, f"[{lo:.4f}, {hi:.4f}]",
                     va="center", ha="center", fontsize=9, color=color)
        else:
            axB.text(hi * 1.35, y, f"[{lo:.4f}, {hi:.4f}]", va="center",
                     fontsize=9, color=color)
    else:
        axB.barh(y, lo, height=0.5, color=color, alpha=0.85, zorder=3)
        axB.text(lo * 1.35, y, f"{lo:.3f}", va="center", fontsize=9,
                 color=color)
    axB.text(0.0008, y, label, va="center", ha="left", fontsize=8.5)

axB.set_xscale("log")
axB.set_xlim(0.0005, 4.0)
axB.set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
axB.set_xticklabels(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"])
axB.set_ylim(0.0, 2.8)
axB.set_yticks([])
axB.set_xlabel("structural distance (log scale, P-multi@104)", fontsize=9.5)
axB.set_title("(B) Magnitudes that motivate the fingerprint", fontsize=11,
              fontweight="bold", loc="left")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.spines["left"].set_visible(False)

fig.suptitle("Feature drift is not noise: it organizes by architecture",
             fontsize=13, fontweight="bold", x=0.02, ha="left")
fig.text(0.02, 0.015,
         "† censored peak: profile accumulates monotonically to the final "
         "layer under P-multi truncation",
         fontsize=8, style="italic", color="gray")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(os.path.join(ROOT, "fig1_teaser_v2.pdf"), dpi=200)
fig.savefig("/tmp/fig1_teaser_v2_preview.png", dpi=130)
print("saved fig1_teaser_v2.pdf")
