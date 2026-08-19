"""Small figure: SD1.5 four-variant P-multi profiles (nearly coincident)
overlaid with a FLUX ramp as contrast.

Data: canonical P-multi@104 store (band1_phase1 variants + flux).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["axes.unicode_minus"] = False

CJK = os.path.expanduser("~/.fonts/NotoSansCJKsc-Regular.otf")
if os.path.exists(CJK):
    font_manager.fontManager.addfont(CJK)
    _name = font_manager.FontProperties(fname=CJK).get_name()
else:
    _name = "DejaVu Sans"
plt.rcParams["font.family"] = [_name, "Songti SC", "Arial Unicode MS",
                               "DejaVu Sans"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = [
    ("SD1.4", "outputs/p0b_cross_checkpoint/band1_phase1/SD1.4_p_multi_104.json",
     "#14406e", 1.7),
    ("RealisticVision",
     "outputs/p0b_cross_checkpoint/band1_phase1/RealisticVision_p_multi_104.json",
     "#2b6a8f", 1.2),
    ("LCM-LoRA",
     "outputs/p0b_cross_checkpoint/band1_phase1/LCM-LoRA_p_multi_104.json",
     "#5b9fc0", 1.2),
    ("RandText",
     "outputs/p0b_cross_checkpoint/band1_phase1/RandText_p_multi_104.json",
     "#8fc1dd", 1.2),
]
FLUX = ("FLUX", "outputs/phase9_flux_fp16/flux_p_multi_104.json", "#a05a2c")


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        d = json.load(f)
    layers = d["canonical_layers"]
    vals = [d["profile"][l] for l in layers]
    x = [i / len(vals) for i in range(len(vals))]
    y = [v / max(vals) for v in vals]
    return x, y


fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.3))

for name, rel, color, lw in VARIANTS:
    x, y = load(rel)
    ax.plot(x, y, color=color, lw=lw, alpha=0.95, zorder=3, label=name)
x, y = load(FLUX[1])
ax.plot(x, y, color=FLUX[2], lw=1.7, zorder=2, label="FLUX (contrast)")

ax.scatter([0.6842], [1.0], s=45, color="#14406e", zorder=5)
ax.annotate("U2.R0 (0.684)", (0.6842, 1.0), textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=9, color="#14406e")
ax.scatter([0.9825], [1.0], s=45, color=FLUX[2], zorder=5)
ax.annotate("single_37 (0.983)†", (0.9825, 1.0), textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=9, color=FLUX[2])

ax.set_xlim(-0.02, 1.10)
ax.set_ylim(-0.08, 1.34)
ax.set_xlabel("normalized layer position (P-multi@104)", fontsize=9.5)
ax.set_ylabel("normalized mean drift (max = 1)", fontsize=9.5)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.5, 1.0])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", fontsize=8.5, frameon=False,
          handlelength=1.6, borderaxespad=0.2)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "fig1_lineage_overlay.pdf"), dpi=200)
fig.savefig("/tmp/fig1_lineage_overlay_preview.png", dpi=130)
print("saved fig1_lineage_overlay.pdf")
