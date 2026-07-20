# -*- coding: utf-8 -*-
"""Figure 1 (teaser): Architecture Fingerprint of Feature Drift — slim three-act.
Designed at 7.0 in wide; scaled to 5.5 in column -> fonts x0.786 (min 9pt -> 7pt).

v2: Fixed font overlaps — increased figure height 3.0→3.5, expanded profile
    vertical span, reduced icon/curve heights, tightened fonts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C = dict(sd15="#3B7DDD", sdxl="#17A2B8", dit="#F28E2B", flux="#D64541", sd35="#7B5EA7")
GREEN = "#2E8B57"; AMBER = "#B97A0F"
INK = "#1A1A1A"; GRAY = "#666666"; EDGE = "#C9CCD3"

rng = np.random.default_rng(7)

def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

def profile(n, peak_at, sig, dips=()):
    x = np.arange(n)
    y = gauss(x, peak_at, sig) + 0.06 + 0.025 * np.sin(x / 2.3 + 1.0)
    for mu, s, d in dips:
        y = y - d * gauss(x, mu, s)
    y = y + rng.normal(0, 0.012, n)
    y = np.clip(y, 0, None)
    return x, y / y.max()

# name, topology, color, x, y, inline-label, label-pos(axes frac, ha)
PROFILES = [
    ("SD 1.5",     "U-Net", C["sd15"], *profile(38, 29, 4.2),
     "up.2.res.0 = 2.97",  (0.02, "left")),
    ("SDXL",       "U-Net", C["sdxl"], *profile(28, 13, 3.2),
     "mid.res.1 = 36.03",  (0.02, "left")),
    ("HunyuanDiT", "DiT",   C["dit"],  *profile(40, 20, 3.6),
     "b20 = 13621",        (0.02, "left")),
    ("FLUX",       "MMDiT", C["flux"], *profile(57, 18, 4.5),
     "j18 = 0.71",         (0.98, "right")),
    ("SD 3.5",     "MMDiT", C["sd35"],
     *profile(24, 22, 2.6, dips=[(13, 1.8, 0.55)]),
     "b22 (late)",         (0.02, "left")),
]

# ---- figure: taller than before (3.0 -> 3.5) for breathing room ----
FIG_W, FIG_H = 7.0, 3.5
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.patch.set_facecolor("white")

bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

def bg_box(x0, y0, w, h, fc, ec, lw=1.0):
    bg.add_patch(FancyBboxPatch((x0, y0), w, h, transform=bg.transAxes,
                 boxstyle="round,pad=0.004,rounding_size=0.012",
                 fc=fc, ec=ec, lw=lw, zorder=0))

# Background panels
BOX_BOT = 0.030
BOX_TOP = 0.938
BOX_H = BOX_TOP - BOX_BOT
bg_box(0.025, BOX_BOT, 0.300, BOX_H, "white", EDGE)
bg_box(0.360, BOX_BOT, 0.280, BOX_H, "white", EDGE)
bg_box(0.675, BOX_BOT, 0.300, BOX_H, "white", EDGE)

# Arrows between acts
for xa, xb in [(0.328, 0.357), (0.643, 0.672)]:
    fig.patches.append(FancyArrowPatch(
        (xa, 0.50), (xb, 0.50), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, lw=2.0, color="#8A8F99", zorder=5))

# ---- helpers ----
def header(x0, num, title, subtitle):
    fig.text(x0, 0.908, num, fontsize=10, fontweight="bold", color="white",
             ha="left", va="center",
             bbox=dict(boxstyle="circle,pad=0.26", fc=INK, ec="none"))
    fig.text(x0 + 0.032, 0.908, title, fontsize=12, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.text(x0, 0.848, subtitle, fontsize=8.5, style="italic",
             color=GRAY, ha="left", va="center")


# ================================================================
# ACT 1 — Discovery
# ================================================================
header(0.043, "1", "Discovery", "topology, not sampling")
L = 0.043

# Profiles span — wider than before
PROF_TOP, PROF_BOT = 0.790, 0.220
n_prof = len(PROFILES)
row_h = (PROF_TOP - PROF_BOT) / n_prof          # ~0.114 per row

# Icon size (reduced: was 0.072 tall, now 0.048)
ICON_W, ICON_H = 0.030, 0.048
ICON_YO = 0.028            # yc - ICON_YO = icon bottom

# Curve axis size (reduced: was 0.048 tall, now 0.038)
CURVE_W = 0.315 - L - 0.042
CURVE_H = 0.038
CURVE_YO = 0.034           # yc - CURVE_YO = curve bottom

for i, (name, topo, col, x, y, layerlab, lpos) in enumerate(PROFILES):
    yc = PROF_TOP - (i + 0.5) * row_h

    # Architecture topology icon
    ax_i = fig.add_axes([L, yc - ICON_YO, ICON_W, ICON_H])
    ax_i.set_facecolor("none"); ax_i.set_xlim(0, 1); ax_i.set_ylim(0, 1)
    ax_i.axis("off")
    if topo == "U-Net":
        ax_i.plot([0.12, 0.12, 0.30, 0.50, 0.70, 0.88, 0.88],
                  [0.92, 0.30, 0.12, 0.07, 0.12, 0.30, 0.92],
                  color=col, lw=1.8, solid_capstyle="round")
        ax_i.plot([0.12, 0.88], [0.92, 0.92], color=col, lw=0.7,
                  ls=(0, (2, 2)), alpha=0.85)
    elif topo == "DiT":
        ax_i.add_patch(FancyBboxPatch(
            (0.34, 0.06), 0.32, 0.88,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            fc=col, ec="none", alpha=0.92))
    else:  # MMDiT
        ax_i.plot([0.26, 0.26, 0.42, 0.50], [0.96, 0.55, 0.40, 0.34],
                  color=col, lw=1.8, solid_capstyle="round")
        ax_i.plot([0.74, 0.74, 0.58, 0.50], [0.96, 0.55, 0.40, 0.34],
                  color=col, lw=1.8, solid_capstyle="round")
        ax_i.add_patch(FancyBboxPatch(
            (0.36, 0.04), 0.28, 0.30,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=col, ec="none", alpha=0.92))

    # Architecture name
    fig.text(L + 0.044, yc + 0.002, name, fontsize=8.5, fontweight="bold",
             color=col, ha="left", va="center")

    # Drift curve
    ax_s = fig.add_axes(
        [L + 0.042, yc - CURVE_YO, CURVE_W, CURVE_H])
    ax_s.set_facecolor("none")
    ax_s.plot(x, y, color=col, lw=1.5)
    ax_s.fill_between(x, 0, y, color=col, alpha=0.16)
    pk = int(np.argmax(y))
    ax_s.plot([x[pk]], [y[pk]], "o", ms=3, color=col,
              mec="white", mew=0.5, zorder=5)
    ax_s.text(lpos[0], 0.30, layerlab, transform=ax_s.transAxes,
              fontsize=7.5, color=INK, ha=lpos[1], va="center",
              bbox=dict(boxstyle="round,pad=0.15", fc="white",
                        ec="none", alpha=0.85))
    ax_s.set_xlim(x.min(), x.max()); ax_s.set_ylim(-0.08, 1.15)
    ax_s.axis("off")

# Stats below profiles
stat_y0 = PROF_BOT - 0.035
stat_dy = 0.045
stats = [("same-family  d = 0.249", 9, INK),
         ("diff-topology  d = 1.077", 9, INK),
         ("LOOCV  r = 0.999995", 8, INK)]
for j, (t, fs, fc) in enumerate(stats):
    fig.text(L, stat_y0 - j * stat_dy, t, fontsize=fs, color=fc,
             ha="left", va="center")


# ================================================================
# ACT 2 — Mechanism
# ================================================================
header(0.378, "2", "Mechanism", "skip-mediated conflict")
M = 0.382

# Causal chain boxes
chain = [("α", 0.040), ("C = ‖s−u‖", 0.078), ("φ$_{\\ell}$", 0.040),
         ("PSNR", 0.054)]
gap = 0.007
cy, ch = 0.680, 0.115
x0 = M
for i, (sym, cw) in enumerate(chain):
    fig.patches.append(FancyBboxPatch(
        (x0, cy), cw, ch, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        fc="#EEF2FB", ec=C["sd15"], lw=1.2, zorder=3))
    fig.text(x0 + cw / 2, cy + ch / 2, sym, fontsize=9, fontweight="bold",
             color=INK, ha="center", va="center", zorder=4)
    if i < 3:
        fig.patches.append(FancyArrowPatch(
            (x0 + cw + 0.001, cy + ch / 2),
            (x0 + cw + gap - 0.001, cy + ch / 2),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
            lw=1.5, color=C["sd15"], zorder=4))
    x0 += cw + gap

# Results
lines_sd15 = [
    (0.600, "cut peak skip (α = 0):", 9, "bold", INK),
    (0.550, "drift −27.7% (p = 1e-8)", 9, "bold", C["sd15"]),
    (0.500, "PSNR +2.20 dB", 9, "bold", C["sd15"]),
    (0.450, "low-drift skip: n.s.", 8.5, "normal", GRAY),
]
lines_sdxl = [
    (0.370, "same cut on SDXL:", 9, "bold", INK),
    (0.320, "−11.59 dB, opposite sign", 8.5, "normal", GRAY),
    (0.270, "→ architecture-specific", 8.5, "italic", GRAY),
]
for y, s, fs, wt, clr in lines_sd15 + lines_sdxl:
    kw = dict(fontsize=fs, color=clr, va="center")
    if wt == "italic":
        kw["fontstyle"] = "italic"
    else:
        kw["fontweight"] = wt
    fig.text(M, y, s, **kw)


# ================================================================
# ACT 3 — Application
# ================================================================
header(0.693, "3", "Application", "sufficient vs bounded")
A = 0.695

# Correction formula box
fig.patches.append(FancyBboxPatch(
    (A, 0.745), 0.258, 0.070, transform=fig.transFigure,
    boxstyle="round,pad=0.005,rounding_size=0.010",
    fc=INK, ec="none", zorder=3))
fig.text(A + 0.129, 0.780, "f$_{out}$ = f$_{recon}$ + λ·(f$_{inv}$ − f$_{recon}$)",
         fontsize=9, color="white", ha="center", va="center", zorder=4,
         fontweight="bold")

# Key claims
fig.text(A, 0.700, "recon: P2P-equiv · MB vs GB", fontsize=9,
         color=GREEN, fontweight="bold", va="center")

# ---- λ cliff plot (dual-axis) ----
axL = fig.add_axes([0.722, 0.220, 0.178, 0.420])
axL.set_facecolor("none")
axR = axL.twinx(); axR.set_facecolor("none")

xpos = np.array([0.006, 0.01, 0.05, 0.1, 0.5, 1.0])
gain = np.array([0, 28, 92, 99, 100, 100])
clip = np.array([0.140, 0.143, 0.041, 0.019, 0.023, 0.019])

axL.set_xscale("log"); axL.set_xlim(0.005, 1.35)
axL.axvspan(0.01, 0.05, color="#E8B4B4", alpha=0.35, zorder=0)
axL.axvspan(0.05, 1.0, color=GREEN, alpha=0.10, zorder=0)
axL.plot(xpos, gain, "-o", color=AMBER, lw=1.8, ms=3.5, zorder=4)
axR.plot(xpos, clip, "-s", color="#5B5BD6", lw=1.5, ms=3, zorder=4)
axL.set_ylim(-5, 118); axR.set_ylim(0.0, 0.165)
axL.set_xticks(xpos[[0, 2, 5]]); axL.set_xticklabels(["0", ".05", "1"])
axL.set_yticks([0, 50, 100])
axR.set_yticks([0.00, 0.05, 0.10, 0.15])
axL.tick_params(labelsize=8); axR.tick_params(labelsize=8, colors="#5B5BD6")
axL.set_xlabel("λ", fontsize=9, labelpad=1)
axL.spines["top"].set_visible(False); axR.spines["top"].set_visible(False)

# Labels inside plot
axL.text(0.008, 9, "LPIPS gain", fontsize=8.5, color=AMBER,
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
axR.text(0.40, 0.028, "CLIP-Dir", fontsize=8.5, color="#5B5BD6",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
axL.text(0.30, 42, "24×", fontsize=12, color=GREEN, fontweight="bold",
         ha="center")

# P2P escape point (green star)
axR.plot([0.3], [0.140], "*", ms=11, color=GREEN, mec=INK, mew=0.5, zorder=6)
axL.annotate("P2P escapes", xy=(0.26, 96), xytext=(0.0105, 68),
             fontsize=8.5, color=GREEN, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                       alpha=0.85),
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))

# Bottom line
fig.text(0.840, 0.068, "edit: either/or, not how-much", fontsize=8.5,
         style="italic", color=GRAY, va="center", ha="center")

# ---- save ----
import os
out_dir = "/tmp/fig1"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/fig1_teaser.png", dpi=300, facecolor="white")
fig.savefig(f"{out_dir}/fig1_teaser.pdf", facecolor="white")
print(f"done → {out_dir}/fig1_teaser.pdf")
