"""
Generate ICLR 2027 main-paper figures.

5 Figures + 2 Tables for 8-page main text.

Figure 2: Architecture Fingerprint — definition + 5-arch overlay + similarity
Figure 3: Topology → Fingerprint mapping (data evidence)
Figure 5: Diagnosis-guided correction + editing examples

(Figures 1 and 4 are conceptual — drawn in draw.io)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent palette
C_ARCH = {  # per-architecture
    "SD 1.5": "#3498DB", "SDXL": "#E74C3C",
    "HunyuanDiT": "#2ECC71", "FLUX": "#F39C12", "SD 3.5": "#9B59B6",
}
C_DOWN, C_MID, C_UP, C_INT = "#27AE60", "#F39C12", "#2E86C1", "#E74C3C"
C_NOISE = "#E67E22"
C_DARK = "#2C3E50"

# ===========================================================================
# Unified data loader
# ===========================================================================

def load_drift_profile(path, arch_name=""):
    """Load drift profile in architectural (depth) order.

    Handles three data formats:
      Format 1: {"aggregated": {layer: {"mean": ...}}}        — SD 1.5, SD 3.5
      Format 2: {"per_image": {layer: {img: val}}}            — SDXL, DiT
      Format 3: {"drift": {"joint_0": {"hidden_drift": ...}}} — FLUX

    CRITICAL: full_ranking is sorted by drift magnitude (not depth) and is
    NOT suitable as a drift profile. Use per_image/aggregated instead.
    """
    with open(path) as f:
        data = json.load(f)

    # Format 1: aggregated
    if "aggregated" in data:
        agg = data["aggregated"]
        k0 = list(agg.keys())[0]
        if isinstance(agg[k0], dict) and "mean" in agg[k0]:
            layers = sorted(agg.keys(),
                          key=lambda x: (x.split(".")[0], len(x), x))
            vals = np.array([agg[l]["mean"] for l in layers])
            return layers, vals

    # Format 2: per_image (SDXL, DiT) — sort by architectural depth
    if "per_image" in data:
        per_image = data["per_image"]
        # SDXL: layers like "down_blocks.0.resnets.0"
        # DiT: layers like "blocks.0"
        # Sort: DiT blocks by numeric index; others by section then name
        k0 = list(per_image.keys())[0]
        if k0.startswith("blocks."):
            # DiT: sort by block number
            layers = sorted(per_image.keys(),
                          key=lambda x: int(x.replace("blocks.", "")))
        else:
            # SDXL / UNet: sort by section (down→mid→up) then name
            layers = sorted(per_image.keys(),
                          key=lambda x: (x.split(".")[0], len(x), x))
        vals = np.array([np.mean([float(v) for v in per_image[l].values()])
                        for l in layers])
        return layers, vals

    # Format 3: FLUX drift
    if "drift" in data:
        drift_data = data["drift"]
        k0 = list(drift_data.keys())[0]
        if isinstance(drift_data[k0], dict) and "hidden_drift" in drift_data[k0]:
            joint_names = [f"joint_{i}" for i in range(19)]
            single_names = [f"single_{i}" for i in range(38)]
            ordered = joint_names + single_names
            vals = np.array([drift_data[n]["hidden_drift"] for n in ordered])
            return ordered, vals

    raise ValueError(f"Cannot parse {arch_name}: keys={list(data.keys())}")


def load_all_architectures():
    paths = {
        "SD 1.5": "outputs/phase1/layer_drift_summary.json",
        "SDXL": "outputs/sdxl_phase1/layer_drift_summary.json",
        "HunyuanDiT": "outputs/dit_phase1/layer_drift_summary.json",
        "FLUX": "outputs/phase6_flux/diagnosis_summary.json",
        "SD 3.5": "outputs/sd35_phase1/layer_drift_summary.json",
    }
    profiles = {}
    for name, p in paths.items():
        try:
            _, vals = load_drift_profile(p, name)
            profiles[name] = vals
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
    return profiles


# ===========================================================================
# Figure 2: Architecture Fingerprint
# ===========================================================================

def fig2_fingerprint():
    """Figure 2: (a) 5-arch overlay (b) structural distance matrix (no interpolation).

    Panel B uses architecture-level structural features (peak position, modality,
    drift concentration, spread) computed directly from raw layer counts — no
    interpolation to a common length. Avoids the artifact where interpolating
    28-point SDXL to 57 points (51% synthetic) inflates similarity with DiT.
    """
    profiles = load_all_architectures()
    if len(profiles) < 3:
        print("[Fig 2] Not enough architectures loaded, skipping")
        return

    # Interpolate to unified length (Panel A: qualitative visual only)
    unified_len = 57
    unified = {}
    for name, vals in profiles.items():
        v = vals / (vals.max() + 1e-8)
        if len(v) != unified_len:
            x_old = np.linspace(0, 1, len(v))
            x_new = np.linspace(0, 1, unified_len)
            v = np.interp(x_new, x_old, v)
        unified[name] = v

    # Structural features (NO interpolation — raw layer counts)
    def extract_features(vals):
        """Extract 4D structural feature vector from raw drift values."""
        vn = vals / (vals.max() + 1e-8)
        n = len(vn)
        peak_idx = np.argmax(vn)
        peak_pos = peak_idx / max(1, n - 1)

        # Count significant local maxima (>0.3 normalized)
        peaks = sum(1 for i in range(1, n - 1)
                    if vn[i] > vn[i - 1] and vn[i] > vn[i + 1] and vn[i] > 0.3)
        n_peaks = min(peaks, 9)  # cap for display

        # Drift concentration: fraction of layers above 0.5
        concentration = (vn > 0.5).sum() / n

        # Spread: normalized span of above-noise (>0.1) layers
        above = np.where(vn > 0.1)[0]
        spread = (above[-1] - above[0]) / n if len(above) > 0 else 0.0

        return np.array([peak_pos, n_peaks / 10.0, concentration, spread])

    names_list = list(profiles.keys())
    n = len(names_list)
    feat_vecs = {name: extract_features(profiles[name]) for name in names_list}

    # Structural distance matrix (Euclidean, 0 = identical)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(
                feat_vecs[names_list[i]] - feat_vecs[names_list[j]])

    # ---- Figure layout ----
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    # --- Panel A: 5-architecture overlay (qualitative, interpolated) ---
    ax = fig.add_subplot(gs[0])
    for name, vals in unified.items():
        x = np.arange(len(vals))
        ax.plot(x, vals, color=C_ARCH[name], linewidth=2.2, alpha=0.9, label=name)

    for name, vals in unified.items():
        pi = np.argmax(vals)
        ax.annotate(name, (pi, vals[pi]), color=C_ARCH[name], fontsize=8,
                   fontweight="bold", xytext=(0, 6), textcoords="offset points",
                   ha="center")

    ax.set_xlabel("Layer Index (interpolated, qualitative)", fontsize=11, color=C_DARK)
    ax.set_ylabel("Normalized Drift Phi(M)", fontsize=11, color=C_DARK)
    ax.set_title("Architecture Fingerprints: 5 Architectures, 2 Paradigms",
                fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(alpha=0.25)
    ax.text(0.02, -0.12,
            "(qualitative overlay -- interpolation for visual alignment only)",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")

    # --- Panel B: Structural distance matrix ---
    ax2 = fig.add_subplot(gs[1])
    # Use Greens colormap (dark = close, light = far)
    im = ax2.imshow(dist, cmap="YlOrRd", vmin=0, vmax=max(1.0, dist.max()),
                    aspect="equal")

    short_names = [n.replace("HunyuanDiT", "H-DiT") for n in names_list]
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_title("Structural Distance (Euclidean)\nlower = more similar",
                 fontsize=12, fontweight="bold", color=C_DARK)

    for i in range(n):
        for j in range(n):
            val = dist[i, j]
            # Black text on light cells, white on dark
            c = "white" if val > 0.4 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=c)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # Caption with actual closest/farthest pairs
    # Find closest and farthest off-diagonal pairs
    triu_idx = np.triu_indices(n, k=1)
    closest = np.argmin(dist[triu_idx])
    farthest = np.argmax(dist[triu_idx])
    i_c, j_c = triu_idx[0][closest], triu_idx[1][closest]
    i_f, j_f = triu_idx[0][farthest], triu_idx[1][farthest]
    fig.text(0.5, -0.03,
             f"Panel B: Structural distance from 4 raw-layer-count features "
             f"(peak position, number of peaks, concentration, spread) -- NO interpolation. "
             f"Closest: {short_names[i_c]}--{short_names[j_c]} d={dist[i_c,j_c]:.3f}; "
             f"Farthest: {short_names[i_f]}--{short_names[j_f]} d={dist[i_f,j_f]:.3f}.",
             ha="center", fontsize=8, color="gray", style="italic")

    fig.suptitle("Figure 2: Architecture Fingerprint -- Phi(M) is reproducible, architecture-specific",
                fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    fig.savefig(OUT_DIR / "fig2_fingerprint.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_fingerprint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 2] -> outputs/figures/fig2_fingerprint.pdf")


# ===========================================================================
# Figure 3: Topology → Fingerprint — Three Mapping Principles
# ===========================================================================

def _compute_fwhm(vals_n):
    """Full width at half maximum for a normalized drift profile."""
    peak_idx = np.argmax(vals_n)
    half_max = vals_n[peak_idx] / 2.0
    # Left crossing
    left = peak_idx
    for i in range(peak_idx, -1, -1):
        if vals_n[i] < half_max:
            left = i
            break
    # Right crossing
    right = peak_idx
    for i in range(peak_idx, len(vals_n)):
        if vals_n[i] < half_max:
            right = i
            break
    return right - left, left, right


def fig3_topology_mapping():
    """Figure 3: Three Mapping Principles — From architecture topology to drift profile.

    Panel A (Bottleneck Localization): Measured drift peak falls within the
      independently predicted bottleneck region for all 5 architectures.
    Panel B (Propagation Mode): Cross-layer skip connections -> broad peak;
      sequential residual stream only -> narrow, localized peak.
    Panel C (Cross-modal Boundary Effect): Interaction direction determines
      whether the boundary produces a drift peak or drift valley.
    """
    profiles = load_all_architectures()
    if len(profiles) < 3:
        print("[Fig 3] Not enough architectures, skipping")
        return

    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.35)

    # Predicted bottleneck windows (from ICLR_PAPER_DEFINITIONS.md Table)
    # Format: (arch_name, bottleneck_type, prediction_window_layers)
    arch_bottleneck = {
        "SD 1.5": ("decoder entry\n(encoder→decoder junction)", "U-Net"),
        "SDXL": ("mid_block\n(information funnel)", "U-Net"),
        "HunyuanDiT": ("blocks 11–21\n(representation transition)", "Transformer\nsingle-stream"),
        "FLUX": ("joint_18→single_0\n(joint→single handoff)", "MM-DiT\ndual-stream"),
        "SD 3.5": ("blocks 22–24\n(output compression)", "MM-DiT-X"),
    }

    # =========================================================================
    # Panel A: Bottleneck Localization — 5 mini drift profiles
    # =========================================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off")
    ax_a.set_title("Principle 1: Bottleneck Localization",
                   fontsize=12, fontweight="bold", color=C_DARK, loc="left", y=1.02)

    names_ordered = ["SD 1.5", "SDXL", "HunyuanDiT", "FLUX", "SD 3.5"]
    n_arch = len([n for n in names_ordered if n in profiles])
    row_h = 0.85 / max(n_arch, 1)

    for idx, name in enumerate(names_ordered):
        if name not in profiles:
            continue
        y_top = 0.88 - idx * row_h
        y_mid = y_top - row_h * 0.5

        vals = profiles[name]
        vals_n = vals / (vals.max() + 1e-8)

        # Mini drift curve axis (inset)
        left_margin = 0.22
        curve_w = 0.72
        curve_h = row_h * 0.75
        ax_curve = ax_a.inset_axes([left_margin, y_top - curve_h, curve_w, curve_h])
        x = np.arange(len(vals_n))
        ax_curve.fill_between(x, 0, vals_n, color=C_ARCH[name], alpha=0.25)
        ax_curve.plot(x, vals_n, color=C_ARCH[name], linewidth=1.2)
        pi = np.argmax(vals_n)
        ax_curve.plot(pi, vals_n[pi], 'o', color="#E74C3C", markersize=5, zorder=5)
        ax_curve.set_xlim(0, len(vals_n) - 1)
        ax_curve.set_ylim(0, 1.05)
        ax_curve.axis("off")

        # Architecture label
        info = arch_bottleneck.get(name, ("unknown", "unknown"))
        ax_a.text(0.01, y_mid, f"{name}\n({info[1]})", fontsize=7.5,
                  fontweight="bold", color=C_ARCH[name], va="center")
        ax_a.text(left_margin - 0.01, y_top + 0.01, info[0], fontsize=6.5,
                  color="gray", va="top", style="italic")

    # Bottom annotation
    ax_a.text(0.5, -0.06,
              "Red dot = measured peak. All 5 fall within independently\n"
              "predicted bottleneck region (p≈3×10⁻⁴ under random placement).",
              transform=ax_a.transAxes, fontsize=7.5, color=C_DARK, ha="center",
              style="italic")

    # =========================================================================
    # Panel B: Propagation Mode — Peak width comparison
    # =========================================================================
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_title("Principle 2: Propagation Mode",
                   fontsize=12, fontweight="bold", color=C_DARK)

    # Compare SD 1.5 (UNet, skip, broad) vs HunyuanDiT (Transformer, no skip, narrow)
    for name, label, ls in [("SD 1.5", "U-Net (skip connections) → broad peak", "-"),
                              ("HunyuanDiT", "Transformer (residual only) → narrow peak", "--")]:
        if name not in profiles:
            continue
        vals_n = profiles[name] / (profiles[name].max() + 1e-8)
        # Normalize x to [0, 1] (percentage through network)
        x_norm = np.linspace(0, 1, len(vals_n))
        ax_b.plot(x_norm, vals_n, color=C_ARCH[name], linewidth=2.2,
                  linestyle=ls, label=label)

        # Compute and annotate FWHM
        fwhm, left, right = _compute_fwhm(vals_n)
        fwhm_frac = fwhm / len(vals_n)
        pi = np.argmax(vals_n)
        ax_b.axhline(y=vals_n[pi] / 2, xmin=x_norm[left], xmax=x_norm[right],
                     color=C_ARCH[name], linewidth=0.8, alpha=0.4)
        ax_b.annotate(f"FWHM = {fwhm_frac:.0%} of layers",
                      xy=(x_norm[pi], vals_n[pi] / 2),
                      fontsize=8, color=C_ARCH[name], fontweight="bold",
                      ha="center", va="bottom")

    ax_b.set_xlabel("Normalized Layer Position", fontsize=10, color=C_DARK)
    ax_b.set_ylabel("Normalized Drift Φ", fontsize=10, color=C_DARK)
    ax_b.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax_b.grid(alpha=0.2)

    # =========================================================================
    # Panel C: Cross-modal Boundary Effect
    # =========================================================================
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_title("Principle 3: Cross-modal\nBoundary Effect",
                   fontsize=12, fontweight="bold", color=C_DARK)

    # FLUX: joint→single boundary — interaction REMOVED → drift peak
    if "FLUX" in profiles:
        vals = profiles["FLUX"]
        vals_n = vals / (vals.max() + 1e-8)
        x = np.arange(len(vals_n))
        # Mark joint/single boundary at index 19 (0-indexed: joint_0..18, single_0..37)
        boundary_idx = 19
        ax_c.plot(x[:boundary_idx], vals_n[:boundary_idx],
                  color="#F39C12", linewidth=1.8, label="FLUX joint (dual-stream)")
        ax_c.plot(x[boundary_idx - 1:], vals_n[boundary_idx - 1:],
                  color="#E67E22", linewidth=1.8, linestyle="--",
                  label="FLUX single (no cross-modal)")
        ax_c.axvline(x=boundary_idx - 1, color="#F39C12", linewidth=1.2,
                     linestyle=":", alpha=0.7)
        pi_f = np.argmax(vals_n)
        ax_c.annotate("joint→single\nboundary\nDRIFT PEAK",
                      xy=(pi_f, vals_n[pi_f]),
                      xytext=(pi_f + 8, vals_n[pi_f] + 0.08),
                      fontsize=7.5, color="#E74C3C", fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2))

    # SD 3.5: dual→standard boundary — interaction ADDED → drift valley
    if "SD 3.5" in profiles:
        vals35 = profiles["SD 3.5"]
        vals35_n = vals35 / (vals35.max() + 1e-8)
        x35 = np.arange(len(vals35_n))
        dual_end = 13  # layers 0-12 are dual attention
        ax_c.plot(x35[:dual_end], vals35_n[:dual_end],
                  color="#8E44AD", linewidth=1.8, label="SD 3.5 dual (cross-modal)")
        ax_c.plot(x35[dual_end - 1:], vals35_n[dual_end - 1:],
                  color="#9B59B6", linewidth=1.8, linestyle="--",
                  label="SD 3.5 standard (no cross-modal)")
        ax_c.axvline(x=dual_end - 1, color="#8E44AD", linewidth=1.2,
                     linestyle=":", alpha=0.7)
        # Valley at transition
        valley_idx = dual_end - 1 + np.argmin(vals35_n[dual_end - 1:dual_end + 5])
        ax_c.annotate("dual→standard\nboundary\nDRIFT VALLEY",
                      xy=(valley_idx, vals35_n[valley_idx]),
                      xytext=(valley_idx + 6, vals35_n[valley_idx] - 0.15),
                      fontsize=7.5, color="#2ECC71", fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1.2))

    ax_c.set_xlabel("Layer Index", fontsize=10, color=C_DARK)
    ax_c.set_ylabel("Normalized Drift Φ", fontsize=10, color=C_DARK)
    ax_c.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax_c.grid(alpha=0.2)
    ax_c.text(0.98, 0.02,
              "Interaction removed → PEAK\nInteraction added → VALLEY",
              transform=ax_c.transAxes, fontsize=8, color=C_DARK,
              ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9E79F", alpha=0.6))

    fig.suptitle("Figure 3: From Architecture Topology to Drift Fingerprint — Three Mapping Principles",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.savefig(OUT_DIR / "fig3_topology.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_topology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 3] → outputs/figures/fig3_topology.pdf")


# ===========================================================================
# Figure 5: Diagnosis-guided Correction — Application Evidence
# ===========================================================================

def fig5_application():
    """Figure 5: Diagnosis-guided Correction — Three testable sub-propositions.

    Panel A (P_λ): λ cliff — correction is all-or-nothing (L-shaped frontier).
    Panel B (P_pos): Position sensitivity is architecture-dependent and predictable.
    Panel C (Editing): Content anchor — LPIPS improves but CLIP-Dir collapses.
    """
    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.1], wspace=0.35)

    # =========================================================================
    # Panel A: λ Cliff — P_λ evidence
    # =========================================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_title("P_λ: λ Cliff — Correction is All-or-Nothing",
                   fontsize=12, fontweight="bold", color=C_DARK)

    # Try loading λ cliff data
    lambda_lpips = None
    try:
        with open("outputs/phase7_editing_100image/planb_summary.json") as f:
            planb = json.load(f)
        frontier = planb.get("lambda_frontier", {})
        lambdas = []
        lpips_vals = []
        clipdir_vals = []
        # lambda_frontier mixes numeric keys ("0.0", "0.01", ...) with
        # sub-dicts ("planb", "midstep") — extract only the numeric entries.
        for k in sorted(frontier.keys(), key=lambda x: float(x) if x.replace(".", "").isdigit() else 999):
            if not k.replace(".", "").isdigit():
                continue
            lambdas.append(float(k))
            lpips_vals.append(frontier[k]["LPIPS"])
            clipdir_vals.append(frontier[k]["CLIPDir"])
        lambda_lpips = (np.array(lambdas), np.array(lpips_vals), np.array(clipdir_vals))
    except Exception as e:
        print(f"  [Fig 5] λ cliff data unavailable: {e}")

    if lambda_lpips is not None:
        lam, lp, cd = lambda_lpips
        ax_lp = ax_a
        ax_cd = ax_a.twinx()

        # LPIPS (left y-axis)
        line1, = ax_lp.plot(lam, lp, 'o-', color="#2E86C1", linewidth=2.2,
                            markersize=8, label="LPIPS↓ (content preservation)")
        ax_lp.set_xlabel("λ", fontsize=11, color=C_DARK)
        ax_lp.set_ylabel("LPIPS↓", fontsize=11, color="#2E86C1")
        ax_lp.tick_params(axis="y", labelcolor="#2E86C1")

        # CLIP-Dir (right y-axis)
        line2, = ax_cd.plot(lam, cd, 's--', color="#E74C3C", linewidth=2.2,
                            markersize=8, label="CLIP-Dir↑ (edit fidelity)")
        ax_cd.set_ylabel("CLIP-Dir↑", fontsize=11, color="#E74C3C")
        ax_cd.tick_params(axis="y", labelcolor="#E74C3C")

        # Shade the cliff region (0.01–0.05) and plateau (0.05–1.0)
        ax_lp.axvspan(0.01, 0.05, alpha=0.08, color="#E74C3C")
        ax_lp.axvspan(0.05, 1.0, alpha=0.05, color="#27AE60")
        ax_lp.annotate("CLIFF\n(0.01–0.05)\nwidth 0.04",
                        xy=(0.03, lp[1]), fontsize=8, color="#E74C3C",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax_lp.annotate("PLATEAU [0.05, 1.0]\nwidth 0.95 (24× wider)\n≥90% LPIPS gain",
                        xy=(0.5, 0.15), fontsize=8, color="#27AE60",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Combined legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax_lp.legend(lines, labels, fontsize=8, loc="upper right")
        ax_lp.grid(alpha=0.2)

        ax_lp.text(0.5, -0.18,
                   "L-shaped frontier: no λ achieves both plateau-level\n"
                   "content preservation AND intact edit fidelity.",
                   transform=ax_lp.transAxes, fontsize=8, color=C_DARK,
                   ha="center", style="italic")
    else:
        ax_a.text(0.5, 0.5, "[λ cliff data unavailable]", ha="center", va="center",
                  transform=ax_a.transAxes, fontsize=11, color="gray")
        ax_a.axis("off")

    # =========================================================================
    # Panel B: Cross-architecture ΔPSNR + Position Sensitivity (P_pos)
    # =========================================================================
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_title("P_pos: Architecture-dependent\nPosition Sensitivity",
                   fontsize=12, fontweight="bold", color=C_DARK)

    archs = ["SD 1.5", "SDXL", "HunyuanDiT", "FLUX"]
    delta_psnr = [2.75, 5.23, 5.65, 3.94]
    pos_types = [
        "Robust\n(random5≈top5)",
        "Robust\n(random5≈top5)",
        "Critical\n(transition>>top5)",
        "Exact equivalence\n(all positions =)",
    ]
    colors_bar = [C_ARCH[a] for a in archs]

    y_pos = range(len(archs))
    bars = ax_b.barh(y_pos, delta_psnr, color=colors_bar, height=0.55,
                     edgecolor="white", linewidth=0.8)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(archs, fontsize=10, fontweight="bold")
    ax_b.set_xlabel("ΔPSNR (dB)", fontsize=10, color=C_DARK)
    ax_b.axvline(x=0, color="black", linewidth=0.5)

    # Position sensitivity annotation on right side
    for i, (v, pt) in enumerate(zip(delta_psnr, pos_types)):
        ax_b.text(v + 0.3, i, f"+{v:.1f} dB", fontsize=9, va="center",
                  fontweight="bold", color=C_DARK)
        ax_b.text(max(delta_psnr) + 1.5, i, pt, fontsize=7.5, va="center",
                  color="gray", style="italic")

    ax_b.set_xlim(0, max(delta_psnr) + 4.5)
    ax_b.grid(axis="x", alpha=0.25)

    # P_simple note at bottom
    ax_b.text(0.5, -0.22,
              "P_simple: Feature-level Δ=−0.27 dB | DCSC no gain | Plan B falsified",
              transform=ax_b.transAxes, fontsize=8, color=C_DARK, ha="center",
              style="italic",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#F9E79F", alpha=0.4))

    # =========================================================================
    # Panel C: 100-image Editing — Content Anchor Story
    # =========================================================================
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_title("Editing: Content Anchor, Not Edit Enhancer",
                   fontsize=12, fontweight="bold", color=C_DARK)

    try:
        with open("outputs/phase7_editing_100image/evaluation_with_clipdir.json") as f:
            edit_data = json.load(f)
        s = edit_data["summary"]
    except Exception as e:
        s = None
        print(f"  [Fig 5] Editing data unavailable: {e}")

    if s is not None:
        # Three metric pairs: LPIPS, CLIP-Dir, PSNR (baseline vs ours)
        metrics = [
            ("LPIPS↓\n(content)", s["baseline_LPIPS"], s["ours_LPIPS"],
             f'−{abs(s["delta_LPIPS"]):.2f}', "#2E86C1", True),
            ("CLIP-Dir↑\n(edit fidelity)", s["baseline_CLIPDir"], s["ours_CLIPDir"],
             f'−{abs(s["delta_CLIPDir"]):.2f}', "#E74C3C", True),
            ("PSNR↑\n(pixel)", s["baseline_PSNR"], s["ours_PSNR"],
             f'+{s["delta_PSNR"]:.1f}', "#27AE60", False),
        ]

        x_pos = np.arange(len(metrics))
        width = 0.3
        for i, (label, base, ours, delta, color, is_lower_better) in enumerate(metrics):
            ax_c.bar(i - width / 2, base, width, color="#BDC3C7", edgecolor="white",
                     linewidth=0.5, label="Baseline" if i == 0 else "")
            ax_c.bar(i + width / 2, ours, width, color=color, edgecolor="white",
                     linewidth=0.5, label="Ours (λ=0.7)" if i == 0 else "")
            # Delta annotation
            y_annot = max(base, ours) + (0.02 if is_lower_better else 0.03)
            ax_c.annotate(delta,
                         xy=(i, max(base, ours)),
                         xytext=(i, y_annot + 0.04),
                         fontsize=9, fontweight="bold", color=color, ha="center",
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels([m[0] for m in metrics], fontsize=10)
        ax_c.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax_c.grid(axis="y", alpha=0.2)

        # Annotation explaining the trade-off
        ax_c.text(0.5, -0.22,
                  f"121 pairs, 104 images. LPIPS {s['delta_LPIPS']:.1f} (p=5e-55, d={s['LPIPS_d']:.1f})\n"
                  f"CLIP-Dir {s['delta_CLIPDir']:.1f} (p=1e-29, d={s['CLIPDir_d']:.1f}) — "
                  f"content preservation at cost of edit fidelity.",
                  transform=ax_c.transAxes, fontsize=7.5, color=C_DARK, ha="center",
                  style="italic")
    else:
        ax_c.text(0.5, 0.5, "[Editing benchmark data unavailable]", ha="center",
                  va="center", transform=ax_c.transAxes, fontsize=11, color="gray")
        ax_c.axis("off")

    fig.suptitle("Figure 5: Diagnosis-guided Correction — Simple because Diagnosis is Sufficient",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.savefig(OUT_DIR / "fig5_application.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_application.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig 5] → outputs/figures/fig5_application.pdf")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating ICLR main-paper figures (2, 3, 5)...")
    fig2_fingerprint()
    fig3_topology_mapping()
    fig5_application()
    print(f"\nDone. Output: {OUT_DIR.resolve()}\n")
    print("Figures 1 and 4 (conceptual) → draw.io")
