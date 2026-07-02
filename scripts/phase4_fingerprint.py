"""
Phase 4：跨架构漂移指纹图

将 SD 1.5（UNet）、SDXL（UNet-XL）、DiT（Transformer）三种架构的
DDIM 反演-重建层漂移热力图并排展示，揭示架构特异性的漂移模式。

用法:
  python scripts/phase4_fingerprint.py
"""

import json, re, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PROJ = Path(__file__).parent.parent
OUT_DIR = Path("outputs/phase4_sota")


def _short_label(name):
    """Abbreviate layer name: 'down_blocks.2.resnets.0' → 'R0', '...attentions.1...' → 'A1'."""
    if "resnets" in name:
        idx = name.split(".resnets.")[-1]
        return f"R{idx}"
    else:
        # Attention keys end with '.transformer_blocks.0'
        attn_part = name.split(".attentions.")[-1]
        idx = attn_part.split(".")[0]
        return f"A{idx}"


def load_sd15_drift(path):
    """SD 1.5: all 38 layers in architectural order (ResNet + Attention)."""
    with open(path) as f:
        data = json.load(f)
    aggregated = data["aggregated"]

    # Complete architectural order: ResNet→Attention interleaved per diffusers UNet2DConditionModel
    layer_order = [
        # down_blocks.0: R0 A0 R1 A1
        "down_blocks.0.resnets.0", "down_blocks.0.attentions.0.transformer_blocks.0",
        "down_blocks.0.resnets.1", "down_blocks.0.attentions.1.transformer_blocks.0",
        # down_blocks.1: R0 A0 R1 A1
        "down_blocks.1.resnets.0", "down_blocks.1.attentions.0.transformer_blocks.0",
        "down_blocks.1.resnets.1", "down_blocks.1.attentions.1.transformer_blocks.0",
        # down_blocks.2: R0 A0 R1 A1
        "down_blocks.2.resnets.0", "down_blocks.2.attentions.0.transformer_blocks.0",
        "down_blocks.2.resnets.1", "down_blocks.2.attentions.1.transformer_blocks.0",
        # down_blocks.3: R0 R1 (no attention at lowest resolution)
        "down_blocks.3.resnets.0", "down_blocks.3.resnets.1",
        # mid_block: R0 A0 R1
        "mid_block.resnets.0", "mid_block.attentions.0.transformer_blocks.0",
        "mid_block.resnets.1",
        # up_blocks.0: R0 R1 R2 (no attention at 32×32)
        "up_blocks.0.resnets.0", "up_blocks.0.resnets.1", "up_blocks.0.resnets.2",
        # up_blocks.1: R0 A0 R1 A1 R2 A2
        "up_blocks.1.resnets.0", "up_blocks.1.attentions.0.transformer_blocks.0",
        "up_blocks.1.resnets.1", "up_blocks.1.attentions.1.transformer_blocks.0",
        "up_blocks.1.resnets.2", "up_blocks.1.attentions.2.transformer_blocks.0",
        # up_blocks.2: R0 A0 R1 A1 R2 A2
        "up_blocks.2.resnets.0", "up_blocks.2.attentions.0.transformer_blocks.0",
        "up_blocks.2.resnets.1", "up_blocks.2.attentions.1.transformer_blocks.0",
        "up_blocks.2.resnets.2", "up_blocks.2.attentions.2.transformer_blocks.0",
        # up_blocks.3: R0 A0 R1 A1 R2 A2
        "up_blocks.3.resnets.0", "up_blocks.3.attentions.0.transformer_blocks.0",
        "up_blocks.3.resnets.1", "up_blocks.3.attentions.1.transformer_blocks.0",
        "up_blocks.3.resnets.2", "up_blocks.3.attentions.2.transformer_blocks.0",
    ]

    drift = []
    labels = []
    for name in layer_order:
        if name in aggregated:
            drift.append(aggregated[name]["mean"])
            labels.append(_short_label(name))
        else:
            drift.append(0)
            labels.append("?")

    drift_2d = np.array(drift).reshape(-1, 1)

    sections = {
        "Encoder": (0, 14),   # down_blocks 0-3: 4+4+4+2 = 14
        "Mid": (14, 17),      # mid_block: 3
        "Decoder": (17, 38),  # up_blocks 0-3: 3+6+6+6 = 21
    }

    return drift_2d, labels, sections


def load_sdxl_drift(path):
    """SDXL: all ~28 layers in architectural order, 3 blocks (not 4)."""
    with open(path) as f:
        data = json.load(f)

    ranking = data["full_ranking"]
    all_pairs = [(r["layer"], r["mean_drift"]) for r in ranking]

    # Build architectural sort key that interleaves ResNet↔Attention
    # Diffusers order per block: R0→A0→R1→A1 (or R0→R1→R2 for
    # attention-free blocks like down_blocks[last] and up_blocks[0])
    def _arch_key(item):
        name = item[0]
        nums = [int(x) for x in re.findall(r"\d+", name)]
        if "down_blocks" in name:
            group = 0
        elif "mid_block" in name:
            group = 1
        else:
            group = 2
        sub_idx = nums[1] if len(nums) > 1 else 0
        is_resnet = 0 if "resnets" in name else 1
        position = sub_idx * 2 + is_resnet  # R0=0, A0=1, R1=2, A1=3, R2=4, A2=5
        return (group, nums[0], position)

    ordered = sorted(all_pairs, key=_arch_key)

    drift_2d = np.array([v for _, v in ordered]).reshape(-1, 1)
    labels = [_short_label(n) for n, _ in ordered]

    n_down = sum(1 for n, _ in ordered if "down_blocks" in n)
    n_mid = sum(1 for n, _ in ordered if "mid_block" in n)

    sections = {
        "Encoder": (0, n_down),
        "Mid": (n_down, n_down + n_mid),
        "Decoder": (n_down + n_mid, len(ordered)),
    }
    return drift_2d, labels, sections


def load_dit_drift(path):
    """DiT: 40 transformer blocks, blocks.0-19 bottom, blocks.20-39 top."""
    with open(path) as f:
        data = json.load(f)

    ranking = data["full_ranking"]
    # Sort by block number
    ordered = sorted(ranking, key=lambda r: int(r["layer"].split(".")[1]))

    drift_2d = np.array([r["mean_drift"] for r in ordered]).reshape(-1, 1)
    labels = [r["layer"].split(".")[1] for r in ordered]

    sections = {
        "Bottom\n(no skip)": (0, 20),
        "Top\n(with skip)": (20, 40),
    }
    return drift_2d, labels, sections


def main():
    print("[Fingerprint] Cross-architecture drift comparison")

    # Load all three
    sd15_data = load_sd15_drift("outputs/phase1/layer_drift_summary.json")
    sdxl_data = load_sdxl_drift("outputs/sdxl_phase1/layer_drift_summary.json")
    dit_data = load_dit_drift("outputs/dit_phase1/layer_drift_summary.json")

    # Normalize each to [0, 1] for visual comparison
    all_data = [sd15_data, sdxl_data, dit_data]
    titles = [
        "SD 1.5 (UNet)\n38 layers · 19 images",
        "SDXL (UNet-XL)\n28 layers · 5 images",
        "HunyuanDiT (Transformer)\n40 blocks · 5 images",
    ]

    # Per-architecture normalization — drift magnitudes differ by 1000x across architectures
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), layout='constrained')
    cmap = plt.cm.YlOrRd

    for ax, (drift_2d, labels, sections), title in zip(axes, all_data, titles):
        # Normalize this architecture's drift to its own [0, p95]
        vals = drift_2d.flatten()
        vmax = np.percentile(vals, 95)
        norm = Normalize(vmin=0, vmax=vmax)
        im = ax.imshow(drift_2d, aspect='auto', cmap=cmap, norm=norm)

        n_layers = len(labels)

        # No tick labels — keep it clean
        ax.set_yticks([])
        ax.set_xticks([])

        # Mark section boundaries and labels on the LEFT side
        section_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        for j, (sec_name, (start, end)) in enumerate(sections.items()):
            if end <= start:
                continue
            if j > 0:
                ax.axhline(y=start - 0.5, color='gray', linewidth=0.5,
                          linestyle=':', alpha=0.4)
            mid_frac = (start + end) / 2 / n_layers
            ax.text(-0.12, mid_frac, sec_name, fontsize=8,
                    color=section_colors[j % len(section_colors)],
                    ha='right', va='center', fontweight='bold',
                    transform=ax.transAxes)

        # ★ at drift peak (data coordinates: row idx maps directly)
        peak_idx = np.argmax(drift_2d.flatten())
        ax.plot(0, peak_idx, '*', color='darkred', markersize=12,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.text(0.15, peak_idx, labels[peak_idx], fontsize=7,
                color='darkred', fontweight='bold', va='center')

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Colorbar at bottom with generous spacing
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.03, pad=0.08, shrink=0.50, aspect=40)
    cbar.set_label("Normalized MSE Drift", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Cross-Architecture Drift Fingerprints\n"
                 "SD 1.5: mid + up_blocks.2 double-peak  —  SDXL: mid block dominant  —  DiT: bottom→top transition",
                 fontsize=13, fontweight='bold')

    # constrained_layout handles spacing automatically
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cross_arch_fingerprint.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Figure] {out_path}")

    # Also create compact summary
    print(f"\nArchitecture drift hotspots:")
    for name, (drift_2d, labels, sections), title in \
        zip(["SD 1.5", "SDXL", "DiT"], all_data, titles):
        vals = drift_2d.flatten()
        top_idx = np.argmax(vals)
        print(f"  {name}: peak at layer {labels[top_idx]} "
              f"(drift={vals[top_idx]:.1f})")

    print(f"\nThe three architectures exhibit fundamentally different drift patterns.")
    print(f"Without Phase 1 diagnosis, the optimal correction layers would be unknowable.")


if __name__ == "__main__":
    main()
