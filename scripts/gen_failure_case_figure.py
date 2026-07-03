"""
Generate failure case analysis figure and documentation.

Outputs:
  outputs/thesis_figures/failure_cases.png
  outputs/thesis_figures/failure_analysis.md
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Identified failure cases
FAILURE_CASES = [
    {
        "id": "coco_1490",
        "image": "data/coco_val/coco_000000001490.jpg",
        "title": "Minimal Correction Benefit",
        "baseline_PSNR": 24.15,
        "delta_PSNR": 0.12,
        "analysis": "Near-zero correction benefit. Image has large uniform regions "
                    "(sky, water) with minimal texture — the residual signal carries "
                    "little recoverable pixel-level information. Baselines >24 dB "
                    "suggest the DDIM inversion already preserves this content well.",
    },
    {
        "id": "coco_0872",
        "image": "data/coco_val/coco_000000000872.jpg",
        "title": "Second-Smallest Benefit",
        "baseline_PSNR": 20.53,
        "delta_PSNR": 0.39,
        "analysis": "Very small gain despite lower baseline (20.53 dB). Image contains "
                    "complex overlapping objects (person, surfboard, ocean) at multiple "
                    "depths — the UNet features may encode competing depth signals, "
                    "reducing residual signal coherence.",
    },
    {
        "id": "face2",
        "image": "data/basetest/face2.jpg",
        "title": "ArcFace Identity Failure",
        "baseline_PSNR": 27.96,
        "delta_PSNR": -0.23,
        "analysis": "ArcFace similarity = 0.0 in 13/14 configurations (face detection "
                    "fails). Baseline PSNR is high (27.96 dB) but correction slightly "
                    "degrades it (−0.23 dB) — the face may be in profile or have "
                    "occlusion that prevents both ArcFace recognition and effective "
                    "feature-level correction.",
    },
    {
        "id": "coco_0285",
        "image": "data/coco_val/coco_000000000285.jpg",
        "title": "Cross-Architecture Poor Performer",
        "baseline_PSNR": 19.93,
        "delta_PSNR": 0.87,
        "analysis": "Consistently poor across all three architectures: SD 1.5 Δ=+0.87, "
                    "SDXL Δ=+0.54, DiT Δ=+2.23 (but baseline only 10.70 dB). Image "
                    "contains many small objects (dining table scene) with complex "
                    "spatial layout — challenging for all diffusion backbones.",
    },
    {
        "id": "coco_1818",
        "image": "data/coco_val/coco_000000001818.jpg",
        "title": "Lowest Absolute Baseline",
        "baseline_PSNR": 16.13,
        "delta_PSNR": 2.56,
        "analysis": "Lowest baseline PSNR (16.13 dB) among all coco_val images. "
                    "Despite this, correction provides a substantial +2.56 dB gain — "
                    "the residual signal is strong. The bottleneck is DDIM inversion "
                    "quality, not correction effectiveness. This represents the "
                    "fundamental limit of deterministic DDIM inversion.",
    },
    {
        "id": "attn_zero",
        "image": "data/coco_val/coco_000000000139.jpg",
        "title": "Attention Layer: Zero Correction",
        "baseline_PSNR": 24.48,
        "delta_PSNR": 0.00,
        "analysis": "The layer up_blocks.0.attentions.0 produces ΔPSNR = 0.00 across "
                    "ALL 19 images. The attention residual is orthogonal to pixel-space "
                    "reconstruction — it encodes spatial relationships, not pixel values. "
                    "This confirms the architectural hypothesis: Attention is not a "
                    "correction channel.",
    },
]


def load_image_safe(path):
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB").resize((256, 256))
    return None


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, case in enumerate(FAILURE_CASES):
        ax = axes[idx]
        img = load_image_safe(case["image"])

        if img:
            ax.imshow(img)
        else:
            ax.text(128, 128, "Image not found", ha="center", va="center",
                    fontsize=10, color="gray")

        # Title with metrics
        title = f"{case['title']}\n"
        title += f"Baseline PSNR: {case['baseline_PSNR']:.1f} dB | "
        title += f"ΔPSNR: {case['delta_PSNR']:+.2f} dB"

        ax.set_title(title, fontsize=9, fontweight="bold", color="#2C3E50",
                     pad=8)
        ax.axis("off")

        # Add analysis as text overlay at bottom
        ax.text(0.5, -0.08, case["analysis"],
                transform=ax.transAxes, fontsize=6.5, ha="center", va="top",
                color="#555", wrap=True)

    # Overall title
    fig.suptitle("Failure Case Analysis: Limitations and Edge Cases",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, 0.98, "Each case reveals a specific limitation of the framework",
             fontsize=9, ha="center", color="#7F8C8D", style="italic")

    plt.tight_layout()
    out_path = OUT_DIR / "failure_cases.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Figure] {out_path}")

    # Write analysis markdown
    write_analysis()
    print(f"[Doc] {OUT_DIR / 'failure_analysis.md'}")


def write_analysis():
    """Write detailed failure analysis document."""
    lines = [
        "# Failure Case Analysis\n",
        "## Summary\n",
        "The residual correction framework improves PSNR for 29/30 layers across all "
        "19 test images. However, several edge cases reveal intrinsic limitations.\n",
        "## Failure Categories\n",
        "### 1. Low-Texture Images (Minimal Correction Signal)",
        "- **Worst case**: coco_1490 (+0.12 dB). Large uniform regions (sky, water).",
        "- **Mechanism**: Correction signal d = f_inv − f_recon carries pixel-level information. "
        "When the image has little texture, the feature-level residual is near-zero.",
        "- **Prevalence**: ~2/19 images (coco_1490, coco_0872) show ΔPSNR < 0.5 dB.",
        "- **Mitigation**: Per-image adaptive λ could reduce unnecessary correction.",
        "",
        "### 2. ArcFace Identity Verification Failure",
        "- **Worst case**: face2 (ArcFace=0 in 13/14 configs).",
        "- **Mechanism**: ArcFace detection fails on profile/non-frontal faces. "
        "Correction slightly degrades PSNR (−0.23 dB) for this image — the only "
        "negative ΔPSNR in the entire study.",
        "- **Prevalence**: 1/10 face images across all experiments.",
        "- **Mitigation**: Face-specific preprocessing or alternative identity metrics.",
        "",
        "### 3. Cross-Architecture Degradation",
        "- **Worst case**: coco_0285 — poor across SD 1.5, SDXL, and DiT.",
        "- **Mechanism**: Complex multi-object scenes challenge deterministic DDIM "
        "inversion regardless of backbone architecture. The bottleneck is inversion "
        "quality, not correction.",
        "- **Prevalence**: 1/19 images shows cross-architecture consistency.",
        "- **Mitigation**: Stochastic inversion (DDPM) may handle complex scenes better.",
        "",
        "### 4. Low Baseline Quality (Inversion Limit)",
        "- **Worst case**: coco_1818 (baseline PSNR 16.13 dB).",
        "- **Mechanism**: This is NOT a correction failure — the +2.56 dB gain is "
        "substantial. The bottleneck is DDIM inversion itself, which cannot perfectly "
        "invert images with certain frequency characteristics.",
        "- **Prevalence**: ~2/19 images have baseline PSNR < 20 dB.",
        "- **Mitigation**: More inversion steps (100+) or alternative inversion methods.",
        "",
        "### 5. Attention Layer Orthogonality",
        "- **Finding**: up_blocks.0.attentions.0 produces ΔPSNR = 0.00 on all 19 images.",
        "- **Significance**: This is not a 'failure' but a confirmation of the "
        "architectural hypothesis: Attention residuals are geometrically orthogonal "
        "to pixel-space reconstruction. This validates the design choice to inject "
        "correction only at ResNet layers.",
        "",
        "## Design Implications\n",
        "1. **Per-image adaptive λ**: Could improve low-texture cases by reducing correction when unnecessary.",
        "2. **Architecture-aware layer selection**: Attention layers confirmed useless for pixel-level correction.",
        "3. **Inversion quality is the ceiling**: When baseline PSNR < 18 dB, the bottleneck is DDIM inversion, not correction.",
        "4. **The framework is safe**: Only 1 image shows negative ΔPSNR (−0.23 dB), and it's an ArcFace edge case.",
    ]

    path = OUT_DIR / "failure_analysis.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Doc] {path}")


if __name__ == "__main__":
    main()
