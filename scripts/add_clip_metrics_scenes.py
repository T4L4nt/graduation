"""
补全场景实验的 CLIP 风格量化指标。

对已有重建图跑 CLIP 编码，计算 CLIP_style / CLIP_content，
更新 cross_scene_summary.json。

用法:
  python scripts/add_clip_metrics_scenes.py
"""

import json, os, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase3_prep import CLIPFeatureExtractor
from phase2_common import DEVICE

SCENES_DIR = Path("outputs/phase4_sota/scenes")
SCENE_STYLES = {
    "portraits": "a portrait with soft cinematic lighting",
    "architecture": "an architectural photograph with clean geometric lines",
    "typography": "a typographic design with bold artistic lettering",
}
OUT_DIR = Path("outputs/phase4_sota/scenes")


def compute_clip_for_scene(scene_name, extractor):
    """Compute CLIP metrics for all reconstructions in a scene directory."""
    scene_dir = SCENES_DIR / scene_name / "recons"
    if not scene_dir.exists():
        print(f"  [SKIP] No recons dir: {scene_dir}")
        return []

    style_text = SCENE_STYLES[scene_name]
    v_style = extractor.encode_text(style_text)
    v_content_ref = extractor.encode_text("a photo")

    results = []
    png_files = sorted(scene_dir.glob("*.png"))

    for png_path in png_files:
        fname = png_path.stem  # e.g. "pexels_1234_s50_baseline"
        parts = fname.split("_s")
        if len(parts) < 2:
            continue
        img_name = parts[0]
        method_tag = parts[1]  # "50_baseline", "50_corrected", "50_style_pin"

        # Load image and encode
        try:
            img = Image.open(png_path).convert("RGB")
            v_img = extractor.encode_image(img)

            # CLIP_content: cos with "a photo" direction
            clip_content = float((v_img * v_content_ref).sum())

            # CLIP_style: cos with scene style direction
            clip_style = float((v_img * v_style).sum())

            results.append({
                "image": img_name,
                "method": method_tag,
                "CLIP_style": clip_style,
                "CLIP_content": clip_content,
            })
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")

    return results


def main():
    print("Loading CLIP ViT-L/14...")
    extractor = CLIPFeatureExtractor(device=DEVICE)

    all_results = {}
    summary = {}

    for scene_name in ["portraits", "architecture", "typography"]:
        print(f"\n{'='*50}")
        print(f"Scene: {scene_name}")
        results = compute_clip_for_scene(scene_name, extractor)
        print(f"  Computed {len(results)} CLIP metrics")

        # Save per-scene CLIP metrics
        out_path = OUT_DIR / scene_name / "clip_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [Saved] {out_path}")

        all_results[scene_name] = results

        # Aggregate by method
        methods = {}
        for r in results:
            m = r["method"]
            if m not in methods:
                methods[m] = {"CLIP_style": [], "CLIP_content": []}
            methods[m]["CLIP_style"].append(r["CLIP_style"])
            methods[m]["CLIP_content"].append(r["CLIP_content"])

        summary[scene_name] = {
            "n_images": len(set(r["image"] for r in results)),
            "methods": {}
        }
        for m, vals in methods.items():
            summary[scene_name]["methods"][m] = {
                "CLIP_style_mean": float(np.mean(vals["CLIP_style"])),
                "CLIP_style_std": float(np.std(vals["CLIP_style"])),
                "CLIP_content_mean": float(np.mean(vals["CLIP_content"])),
                "CLIP_content_std": float(np.std(vals["CLIP_content"])),
            }

    # Save summary
    summary_path = OUT_DIR / "clip_metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Summary] {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("CLIP METRICS SUMMARY")
    print(f"{'='*60}")
    for scene_name, data in summary.items():
        print(f"\n  {scene_name} ({data['n_images']} images):")
        for method, metrics in data["methods"].items():
            print(f"    {method:<25s} CLIP_style={metrics['CLIP_style_mean']:.4f}±{metrics['CLIP_style_std']:.4f}  "
                  f"CLIP_content={metrics['CLIP_content_mean']:.4f}±{metrics['CLIP_content_std']:.4f}")


if __name__ == "__main__":
    main()
