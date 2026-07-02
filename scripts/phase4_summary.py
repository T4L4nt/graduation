"""
Phase 4 SOTA 综合对比表

汇总所有方法的定量指标，生成对比表格和可视化网格。

方法列表:
  - DDIM (baseline, 反演-重建, 无校正)
  - NTI (Null-Text Inversion, BLIP captions, CFG=7.5, 优化每步 null-text)
  - EDICT (Exact Diffusion Inversion, p=0.93, 精确可逆)
  - Prompt-to-Prompt (交叉注意力混合, λ=0.7)
  - ControlNet (Canny 边缘条件生成, best style)
  - Ours_Corr (Phase 2 残差校正, λ=0.7, top-5 layers)
  - Ours_StylePin (Phase 3 校正+风格+钉扎)

用法:
  python scripts/phase4_summary.py
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent.parent
OUT_DIR = Path("outputs/phase4_sota")


def load_metrics(path):
    """Load metrics JSON from a results directory or file."""
    p = Path(path)
    if p.is_dir():
        candidates = list(p.glob("metrics*.json"))
        if candidates:
            p = candidates[0]
        else:
            return []
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def get_best_psnr(results, method_prefix, steps=50):
    """Get the best PSNR entry for a given method prefix at given steps."""
    candidates = [
        r for r in results
        if r.get("method", "").startswith(method_prefix)
        and r.get("steps", steps) == steps
    ]
    if not candidates:
        # Try without steps filter
        candidates = [r for r in results if r.get("method", "").startswith(method_prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("PSNR", 0))


def main():
    print("=" * 60)
    print("Phase 4: SOTA Comprehensive Comparison")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect results from all methods
    sources = {
        "DDIM": "outputs/phase2_edict/metrics.json",
        "NTI": "outputs/phase2_nti/metrics.json",
        "EDICT": "outputs/phase2_edict/metrics.json",
        "P2P": "outputs/phase4_sota/p2p/metrics.json",
        "ControlNet": "outputs/phase4_sota/controlnet/metrics.json",
        "Ours_Corr": "outputs/phase2_edict/metrics.json",
        "Ours_StylePin": "outputs/phase3_prep/metrics_full.json",
    }

    # Common images (coco_val only)
    coco_images = sorted([
        "coco_000000000139", "coco_000000000285", "coco_000000000632",
    ])

    # Build comparison table
    table = []

    for img_name in coco_images:
        row = {"Image": img_name}

        for method, path in sources.items():
            results = load_metrics(path)
            if not results:
                row[method] = {"PSNR": None, "LPIPS": None}
                continue

            if method == "DDIM":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and r.get("method") == "DDIM"
                    and r.get("steps", 50) == 50
                ]
            elif method == "NTI":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and r.get("method") == "NTI"
                    and r.get("steps", 50) == 50
                ]
            elif method == "EDICT":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and r.get("method") == "EDICT"
                    and r.get("steps", 50) == 50
                ]
            elif method == "P2P":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and r.get("method") == "P2P_attn"
                    and r.get("lam") == 0.7
                ]
            elif method == "ControlNet":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                ]
                if candidates:
                    # Pick best CLIP_style
                    best = max(candidates, key=lambda r: r.get("CLIP_style", 0))
                    candidates = [best]
            elif method == "Ours_Corr":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and r.get("method", "").startswith("DDIM+Corr")
                    and r.get("steps", 50) == 50
                ]
            elif method == "Ours_StylePin":
                candidates = [
                    r for r in results
                    if r.get("image") == img_name
                    and "pin" in r.get("lambda", "")
                ]
                # Also try alternate key
                if not candidates:
                    candidates = [
                        r for r in results
                        if r.get("image") == img_name
                        and r.get("method", "") == "style_pin"
                    ]

            if candidates:
                best = candidates[0]
                row[method] = {
                    "PSNR": best.get("PSNR"),
                    "LPIPS": best.get("LPIPS"),
                    "SSIM": best.get("SSIM"),
                }
            else:
                row[method] = {"PSNR": None, "LPIPS": None, "SSIM": None}

        table.append(row)

    # Print comparison table
    methods = ["DDIM", "EDICT", "NTI", "P2P", "ControlNet", "Ours_Corr", "Ours_StylePin"]
    print(f"\n{'='*120}")
    print("SOTA 对比表 (50 steps, coco_val 3 images)")
    print(f"{'Image':<18s}", end="")
    for m in methods:
        print(f"  {m:>16s}", end="")
    print(f"\n{'='*120}")

    for row in table:
        print(f"{row['Image']:<18s}", end="")
        for m in methods:
            v = row.get(m, {}).get("PSNR")
            if v is not None:
                print(f"  {v:>12.2f} dB", end="  ")
            else:
                print(f"  {'N/A':>12s}", end="  ")
        print()

    # Averages
    print(f"{'AVERAGE':<18s}", end="")
    for m in methods:
        vals = [r.get(m, {}).get("PSNR") for r in table if r.get(m, {}).get("PSNR") is not None]
        if vals:
            avg = np.mean(vals)
            ddim_vals = [r["DDIM"]["PSNR"] for r in table if r.get("DDIM", {}).get("PSNR") is not None]
            ddim_avg = np.mean(ddim_vals) if ddim_vals else 0
            print(f"  {avg:>8.2f} dB", end=f"  Δ{avg-ddim_avg:+5.2f}  ")
        else:
            print(f"  {'N/A':>12s}", end="  ")
    print()

    # LPIPS table
    print(f"\n{'Image':<18s}", end="")
    for m in methods:
        print(f"  {m:>14s}", end="")
    print(f"\n{'-'*110}")

    for row in table:
        print(f"{row['Image']:<18s}", end="")
        for m in methods:
            v = row.get(m, {}).get("LPIPS")
            if v is not None:
                print(f"  {v:>12.4f}", end="  ")
            else:
                print(f"  {'N/A':>12s}", end="  ")
        print()

    # Save JSON
    with open(OUT_DIR / "comparison_table.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    # Generate comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(coco_images))
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        vals = []
        for row in table:
            v = row.get(method, {}).get("PSNR")
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=method, color=colors[i])
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel("PSNR (dB)")
    ax.set_title("SOTA Comparison: Content Preservation (50 steps)")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([c.replace("coco_00000000", "") for c in coco_images], fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "comparison_psnr.png", dpi=150)
    plt.close()
    print(f"\n[Figure] {OUT_DIR / 'comparison_psnr.png'}")

    # Summary markdown table
    print(f"\n## 论文对比表 (Markdown)")
    print(f"| Method | PSNR↑ | LPIPS↓ | Training | Memory |")
    print(f"|--------|-------|--------|----------|--------|")
    # Compute overall averages
    for method in methods:
        psnr_vals = [r.get(method, {}).get("PSNR") for r in table if r.get(method, {}).get("PSNR") is not None]
        lpips_vals = [r.get(method, {}).get("LPIPS") for r in table if r.get(method, {}).get("LPIPS") is not None]
        if psnr_vals:
            avg_p = np.mean(psnr_vals)
            avg_l = np.mean(lpips_vals)
            train = {"DDIM": "None", "EDICT": "None", "NTI": "Optimization",
                     "P2P": "None", "ControlNet": "Pre-trained",
                     "Ours_Corr": "None", "Ours_StylePin": "None"}[method]
            mem = {"DDIM": "Low", "EDICT": "2x", "NTI": "Low",
                   "P2P": "~GB (attn maps)", "ControlNet": "~1.4GB (model)",
                   "Ours_Corr": "Low (~MB)", "Ours_StylePin": "Low (~MB)"}[method]
            print(f"| {method} | {avg_p:.2f} | {avg_l:.3f} | {train} | {mem} |")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
