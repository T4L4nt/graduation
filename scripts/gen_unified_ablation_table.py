"""
Generate unified ablation summary table.

Outputs: outputs/thesis_figures/unified_ablation_table.tex + .md
"""

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def get_row_from_coco_val(data):
    """Extract aggregate from coco_val_summary.json."""
    return {
        "PSNR": data.get("avg_baseline_PSNR"),
        "LPIPS": data.get("avg_baseline_LPIPS"),
        "ΔPSNR": None,
    }


def get_row_ours(data):
    return {
        "PSNR": data.get("avg_ours_PSNR"),
        "LPIPS": data.get("avg_ours_LPIPS"),
        "ΔPSNR": data.get("avg_delta_PSNR"),
    }


def get_ablation_rows(metrics_list, layer_key="layers", method_key="method"):
    """Aggregate ablation metrics by config."""
    groups = {}
    for entry in metrics_list:
        config = entry.get(layer_key) or entry.get(method_key) or entry.get("lambda", "?")
        if isinstance(config, list):
            config = "top5"  # normalize

        if config not in groups:
            groups[config] = {"PSNR": [], "LPIPS": [], "ΔPSNR": []}

        if "PSNR" in entry:
            groups[config]["PSNR"].append(entry["PSNR"])
        if "LPIPS" in entry:
            groups[config]["LPIPS"].append(entry["LPIPS"])
        if "delta_psnr" in entry:
            groups[config]["ΔPSNR"].append(entry["delta_psnr"])
        elif "ΔPSNR" in entry:
            groups[config]["ΔPSNR"].append(entry["ΔPSNR"])

    rows = {}
    for config, vals in groups.items():
        psnr_vals = [v for v in vals["PSNR"] if v is not None]
        lpips_vals = [v for v in vals["LPIPS"] if v is not None]
        delta_vals = [v for v in vals["ΔPSNR"] if v is not None]

        rows[config] = {
            "PSNR": np.mean(psnr_vals) if psnr_vals else None,
            "LPIPS": np.mean(lpips_vals) if lpips_vals else None,
            "ΔPSNR": np.mean(delta_vals) if delta_vals else None,
        }
    return rows


def get_style_rows(metrics_list):
    """Extract style-related rows from phase3 compare data."""
    groups = {}
    for entry in metrics_list:
        config = entry.get("lambda", entry.get("method", "?"))

        # Normalize config names
        if config == "baseline":
            config = "DDIM Baseline"
        elif "corr_only" in str(config):
            config = "+ Correction"
        elif "style_only" in str(config):
            config = "Style Only (no correction)"
        elif "corr+style" in str(config) and "pin" not in str(config):
            config = "+ Correction + Style"
        elif "pin" in str(config):
            config = "Full Framework (+ pinning)"
        else:
            continue

        if config not in groups:
            groups[config] = {"PSNR": [], "LPIPS": [], "ΔPSNR": []}

        if "PSNR" in entry and entry["PSNR"] is not None:
            groups[config]["PSNR"].append(entry["PSNR"])
        if "LPIPS" in entry and entry["LPIPS"] is not None:
            groups[config]["LPIPS"].append(entry["LPIPS"])

    rows = {}
    baseline_psnr = None
    for config in ["DDIM Baseline"]:
        if config in groups:
            vals = groups[config]["PSNR"]
            if vals:
                baseline_psnr = np.mean(vals)
                rows[config] = {
                    "PSNR": baseline_psnr,
                    "LPIPS": np.mean(groups[config]["LPIPS"]) if groups[config]["LPIPS"] else None,
                    "ΔPSNR": None,
                }

    for config, vals in groups.items():
        if config == "DDIM Baseline":
            continue
        psnr_mean = np.mean(vals["PSNR"]) if vals["PSNR"] else None
        rows[config] = {
            "PSNR": psnr_mean,
            "LPIPS": np.mean(vals["LPIPS"]) if vals["LPIPS"] else None,
            "ΔPSNR": psnr_mean - baseline_psnr if (psnr_mean and baseline_psnr) else None,
        }

    return rows


def build_table():
    """Build unified ablation table."""
    rows = []

    # 1. DDIM Baseline from coco_val summary
    coco = load_json("outputs/thesis_figures/coco_val_summary.json")
    if coco:
        rows.append({
            "Component": "DDIM Baseline",
            "PSNR": coco.get("avg_baseline_PSNR"),
            "LPIPS": coco.get("avg_baseline_LPIPS"),
            "ΔPSNR": None,
            "Source": "coco_val (19 imgs)",
        })
        rows.append({
            "Component": "+ Residual Correction (top5)",
            "PSNR": coco.get("avg_ours_PSNR"),
            "LPIPS": coco.get("avg_ours_LPIPS"),
            "ΔPSNR": coco.get("avg_delta_PSNR"),
            "Source": "coco_val (19 imgs)",
        })

    # 2. Layer ablation from phase2_full
    ablation = load_json("outputs/phase2_full/ablation/metrics.json")
    if ablation:
        ab_rows = get_ablation_rows(ablation)
        mapping = {
            "random5": ("+ Correction (random5)", "val (5 imgs)"),
            "encoder5": ("+ Correction (encoder5)", "val (5 imgs)"),
            "attention5": ("+ Correction (attention5)", "val (5 imgs)"),
            "latent_interp": ("+ Latent Interpolation", "val (5 imgs)"),
        }
        for key, (label, source) in mapping.items():
            if key in ab_rows and ab_rows[key]["PSNR"]:
                rows.append({
                    "Component": label,
                    "PSNR": ab_rows[key]["PSNR"],
                    "LPIPS": ab_rows[key]["LPIPS"],
                    "ΔPSNR": ab_rows[key].get("ΔPSNR"),
                    "Source": source,
                })

    # 3. Style/pinning from phase3 compare
    compare = load_json("outputs/phase3_prep/compare/metrics.json")
    if compare:
        style_rows = get_style_rows(compare)
        for config in ["Style Only (no correction)",
                       "+ Correction + Style",
                       "Full Framework (+ pinning)"]:
            if config in style_rows:
                rows.append({
                    "Component": config,
                    "PSNR": style_rows[config]["PSNR"],
                    "LPIPS": style_rows[config]["LPIPS"],
                    "ΔPSNR": style_rows[config].get("ΔPSNR"),
                    "Source": "val (5 imgs)",
                })

    # 4. Add ResNet vs Attention comparison row
    info_path = Path("outputs/phase4_info_theory/per_layer_correction.json")
    if info_path.exists():
        with open(info_path) as f:
            info_data = json.load(f)
        results = info_data.get("results", {})
        resnet_d = [r["mean_delta_psnr"] for r in results.values()
                    if r["mean_delta_psnr"] is not None and "attentions" not in str(r)]
        attn_d = [r["mean_delta_psnr"] for r in results.values()
                  if r["mean_delta_psnr"] is not None and "attentions" in str(r)]
        if resnet_d:
            rows.append({
                "Component": "  └ ResNet (per-layer avg)",
                "PSNR": None,
                "LPIPS": None,
                "ΔPSNR": np.mean(resnet_d),
                "Source": "coco_val (19 imgs)",
            })
        if attn_d:
            rows.append({
                "Component": "  └ Attention (per-layer avg)",
                "PSNR": None,
                "LPIPS": None,
                "ΔPSNR": np.mean(attn_d),
                "Source": "coco_val (19 imgs)",
            })

    return rows


def fmt(val, decimals=2, suffix=""):
    if val is None:
        return "—"
    return f"{val:+.{decimals}f}{suffix}" if val != abs(val) or suffix else f"{val:.{decimals}f}{suffix}"


def write_markdown(rows):
    path = OUT_DIR / "unified_ablation_table.md"
    with open(path, "w") as f:
        f.write("# Unified Ablation Summary\n\n")
        f.write("| Component | PSNR (dB) | LPIPS | ΔPSNR (dB) | Source |\n")
        f.write("|-----------|-----------|-------|-------------|--------|\n")
        for r in rows:
            f.write(f"| {r['Component']} | {fmt(r['PSNR'], 2)} | "
                    f"{fmt(r['LPIPS'], 3)} | {fmt(r['ΔPSNR'], 2)} | {r['Source']} |\n")
    print(f"[Done] {path}")


def write_latex(rows):
    path = OUT_DIR / "unified_ablation_table.tex"
    with open(path, "w") as f:
        f.write("% Unified Ablation Summary Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Unified ablation summary. "
                "Each row adds or removes a component from the framework.}\n")
        f.write("  \\label{tab:unified_ablation}\n")
        f.write("  \\begin{tabular}{lccc}\n")
        f.write("    \\toprule\n")
        f.write("    Component & PSNR (dB) & LPIPS & $\\Delta$PSNR \\\\\n")
        f.write("    \\midrule\n")
        for r in rows:
            f.write(f"    {r['Component']} & {fmt(r['PSNR'], 2)} & "
                    f"{fmt(r['LPIPS'], 3)} & {fmt(r['ΔPSNR'], 2)} \\\\\n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[Done] {path}")


def main():
    print("Building unified ablation table...")
    rows = build_table()

    print(f"\n  Found {len(rows)} rows\n")
    write_markdown(rows)
    write_latex(rows)

    # Print summary
    print("\n  Table Preview:")
    for r in rows:
        print(f"  {r['Component']:<40s} PSNR={fmt(r['PSNR'],2):>6s}  "
              f"LPIPS={fmt(r['LPIPS'],3):>6s}  ΔPSNR={fmt(r['ΔPSNR'],2):>6s}")


if __name__ == "__main__":
    main()
