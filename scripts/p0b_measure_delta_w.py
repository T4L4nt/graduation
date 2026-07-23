"""
P0b: Measure SD 1.4 ↔ SD 1.5 per-layer weight difference.

Computes ‖ΔW_l‖ / ‖W_l‖ for each weight matrix in the UNet,
reports distribution (mean, median, max, per-layer-type breakdown).
Marks the measured ΔW on the C1 dose-response axis to close the loop.

Usage:
  python -u scripts/p0b_measure_delta_w.py
"""

import json, sys
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEVICE = "cuda"; DTYPE = torch.float16

from diffusers import StableDiffusionPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_14 = "CompVis/stable-diffusion-v1-4"
MODEL_15 = "runwayml/stable-diffusion-v1-5"

# Fallback: load from local cache only (no network)
LOAD_KWARGS = {"local_files_only": True, "torch_dtype": torch.float16}


def is_weight(name, param):
    if "bias" in name: return False
    if "norm" in name.lower(): return False
    if param.ndim < 2: return False
    return True


def main():
    print("Loading SD 1.4...")
    p14 = StableDiffusionPipeline.from_pretrained(MODEL_14, **LOAD_KWARGS).to(DEVICE)
    sd14 = {n: p.detach().cpu().float().clone() for n, p in p14.unet.named_parameters() if is_weight(n, p)}
    del p14; torch.cuda.empty_cache()

    print("Loading SD 1.5...")
    p15 = StableDiffusionPipeline.from_pretrained(MODEL_15, **LOAD_KWARGS).to(DEVICE)
    sd15 = {n: p.detach().cpu().float().clone() for n, p in p15.unet.named_parameters() if is_weight(n, p)}
    del p15; torch.cuda.empty_cache()

    common = set(sd14.keys()) & set(sd15.keys())
    print(f"Common weight matrices: {len(common)}")

    per_layer = []
    by_block_type = {}

    for name in sorted(common):
        w14 = sd14[name]; w15 = sd15[name]
        dw = (w15 - w14).view(-1)
        w14_flat = w14.view(-1)
        wn = w14_flat.norm().item()
        dwn = dw.norm().item()
        rel_dw = dwn / max(wn, 1e-8)

        # Determine block type
        if "down_blocks" in name: bt = "down_blocks"
        elif "up_blocks" in name: bt = "up_blocks"
        elif "mid_block" in name: bt = "mid_block"
        else: bt = "other"

        if "attentions" in name.lower() or "transformer" in name.lower():
            bt += "/attention"
        elif "resnets" in name:
            bt += "/resnet"
        else:
            bt += "/other"

        entry = {"name": name, "rel_dw": float(rel_dw), "block_type": bt,
                 "w_norm": float(wn), "dw_norm": float(dwn)}
        per_layer.append(entry)

        if bt not in by_block_type: by_block_type[bt] = []
        by_block_type[bt].append(float(rel_dw))

    rel_dws = [e["rel_dw"] for e in per_layer]
    rel_dws = np.array(rel_dws)

    print(f"\n{'='*60}")
    print("SD 1.4 ↔ SD 1.5 Per-Layer Relative Weight Difference")
    print(f"{'='*60}")
    print(f"  N common layers: {len(common)}")
    print(f"  ‖ΔW‖/‖W‖ (per weight matrix):")
    print(f"    mean   = {np.mean(rel_dws):.6f}")
    print(f"    median = {np.median(rel_dws):.6f}")
    print(f"    p50    = {np.percentile(rel_dws, 50):.6f}")
    print(f"    p90    = {np.percentile(rel_dws, 90):.6f}")
    print(f"    p95    = {np.percentile(rel_dws, 95):.6f}")
    print(f"    p99    = {np.percentile(rel_dws, 99):.6f}")
    print(f"    max    = {np.max(rel_dws):.6f}")
    print(f"    min    = {np.min(rel_dws):.6f}")

    print(f"\n  By block type:")
    for bt in sorted(by_block_type.keys()):
        vals = by_block_type[bt]
        print(f"    {bt:20s}: n={len(vals):3d}  mean={np.mean(vals):.6f}  p95={np.percentile(vals,95):.6f}  max={np.max(vals):.6f}")

    # Top layers by ΔW
    print(f"\n  Top-10 layers by relative ΔW:")
    top10 = sorted(per_layer, key=lambda x: x["rel_dw"], reverse=True)[:10]
    for e in top10:
        print(f"    {e['rel_dw']:.6f}  {e['name']}")

    # C1 closure check
    print(f"\n{'='*60}")
    print("C1 Closure Check")
    print(f"{'='*60}")
    # Where does the measured ΔW fall on the C1 dose-response axis?
    # From v4: noise floor p95 ≈ 0.016, stable regime ε ≤ ? (to be confirmed)
    print(f"  Measured ΔW (median): {np.median(rel_dws):.6f}")
    print(f"  Measured ΔW (p95):    {np.percentile(rel_dws, 95):.6f}")
    print(f"  Measured ΔW (max):    {np.max(rel_dws):.6f}")
    print(f"  C1 stable threshold (to be updated from v4 results)")

    # Save
    summary = {
        "models": {"sd14": MODEL_14, "sd15": MODEL_15},
        "n_common_layers": len(common),
        "statistics": {
            "mean": float(np.mean(rel_dws)), "median": float(np.median(rel_dws)),
            "p90": float(np.percentile(rel_dws, 90)), "p95": float(np.percentile(rel_dws, 95)),
            "p99": float(np.percentile(rel_dws, 99)), "max": float(np.max(rel_dws)),
            "min": float(np.min(rel_dws)),
        },
        "by_block_type": {bt: {"n": len(v), "mean": float(np.mean(v)),
                                "p95": float(np.percentile(v, 95)),
                                "max": float(np.max(v))}
                          for bt, v in by_block_type.items()},
        "top10": [{"name": e["name"], "rel_dw": e["rel_dw"]} for e in top10],
        "per_layer": per_layer,
    }

    sp = OUT_DIR / "sd14_sd15_delta_w.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {sp}")


if __name__ == "__main__":
    main()
