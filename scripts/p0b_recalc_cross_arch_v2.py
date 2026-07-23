"""
Recalculate cross-architecture structural distance matrix with v2 metric.

v2: D_total = sqrt(D_pp^2 + D_mag^2)  (continuous only, no peak_count)
D_pp = |peak_position_a - peak_position_b|
D_mag = L2(concentration, spread)
D_shape NOT computed (Spearman requires same-length profiles)

Architectures: SD 1.5 (38L), SDXL (28L), HunyuanDiT (40L), FLUX (57L), SD 3.5 (24L)
"""

import json, numpy as np
from pathlib import Path
from scipy.signal import find_peaks

OUT = Path("outputs/p0b_cross_checkpoint")
OUT.mkdir(parents=True, exist_ok=True)

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x) - (n+1)*s)/(n*s)) if s > 0 else 0.0

def extract_v2(profile, layer_names=None):
    """Extract v2 continuous features from drift profile."""
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax > dmin else p.copy()
    pp = float(np.argmax(pn))/L
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    peaks, props = find_peaks(pn, prominence=0.1)
    n_peaks = len(peaks)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks)>0 else 0.0
    peak_layer = layer_names[int(np.argmax(pn))] if layer_names else f"layer_{int(np.argmax(pn))}"
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "n_peaks": n_peaks, "top_prominence": top_prom,
            "L": L, "peak_layer": peak_layer}

def distance_v2_cross(fa, fb):
    """Cross-architecture v2: D_pp + D_mag only (no Spearman)."""
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"],
                            fa["spread"]-fb["spread"]])
    d_total = float(np.linalg.norm([d_pp, d_mag]))
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_mag": d_mag}

# ---- Load per-architecture drift profiles ----
archs = {}

# SD 1.5
with open("outputs/phase1/layer_drift_summary.json") as f:
    d15 = json.load(f)
sd15_layers = []
sd15_profile = []
for ln, stats in sorted(d15["aggregated"].items(), key=lambda x: x[0]):
    sd15_layers.append(ln)
    sd15_profile.append(stats["mean"])
archs["SD1.5"] = {"profile": np.array(sd15_profile), "layers": sd15_layers, "L": len(sd15_profile)}

# SDXL
with open("outputs/sdxl_phase1/layer_drift_summary.json") as f:
    dxl = json.load(f)
sdxl_layers = []
sdxl_profile = []
for entry in sorted(dxl["full_ranking"], key=lambda x: x["layer"]):
    sdxl_layers.append(entry["layer"])
    sdxl_profile.append(entry["mean_drift"])
archs["SDXL"] = {"profile": np.array(sdxl_profile), "layers": sdxl_layers, "L": len(sdxl_profile)}

# HunyuanDiT
with open("outputs/dit_phase1/layer_drift_summary.json") as f:
    ddi = json.load(f)
dit_layers = []
dit_profile = []
for entry in sorted(ddi["full_ranking"], key=lambda x: x["layer"]):
    dit_layers.append(entry["layer"])
    dit_profile.append(entry["mean_drift"])
archs["DiT"] = {"profile": np.array(dit_profile), "layers": dit_layers, "L": len(dit_profile)}

# FLUX
with open("outputs/phase9_flux_fp16/flux_fp16_unified_format.json") as f:
    dfl = json.load(f)
flux_layers = []
flux_profile = []
for entry in sorted(dfl["layers"], key=lambda x: x["name"]):
    flux_layers.append(entry["name"])
    flux_profile.append(entry["drift"])
archs["FLUX"] = {"profile": np.array(flux_profile), "layers": flux_layers, "L": len(flux_profile)}

# SD 3.5
with open("outputs/sd35_phase1/layer_drift_summary.json") as f:
    d35 = json.load(f)
sd35_layers = []
sd35_profile = []
# SD 3.5: aggregated has per-layer mean drift, ranking is list of layer names sorted by drift
for ln in sorted(d35["aggregated"].keys(), key=lambda x: x):
    sd35_layers.append(ln)
    sd35_profile.append(d35["aggregated"][ln]["mean"])
archs["SD3.5"] = {"profile": np.array(sd35_profile), "layers": sd35_layers, "L": len(sd35_profile)}

# Extract v2 features
features = {}
for name, arch in archs.items():
    f = extract_v2(arch["profile"], arch["layers"])
    features[name] = f
    print(f"{name:6s}: L={f['L']:3d}  pp={f['peak_position']:.4f}  conc={f['concentration']:.4f}  "
          f"sp={f['spread']:.4f}  n_peaks={f['n_peaks']}  peak={f['peak_layer']}")

# Compute pairwise v2 distances
names = sorted(archs.keys())
print(f"\n{'='*75}")
print("Cross-Architecture Structural Distance — v2 Metric (continuous only)")
print(f"{'='*75}")
print(f"{'Pair':>20s}  {'D_total':>10s}  {'D_pp':>8s}  {'D_mag':>8s}  {'old_Ds':>10s}")
print("-"*65)

matrix = {}
old_refs = {
    "SD1.5-SDXL": 0.249, "SD1.5-DiT": 0.624, "SD1.5-FLUX": 0.637,
    "SD1.5-SD3.5": 0.722, "SDXL-DiT": 0.506, "SDXL-FLUX": 0.628,
    "SDXL-SD3.5": 0.803, "DiT-FLUX": 1.077, "DiT-SD3.5": 1.165,
    "FLUX-SD3.5": 0.385  # old bf16-biased value
}

for i, na in enumerate(names):
    for nb in names[i+1:]:
        dd = distance_v2_cross(features[na], features[nb])
        key = f"{na}-{nb}"
        matrix[key] = dd
        old = old_refs.get(key, float('nan'))
        print(f"{key:>20s}  {dd['D_total']:10.6f}  {dd['D_peak_pos']:8.6f}  {dd['D_mag']:8.6f}  {old:10.4f}")

# Ranking stability check
print(f"\n--- v2 vs old ranking comparison ---")
v2_pairs = sorted(matrix.items(), key=lambda x: x[1]["D_total"])
old_pairs = sorted(old_refs.items(), key=lambda x: x[1])
print("v2 ranking (smallest→largest):")
for i, (k, v) in enumerate(v2_pairs):
    print(f"  {i}: {k:>20s}  D_total={v['D_total']:.6f}")
print("old ranking:")
for i, (k, v) in enumerate(old_pairs):
    print(f"  {i}: {k:>20s}  D_s={v:.4f}")

# Summary
summary = {
    "metric": "v2 continuous (D_pp + D_mag, no peak_count, no Spearman for cross-arch)",
    "features": {n: f for n, f in features.items()},
    "pairwise": {k: v for k, v in matrix.items()},
    "noise_floor_reference": {"median": 0.0071, "p95": 0.0163},
}
with open(OUT / "cross_arch_v2_matrix.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved to {OUT}/cross_arch_v2_matrix.json")
