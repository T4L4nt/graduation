"""
Recalculate cross-architecture structural distance matrix with v2 metric.

v2: D_total = sqrt(D_pp^2 + D_mag^2)  (continuous only, no peak_count)
D_pp = |peak_position_a - peak_position_b|
D_mag = L2(concentration, spread)

Architectures: SD 1.5 (38L), SDXL (28L), H-DiT (40L), FLUX (57L), SD 3.5 (24L), PixArt-Σ (28L)

Uses scripts/layer_order.py for canonical execution-order topology per architecture.
"""

import json, numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from layer_order import (
    canonical_sort_key, peak_position, layer_hash,
    unet_topo_key, natural_key,
)

OUT = Path("outputs/p0b_cross_checkpoint")
OUT.mkdir(parents=True, exist_ok=True)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x) - (n+1)*s)/(n*s)) if s > 0 else 0.0


def extract_v2(profile, layer_names):
    """Extract v2 continuous features from drift profile (canonically ordered)."""
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax > dmin else p.copy()
    idx = int(np.argmax(pn))
    pp = idx / L
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    peaks, props = find_peaks(pn, prominence=0.1)
    n_peaks = len(peaks)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks)>0 else 0.0
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "n_peaks": n_peaks, "top_prominence": top_prom,
            "L": L, "peak_layer": layer_names[idx]}


def distance_v2_cross(fa, fb):
    """Cross-architecture v2: D_pp + D_mag only (no Spearman)."""
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"],
                            fa["spread"]-fb["spread"]])
    d_total = float(np.linalg.norm([d_pp, d_mag]))
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_mag": d_mag}


# ── Load per-architecture drift profiles in canonical order ────────────
archs = {}
hashes = {}

# SD 1.5  (UNet)
with open("outputs/phase1/layer_drift_summary.json") as f:
    d15 = json.load(f)
agg15 = d15["aggregated"]
sd15_key = unet_topo_key
sd15_layers = sorted(agg15.keys(), key=sd15_key)
sd15_profile = [agg15[ln]["mean"] for ln in sd15_layers]
archs["SD1.5"] = {"profile": np.array(sd15_profile), "layers": sd15_layers, "L": len(sd15_profile)}
hashes["SD1.5"] = layer_hash("SD1.5", list(agg15.keys()))

# SDXL  (UNet)
with open("outputs/sdxl_phase1/layer_drift_summary.json") as f:
    dxl = json.load(f)
mean_lookup = {e["layer"]: e["mean_drift"] for e in dxl["full_ranking"]}
sdxl_layers = sorted(mean_lookup.keys(), key=unet_topo_key)
sdxl_profile = [mean_lookup[ln] for ln in sdxl_layers]
archs["SDXL"] = {"profile": np.array(sdxl_profile), "layers": sdxl_layers, "L": len(sdxl_profile)}
hashes["SDXL"] = layer_hash("SDXL", list(mean_lookup.keys()))

# H-DiT  (HunyuanDiT, plain DiT blocks)
with open("outputs/dit_phase1/layer_drift_summary.json") as f:
    ddi = json.load(f)
mean_lookup_dit = {e["layer"]: e["mean_drift"] for e in ddi["full_ranking"]}
dit_key = canonical_sort_key("H-DiT")
dit_layers = sorted(mean_lookup_dit.keys(), key=dit_key)
dit_profile = [mean_lookup_dit[ln] for ln in dit_layers]
archs["H-DiT"] = {"profile": np.array(dit_profile), "layers": dit_layers, "L": len(dit_profile)}
hashes["H-DiT"] = layer_hash("H-DiT", list(mean_lookup_dit.keys()))

# FLUX  (MMDiT, 104-image protocol, natural sort over hidden_joint_N/hidden_single_N)
with open("outputs/unified_100img/FLUX_drift.json") as f:
    dfl = json.load(f)
flux_per_img = dfl["per_image"]
flux_key = canonical_sort_key("FLUX")
flux_layers = sorted(flux_per_img.keys(), key=flux_key)
flux_profile = [float(np.mean(list(flux_per_img[ln].values()))) for ln in flux_layers]
archs["FLUX"] = {"profile": np.array(flux_profile), "layers": flux_layers, "L": len(flux_profile)}
hashes["FLUX"] = layer_hash("FLUX", list(flux_per_img.keys()))

# SD 3.5  (MMDiT, 104-image protocol)
with open("outputs/sd35_phase1/layer_drift_summary_100img.json") as f:
    d35 = json.load(f)
agg35 = d35["aggregated"]
sd35_key = canonical_sort_key("SD3.5")
sd35_layers = sorted(agg35.keys(), key=sd35_key)
sd35_profile = [agg35[ln]["mean"] for ln in sd35_layers]
archs["SD3.5"] = {"profile": np.array(sd35_profile), "layers": sd35_layers, "L": len(sd35_profile)}
hashes["SD3.5"] = layer_hash("SD3.5", list(agg35.keys()))

# PixArt-Σ  — pre-computed features (raw profile from earlier measurement)
with open("outputs/p0b_cross_checkpoint/pixart_fingerprint.json") as f:
    dpx = json.load(f)
pf = dpx["features"]
n_blocks = dpx["n_blocks"]
pp_float = pf["peak_position"]  # e.g. 20/28 ≈ 0.7142857
pp_idx = int(round(pp_float * n_blocks))  # fix floating-point truncation
archs["PixArt-Sigma"] = {
    "profile": np.zeros(n_blocks),
    "layers": [f"block_{i}" for i in range(n_blocks)],
    "L": n_blocks,
    "_precomputed": {
        "peak_position": pp_float,
        "concentration": pf["concentration"],
        "spread": pf["spread"],
        "n_peaks": int(pf["n_peaks"]),
        "top_prominence": None,
        "L": n_blocks,
        "peak_layer": f"block_{pp_idx}"
    }
}
hashes["PixArt-Sigma"] = "precomputed"


# ── Extract features ────────────────────────────────────────────────────
features = {}
for name, arch in archs.items():
    if "_precomputed" in arch:
        f = arch["_precomputed"]
        features[name] = f
    else:
        f = extract_v2(arch["profile"], arch["layers"])
        features[name] = f
    print(f"{name:12s}: L={f['L']:3d}  pp={f['peak_position']:.4f}  conc={f['concentration']:.4f}  "
          f"sp={f['spread']:.4f}  n_peaks={f['n_peaks']}  peak={f['peak_layer']}")


# ── Pairwise v2 distances ───────────────────────────────────────────────
names = sorted(archs.keys())
print(f"\n{'='*85}")
print("Cross-Architecture Structural Distance — v2 Metric (canonical topo order)")
print(f"{'='*85}")
print(f"{'Pair':>24s}  {'D_total':>10s}  {'D_pp':>10s}  {'D_mag':>10s}")
print("-"*70)

matrix = {}

for i, na in enumerate(names):
    for nb in names[i+1:]:
        dd = distance_v2_cross(features[na], features[nb])
        key = f"{na}-{nb}"
        matrix[key] = dd
        print(f"{key:>24s}  {dd['D_total']:10.6f}  {dd['D_peak_pos']:10.6f}  {dd['D_mag']:10.6f}")

# Ranking
print(f"\n--- Ranking (smallest → largest D_total) ---")
v2_pairs = sorted(matrix.items(), key=lambda x: x[1]["D_total"])
for i, (k, v) in enumerate(v2_pairs):
    print(f"  {i:2d}: {k:>24s}  D_total={v['D_total']:.6f}  D_pp={v['D_peak_pos']:.6f}  D_mag={v['D_mag']:.6f}")

# Noise floor notes
# Bootstrap noise floor (B=100 intra-model): median=0.0071, p95=0.0163
# 1/L granularity for these architectures: 1/24≈0.042 to 1/57≈0.018
l_min = min(f["L"] for f in features.values())
l_max = max(f["L"] for f in features.values())
print(f"\nQuantisation granularity 1/L: {1/l_max:.3f} – {1/l_min:.3f}  (L range {l_min}–{l_max})")
print(f"Bootstrap noise floor: median 0.0071, p95 0.0163 (B=100 intra-model)")

# ── Protocol metadata (per-architecture) ─────────────────────────────────
protocol = {
    "SD1.5":        {"n_images": 19, "steps": 50, "sampler": "DDIM", "source": "outputs/phase1/layer_drift_summary.json"},
    "SDXL":         {"n_images": 19, "steps": 50, "sampler": "DDIM", "source": "outputs/sdxl_phase1/layer_drift_summary.json"},
    "H-DiT":        {"n_images": 19, "steps": 50, "sampler": "DDIM v_pred", "source": "outputs/dit_phase1/layer_drift_summary.json"},
    "FLUX":         {"n_images": 104,"steps": 50, "sampler": "Euler", "source": "outputs/unified_100img/FLUX_drift.json (natural sort)"},
    "SD3.5":        {"n_images": 104,"steps": 50, "sampler": "Euler", "source": "outputs/sd35_phase1/layer_drift_summary_100img.json"},
    "PixArt-Sigma": {"n_images": 19, "steps": 50, "sampler": "DDIM", "source": "precomputed (pixart_fingerprint.json)"},
}
protocol_warning = (
    "SD1.5/SDXL/H-DiT/PixArt-Sigma use 19-image (coco_val); "
    "FLUX & SD3.5 use 104-image (coco_val100). "
    "Unify to 104-image across all architectures before final submission."
)

# ── Summary ─────────────────────────────────────────────────────────────
summary = {
    "metric": "v2 continuous (D_pp + D_mag, canonical topo order per architecture)",
    "protocol": protocol,
    "protocol_warning": protocol_warning,
    "canonical_layer_hashes": hashes,
    "reference_L_range": {"min": l_min, "max": l_max},
    "features": {n: f for n, f in features.items()},
    "pairwise": {k: v for k, v in matrix.items()},
}
with open(OUT / "cross_arch_v2_matrix.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {OUT}/cross_arch_v2_matrix.json")
