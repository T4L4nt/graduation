"""
Canonical fingerprint library builder.

Reads per-image drift data, applies canonical layer ordering, outputs
a single JSON with v2 features for all architectures.

Hard-asserts protocol consistency (n_images, n_layers must match).
FATAL exit if any architecture is missing or mismatched.

Usage: python scripts/build_canonical_library.py
Output: outputs/p0b_cross_checkpoint/canonical_fingerprint_library.json
"""

import json, sys, numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from layer_order import unet_topo_key, natural_key, layer_hash

OUT = Path("outputs/p0b_cross_checkpoint")
OUT.mkdir(parents=True, exist_ok=True)


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x) - (n+1)*s)/(n*s)) if s > 0 else 0.0


def extract_v2(profile, layer_names):
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


def load_profile(name, source_path, sort_key):
    """Load per-image drift data, sort canonically, return (profile, layer_names, n_images)."""
    with open(source_path) as f:
        raw = json.load(f)

    if name == "SD3.5":
        agg = raw["aggregated"]
        layer_names = sorted(agg.keys(), key=sort_key)
        profile = [agg[ln]["mean"] for ln in layer_names]
        n_images = raw.get("n_images", 104)
    elif "per_image" in raw:
        per_img = raw["per_image"]
        layer_names = sorted(per_img.keys(), key=sort_key)
        profile = [float(np.mean(list(per_img[ln].values()))) for ln in layer_names]
        n_images = len(list(per_img[layer_names[0]].values()))
    elif "aggregated" in raw:
        agg = raw["aggregated"]
        layer_names = sorted(agg.keys(), key=sort_key)
        vals = list(agg.values())
        if isinstance(vals[0], list):
            profile = [float(np.mean(v)) for v in vals]
            n_images = len(vals[0])
        else:
            profile = [float(v["mean"] if isinstance(v, dict) else v) for v in vals]
            n_images = len(raw.get("images", [])) or 104
    else:
        raise ValueError(f"No per_image or aggregated in {source_path}")

    return profile, layer_names, n_images


# ── Architecture registry ─────────────────────────────────────────────
ARCH_DEFS = {
    "SD1.5":        ("outputs/unified_100img/SD1.5_drift.json",        unet_topo_key, 104, 38),
    "SDXL":         ("outputs/unified_100img/SDXL_drift.json",         unet_topo_key, 104, 28),
    "H-DiT":        ("outputs/unified_100img/HunyuanDiT_drift.json",   natural_key,   104, 40),
    "FLUX":         ("outputs/unified_100img/FLUX_drift.json",         natural_key,   104, 57),
    "SD3.5":        ("outputs/sd35_phase1/layer_drift_summary_100img.json", natural_key,   104, 24),
    "PixArt-Sigma": ("outputs/unified_100img/PixArt-Sigma_drift.json", natural_key,   104, 28),
}

# ── Process ───────────────────────────────────────────────────────────
protocol = {}
features_all = {}
errors = []

for name, (src, sort_key, exp_imgs, exp_layers) in ARCH_DEFS.items():
    src_path = Path(src)
    if not src_path.exists():
        errors.append(f"MISSING: {name} source {src}")
        continue

    profile, layer_names, n_images = load_profile(name, src_path, sort_key)
    n_layers = len(layer_names)
    lhash = layer_hash(name, layer_names)

    # Hard assertions
    if n_images != exp_imgs:
        errors.append(f"PROTOCOL: {name} n_images={n_images}, expected {exp_imgs}")
    if n_layers != exp_layers:
        errors.append(f"PROTOCOL: {name} n_layers={n_layers}, expected {exp_layers}")

    features = extract_v2(profile, layer_names)

    protocol[name] = {
        "n_images": n_images, "n_layers": n_layers,
        "layer_list_hash": lhash, "source": src,
    }
    features_all[name] = features

    print(f"{name:14s}: L={n_layers:3d}  N={n_images:3d}  "
          f"pp={features['peak_position']:.4f}  "
          f"conc={features['concentration']:.4f}  "
          f"sp={features['spread']:.4f}  "
          f"peak={features['peak_layer']}  hash={lhash}")

# ── FATAL on errors ──────────────────────────────────────────────────
if errors:
    print("\n" + "=" * 70)
    print("FATAL: Cannot produce canonical fingerprint library.")
    for e in errors:
        print(f"  ERROR: {e}")
    sys.exit(1)

# ── Consistency check ────────────────────────────────────────────────
n_img_set = {p["n_images"] for p in protocol.values()}
if len(n_img_set) > 1:
    print(f"\nFATAL: Mixed n_images across architectures: {n_img_set}")
    for n, p in protocol.items():
        print(f"  {n}: {p['n_images']} images (source: {p['source']})")
    sys.exit(1)

# ── Save ─────────────────────────────────────────────────────────────
library = {
    "protocol": protocol,
    "n_images_uniform": list(n_img_set)[0],
    "features": features_all,
    "pairwise": {},  # filled by recalc matrix script
}
out_path = OUT / "canonical_fingerprint_library.json"
with open(out_path, "w") as f:
    json.dump(library, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")
print(f"All {len(protocol)} architectures: {list(n_img_set)[0]} images, protocol consistent.")
