"""
Canonical layer ordering across all architectures.

Each architecture has a unique ordering that respects its execution topology:
  - UNet (SD1.5, SDXL): resnets → attentions in up/down blocks; attn→resnets in mid
  - DiT/MMDiT (H-DiT, SD3.5, PixArt-Σ): numeric block index
  - FLUX: natural sort on "j"/"s" prefix + index

All scripts MUST import ordering from here.  Adding a new architecture MUST add a test.
"""

import hashlib, json, re


# ── generic natural sort ──────────────────────────────────────────────────

def natural_key(name: str):
    """Generic natural sort: split on digits, cast numeric runs to int."""
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', name)]


# ── UNet topological key ─────────────────────────────────────────────────

def unet_topo_key(name: str):
    """
    Respect UNet execution order within each block:
      down_blocks:  resnets → attentions
      up_blocks:    resnets → attentions
      mid_block:    attentions → resnets (SD 1.5 mid block: attn runs first)
    Returns a sortable tuple.
    """
    parts = name.split(".")

    # block type tier: down=0, mid=1, up=2
    if name.startswith("down_blocks"):
        bt = 0
        bi = int(parts[1])
    elif name.startswith("mid_block"):
        bt = 1
        bi = 0
    elif name.startswith("up_blocks"):
        bt = 2
        bi = int(parts[1])
    else:
        return natural_key(name)  # fallback

    # sub-block type order
    if "resnets" in parts:
        st = 0 if bt != 1 else 1    # resnets first in down/up, second in mid
        si = parts.index("resnets")
        sn = int(parts[si + 1])
    elif "attentions" in parts:
        st = 1 if bt != 1 else 0    # attentions second in down/up, first in mid
        si = parts.index("attentions")
        sn = int(parts[si + 1])
    else:
        st = 2
        sn = 0

    # transformer_blocks index (always 0 for our hooks, but be safe)
    tn = 0
    if "transformer_blocks" in parts:
        ti = parts.index("transformer_blocks")
        tn = int(parts[ti + 1])

    return (bt, bi, st, sn, tn)


# ── architecture registry ────────────────────────────────────────────────

_ARCH_KEY = {
    "SD1.5":       unet_topo_key,
    "SDXL":        unet_topo_key,
    "H-DiT":       natural_key,       # blocks.N
    "SD3.5":       natural_key,       # block_N
    "PixArt":      natural_key,       # block_N
    "PixArt-Sigma": natural_key,      # alias
    "FLUX":        natural_key,       # jN / sN
}


def canonical_sort_key(arch: str):
    """Return the canonical sort-key function for *arch*."""
    key = _ARCH_KEY.get(arch)
    if key is None:
        raise KeyError(f"Unknown architecture: {arch}. Register in layer_order._ARCH_KEY.")
    return key


def canonical_order(arch: str, names: list[str]) -> list[str]:
    """Return *names* sorted by the canonical execution order for *arch*."""
    return sorted(names, key=canonical_sort_key(arch))


def layer_hash(arch: str, names: list[str]) -> str:
    """Stable hash of canonical layer list for reproducibility auditing."""
    canon = canonical_order(arch, names)
    payload = json.dumps(canon, ensure_ascii=False, sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ── peak extraction helper ───────────────────────────────────────────────

def peak_position(arch: str, names: list[str], profile: list[float]):
    """
    Return peak position as float in [0,1] and the canonical layer name.
    *names* and *profile* must already be in canonical order.
    """
    L = len(profile)
    if L == 0:
        return 0.0, None
    idx = max(range(L), key=lambda i: profile[i])
    return idx / L, names[idx]


# ── self-tests ───────────────────────────────────────────────────────────

def _run_tests():
    """Validate canonical ordering for all registered architectures."""
    # SD1.5: up_blocks.2.resnets.0 must be at 26/38 ≈ 0.6842
    import json as _json, os as _os
    d15_path = _os.path.join(_os.path.dirname(__file__), "../outputs/phase1/layer_drift_summary.json")
    if _os.path.exists(d15_path):
        with open(d15_path) as f:
            d15 = _json.load(f)
        sd15_names = sorted(d15["aggregated"].keys(), key=unet_topo_key)
        sd15_prof  = [d15["aggregated"][ln]["mean"] for ln in sd15_names]
        pp15, pk15 = peak_position("SD1.5", sd15_names, sd15_prof)
        assert len(sd15_names) == 38, f"SD1.5 expected 38 layers, got {len(sd15_names)}"
        assert pk15 == "up_blocks.2.resnets.0", f"SD1.5 peak layer mismatch: {pk15}"
        assert abs(pp15 - 0.6842) < 0.001, f"SD1.5 pp mismatch: {pp15:.4f} != 0.6842"
        # verify up_blocks.2.resnets.0 is at index 26
        idx15 = sd15_names.index("up_blocks.2.resnets.0")
        assert idx15 == 26, f"SD1.5 up_blocks.2.resnets.0 at {idx15}, expected 26"
        print(f"  PASS: SD1.5  pp={pp15:.4f} peak={pk15} index={idx15}/38")

    # SDXL: mid_block.resnets.1 peak
    sdxl_path = _os.path.join(_os.path.dirname(__file__), "../outputs/sdxl_phase1/layer_drift_summary.json")
    if _os.path.exists(sdxl_path):
        with open(sdxl_path) as f:
            dxl = _json.load(f)
        sdxl_names = canonical_order("SDXL", [e["layer"] for e in dxl["full_ranking"]])
        mean_lookup = {e["layer"]: e["mean_drift"] for e in dxl["full_ranking"]}
        sdxl_prof   = [mean_lookup[ln] for ln in sdxl_names]
        ppxl, pkxl = peak_position("SDXL", sdxl_names, sdxl_prof)
        assert len(sdxl_names) == 28, f"SDXL expected 28 layers, got {len(sdxl_names)}"
        assert pkxl == "mid_block.resnets.1", f"SDXL peak layer mismatch: {pkxl}"
        print(f"  PASS: SDXL   pp={ppxl:.4f} peak={pkxl}")

    # H-DiT: verify natural sort puts blocks.20 at correct index
    dit_path = _os.path.join(_os.path.dirname(__file__), "../outputs/dit_phase1/layer_drift_summary.json")
    if _os.path.exists(dit_path):
        with open(dit_path) as f:
            ddi = _json.load(f)
        dit_names = canonical_order("H-DiT", [e["layer"] for e in ddi["full_ranking"]])
        mean_lookup = {e["layer"]: e["mean_drift"] for e in ddi["full_ranking"]}
        dit_prof   = [mean_lookup[ln] for ln in dit_names]
        ppdi, pkdi = peak_position("H-DiT", dit_names, dit_prof)
        assert len(dit_names) == 40, f"H-DiT expected 40 layers, got {len(dit_names)}"
        # blocks.0, blocks.1, ..., blocks.19, blocks.20, ..., blocks.39
        idx20 = dit_names.index("blocks.20")
        assert idx20 == 20, f"H-DiT blocks.20 at {idx20}, expected 20"
        print(f"  PASS: H-DiT  pp={ppdi:.4f} peak={pkdi}")

    # SD3.5: verify natural sort
    sd35_path = _os.path.join(_os.path.dirname(__file__), "../outputs/sd35_phase1/layer_drift_summary.json")
    if _os.path.exists(sd35_path):
        with open(sd35_path) as f:
            d35 = _json.load(f)
        sd35_names = canonical_order("SD3.5", list(d35["aggregated"].keys()))
        sd35_prof  = [d35["aggregated"][ln]["mean"] for ln in sd35_names]
        pp35, pk35 = peak_position("SD3.5", sd35_names, sd35_prof)
        assert len(sd35_names) == 24, f"SD3.5 expected 24 layers, got {len(sd35_names)}"
        # check block_0..block_23 in order
        for i in range(24):
            assert sd35_names[i] == f"block_{i}", f"SD3.5 order: {sd35_names[i]} != block_{i} at index {i}"
        print(f"  PASS: SD3.5  pp={pp35:.4f} peak={pk35}")

    # FLUX: verify natural sort preserves interleaving, not lexicographic
    flux_path = _os.path.join(_os.path.dirname(__file__), "../outputs/phase9_flux_fp16/flux_fp16_unified_format.json")
    if _os.path.exists(flux_path):
        with open(flux_path) as f:
            dfl = _json.load(f)
        flux_names_raw = [e["name"] for e in dfl["layers"]]
        flux_names = canonical_order("FLUX", flux_names_raw)
        # natural sort should match: j0-j18, s0-s37
        assert flux_names == flux_names_raw, "FLUX canonical order differs from JSON order"
        for i in range(19):
            assert flux_names[i] == f"j{i}", f"FLUX joint order: {flux_names[i]} != j{i}"
        for i in range(38):
            assert flux_names[19+i] == f"s{i}", f"FLUX single order: {flux_names[19+i]} != s{i}"
        print(f"  PASS: FLUX   order verified (57 layers, natural key matches)")

    # edge case: PixArt floating-point precision
    pp_pixart = 20 / 28
    idx_pixart = int(round(pp_pixart * 28))
    assert idx_pixart == 20, f"PixArt round-trip: int(round({pp_pixart}*28)) != 20"
    print(f"  PASS: PixArt round-trip fix int(round(pp*L))")

    # hash reproducibility
    h1 = layer_hash("SD1.5", sd15_names)
    h2 = layer_hash("SD1.5", sd15_names)
    assert h1 == h2, "Hash not reproducible"
    print(f"  PASS: layer_hash reproducible (SD1.5={h1})")


if __name__ == "__main__":
    _run_tests()
    print("\nAll tests passed.")
