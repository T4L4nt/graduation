# Protocol Manifest — Architecture Fingerprint Measurement

## Active Protocol: P-multi (Definition 1 compliant)

```
protocol_id: P-multi-v1
definition:   d_l(x) = E_{t∈K} ‖f_inv(x,t) − f_recon(x,t)‖₂
```

### K (timestep set) — RELATIVE definition
K is defined relative to the scheduler's step grid, not by absolute index:

> **Rule:** among the N sampler steps, take 9 indices: the first 3, the 3 around the midpoint, and the last 3.

```python
n = len(timesteps)          # scheduler-specific grid size
if n <= 6:
    key_indices = list(range(n))
else:
    key_indices = [0, 1, 2, n//2-1, n//2, n//2+1, n-3, n-2, n-1]
```

Per-model expansion (N=50):
- DDIM (SD1.5/SDXL/H-DiT/PixArt): `K = {0,1,2,24,25,26,47,48,49}` of scheduler.timesteps
- Euler/Flow (FLUX/SD3.5): `K = {0,1,2,24,25,26,47,48,49}` of the sigma grid
- The matched (t, layer) comparison uses the same timestep value for ref and recon paths

### Validation gate (new spec — no P-multi history exists for 5/6 architectures)
1. Shared module import: `layer_order` + aggregation helpers (no rewrite)
2. Code review of the adapter (per-architecture noise schedule + hook)
3. Runtime assertions: no NaN, `pp×L` is a valid integer index, profile smoothness (adjacent-layer L1 delta < 3× profile std)
4. Loose comparison vs old values: alarm only if deviation > 2× (protocol effect expected)

### Reference
DDPM-forward noise on original latent at timestep t:
- `z_ref = √α_t · x_0 + √(1-α_t) · ε_seed`  (seed=42, one seed per image)
- `f_ref[t] = UNet(z_ref, t)`
- Independent of inversion trajectory (external reference)

### Comparison
Reconstruction trajectory at matched timestep:
- Record inversion `inv_latent`, run DDIM reconstruction, save `recon_latents[idx]` at each step
- `f_recon[t] = UNet(recon_latents[idx], t)`

### Layer statistics
- `resnets.N`: raw block output (4D tensor [B,C,H,W])
- `transformer_blocks.0`: spatial mean pool ([B,N,C] → [B,1,C])
- Hook targets: `discover_hook_targets()` → lexicographic sort → canonical reorder via `layer_order.unet_topo_key`
- (For DiT/MMDiT architectures: equivalent hook at each transformer block output)

### Aggregation chain
1. Per-layer, per-timestep: L2(f_ref[t,l] − f_recon[t,l])
2. Mean over K → d_l(x) per image
3. Mean over images → aggregate profile vector
4. Min-max normalize → [0,1] profile (per architecture, across layers)

### Feature extraction
- `pp = argmax(p_normalized) / L`  (canonical layer order, hash recorded)
- `concentration = ∑ top-20% p_normalized / ∑ p_normalized`
- `spread = Gini(p_normalized)`  — `gini(x) = (2∑ i·x_sorted[i] − (n+1)∑x_sorted) / (n·∑x_sorted)`
- Assertion: `pp × L` must be valid layer index (integer)

### Inversion protocol
- Sampler: DDIM, 50 steps, η=0
- Prompt: "" (empty), guidance_scale=1.0
- Seed strategy: torch.manual_seed(42) before each image's DDPM reference noise
- dtype: fp16 (torch.float16)
- Image size: architecture-native (SD1.5=512², SDXL=1024², etc.)

### Output schema
Every P-multi output JSON must contain:
```
n_images, steps, protocol_id="P-multi-v1", K_indices,
layer_list_hash (sha256/12 of canonical layer list),
features: {peak_position, concentration, spread, n_peaks, L, peak_layer},
per_image: {layer: {img: drift}}, profile: {layer: mean_drift}
```

---

## Historical Protocol Registry

| Protocol ID | Description | K | Ref | Used in | Status |
|-------------|-------------|---|-----|---------|--------|
| P-multi | Multi-timestep mean, DDPM ref, 9 indices | {0,1,2,23,24,25,47,48,49} | external DDPM | Phase1 (SD1.5, SDXL, SD3.5) | **active canonical** |
| P-t0 | Single-step at last timestep (t≈0) | {last step} | inversion last-step | unified_100img (all 6), SDXL phase1, SD3.5 phase1 | robustness appendix |
| P-tT | Single-step at first timestep (t≈T) | {t=timesteps[0]} | inversion first-step | p0b_*_fingerprint.py (RV, LCM, RandText, SD1.4) | deprecated |

### Known protocol divergence
- P-multi vs P-t0: conc/sp differ by up to ~20% (SD1.5 0.583 vs 0.524)
- P-multi vs P-tT: conc/sp differ by unknown amount (Band 1 pending remeasurement)
- pp is stable across all three protocols (Δ=0 for SD1.5, SDXL, H-DiT, SD3.5)

---

## Architecture-specific notes

### UNet (SD1.5, SDXL)
- Canonical order: `unet_topo_key` (resnets before attentions in down/up blocks; attn before resnets in mid)
- Hook targets identical to phase1_diagnostics.py

### DiT (H-DiT, PixArt-Σ)
- Canonical order: `natural_key` on block index
- Reference: clean latent forward at t=0 (P-multi adaptation for architecture without noise-level UNet)
- K = {t=0} (single-step) — adaptation documented
- hook: transformer block output (raw hidden_states for single-stream, image hidden_states [output[1]] for MMDiT)

### MMDiT (FLUX, SD3.5)
- Canonical order: `natural_key`
- hook: image hidden_states = output[1] of JointTransformerBlock
- Euler scheduler (sigma-based), not DDIM
- Reference: clean latent forward at t=0
- K = {t=0} (single-step adaptation)
