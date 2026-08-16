"""
P0b v3: C1 weight perturbation — D_s decomposition, PSNR co-axis,
        cliff densification, bootstrap noise floor, multi-seed.

v3 additions over v2:
  1. D_s decomposed into 3 sub-components:
     - D_peak:  peak position difference (|pos_a - pos_b|)
     - D_shape: profile shape difference (Spearman distance on normalized profile)
     - D_mag:   magnitude distribution difference (concentration + spread)
  2. PSNR(epsilon) on same axis to establish death hierarchy
  3. Denser cliff sampling: 1.5e-5, 2e-5, 2.5e-5
  4. Bootstrap noise floor: B=100 resamples of 19 images, median + 95th CI
  5. 3 noise seeds per epsilon to check cliff sharpness
  6. NaN diagnosis: fp16 overflow check

Usage:
  python -u scripts/p0b_weight_perturb_v3.py --images 19
"""

import copy, json, sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
COCO_VAL_DIR = PROJECT_ROOT / "data" / "coco_val"
TEST_IMAGES = sorted(COCO_VAL_DIR.glob("*.jpg"))

# Epsilons with dense cliff sampling
EPSILONS = [1e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 5e-5, 1e-4, 3e-4, 1e-3]
N_SEEDS = 3  # noise seeds per epsilon
N_BOOT = 100  # bootstrap replicates for noise floor


# ---------------------------------------------------------------------------
# UNet hooks
# ---------------------------------------------------------------------------
def discover_hook_targets(unet):
    targets = []
    for name, module in unet.named_modules():
        parts = name.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts) == idx + 2 and parts[-1].isdigit():
                targets.append(name)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts) == idx + 2 and parts[-1] == "0":
                targets.append(name)
    return sorted(targets)

class UNetFeatureHooker:
    def __init__(self, unet):
        self.unet = unet; self.features = {}; self.handles = []
    def _hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach().float().cpu()
        return fn
    def register(self, targets):
        self.remove()
        for name, module in self.unet.named_modules():
            if name in targets:
                self.handles.append(module.register_forward_hook(self._hook_fn(name)))
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear(); self.features.clear()


# ---------------------------------------------------------------------------
# Weight perturbation
# ---------------------------------------------------------------------------
def is_weight_param(name, param):
    if "bias" in name: return False
    if "norm" in name.lower() or "ln_" in name.lower(): return False
    if param.ndim < 2: return False
    return True

def perturb_weights(unet, epsilon, seed=42):
    torch.manual_seed(seed)
    unet_pert = copy.deepcopy(unet)
    with torch.no_grad():
        for name, param in unet_pert.named_parameters():
            if not is_weight_param(name, param): continue
            w_flat = param.data.view(-1).float()
            w_norm = w_flat.norm().item()
            noise_scale = epsilon * max(w_norm, 1e-8)
            noise = torch.randn_like(w_flat) * noise_scale
            param.data.copy_((w_flat + noise).view(param.shape).to(param.dtype))
    return unet_pert


# ---------------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------------
def encode_empty_prompt(pipe):
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        return pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

def ddim_inversion(pipe, latents, prompt_embeds, num_steps):
    sched = pipe.scheduler; sched.set_timesteps(num_steps, device=DEVICE)
    ts = sched.timesteps; z = latents.clone()
    extended = ts.tolist() + [0]
    with torch.no_grad():
        for i in range(len(extended)-1, 0, -1):
            tc, tn = extended[i], extended[i-1]
            npred = pipe.unet(z, tc, encoder_hidden_states=prompt_embeds).sample
            ac = sched.alphas_cumprod[tc]; an = sched.alphas_cumprod[tn]
            c1 = (an/ac).sqrt(); sc = (1-ac).sqrt(); sn = (1-an).sqrt()
            z = c1*z + (sn - c1*sc)*npred
    return z

def ddim_recon(pipe, noise, prompt_embeds, num_steps):
    sched = pipe.scheduler; sched.set_timesteps(num_steps, device=DEVICE)
    z = noise.clone()
    with torch.no_grad():
        for t in sched.timesteps:
            npred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
            z = sched.step(npred, t, z).prev_sample
    return z

def load_and_encode(pipe, path, size=512):
    img = Image.open(path).convert("RGB").resize((size,size))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tensor = 2*tensor - 1
    with torch.no_grad():
        latent = pipe.vae.encode(tensor).latent_dist.sample()
        return latent * pipe.vae.config.scaling_factor


# ---------------------------------------------------------------------------
# Structural features + D_s decomposition
# ---------------------------------------------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    if s == 0: return 0.0
    return float((2*np.sum(np.arange(1,n+1)*x) - (n+1)*s) / (n*s))

def extract_features(profile):
    """4 features: peak_pos, peak_count, concentration, spread"""
    from scipy.signal import find_peaks
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    pp = float(np.argmax(p))/L
    peaks,_ = find_peaks(p, prominence=0.1)
    pc = int(np.sum(p[peaks] > 0.5)) if len(peaks) > 0 else 0
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(p)[-k:]
    conc = float(np.sum(p[top])/np.sum(p)) if np.sum(p) > 0 else 0.0
    sp = float(gini(p))
    return {"peak_position": pp, "peak_count": pc, "concentration": conc, "spread": sp}

def structural_distance(fa, fb):
    keys = ["peak_position","peak_count","concentration","spread"]
    return float(np.linalg.norm(np.array([fa[k] for k in keys]) - np.array([fb[k] for k in keys])))

def decomposed_distance(profile_a_norm, profile_b_norm, feat_a, feat_b):
    """Decompose D_s into D_peak, D_shape, D_mag."""
    pa, pb = np.asarray(profile_a_norm), np.asarray(profile_b_norm)
    # D_peak: normalized peak position difference
    d_peak = abs(feat_a["peak_position"] - feat_b["peak_position"])
    # D_shape: 1 - Spearman rho (shape dissimilarity)
    rho, _ = spearmanr(pa, pb)
    d_shape = 1.0 - float(rho) if not np.isnan(rho) else 1.0
    # D_mag: concentration + spread difference
    d_mag = np.linalg.norm([
        feat_a["concentration"] - feat_b["concentration"],
        feat_a["spread"] - feat_b["spread"]
    ])
    d_total = float(np.linalg.norm([d_peak, d_shape, d_mag]))
    return {"D_peak": d_peak, "D_shape": d_shape, "D_mag": d_mag, "D_total": d_total}


# ---------------------------------------------------------------------------
# Single diagnostic pass (returns drift profile + PSNR for first image)
# ---------------------------------------------------------------------------
def diagnose(pipe, prompt_embeds, hooker, targets, images, num_steps=50):
    per_image = []; first_psnr = None
    for i, img_path in enumerate(tqdm(images, desc="    diag", leave=False)):
        try:
            latent = load_and_encode(pipe, str(img_path))
            z_inv = ddim_inversion(pipe, latent, prompt_embeds, num_steps)
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=prompt_embeds).sample
            inv_f = {k: v.clone() for k,v in hooker.features.items()}
            z_recon = ddim_recon(pipe, z_inv, prompt_embeds, num_steps)
            hooker.features.clear()
            with torch.no_grad():
                pipe.unet(z_recon, pipe.scheduler.timesteps[0], encoder_hidden_states=prompt_embeds).sample
            recon_f = {k: v.clone() for k,v in hooker.features.items()}
            ld = {}
            for ln in targets:
                if ln in inv_f and ln in recon_f:
                    ld[ln] = float(torch.norm(inv_f[ln]-recon_f[ln], p=2).item())
            per_image.append(ld)
            if i == 0:
                t = pipe.vae.decode(z_recon/pipe.vae.config.scaling_factor).sample
                ref_t = pipe.vae.decode(latent/pipe.vae.config.scaling_factor).sample
                mse = float((t-ref_t).pow(2).mean().item())
                first_psnr = float(10*np.log10(4.0/max(mse,1e-12)))
        except Exception as e:
            print(f"      err {img_path.name}: {e}")
    if not per_image: return None, None
    layers = list(per_image[0].keys())
    mean_d = {l: float(np.mean([d[l] for d in per_image if l in d])) for l in layers}
    ordered = sorted(mean_d.items(), key=lambda x: targets.index(x[0]))
    profile = np.array([v for _,v in ordered])
    dmin, dmax = profile.min(), profile.max()
    norm = (profile-dmin)/(dmax-dmin) if dmax>dmin else profile.copy()
    feat = extract_features(norm)
    feat["peak_layer"] = ordered[int(np.argmax(norm))][0]
    feat["drift_range"] = [float(dmin), float(dmax)]
    feat["peak_margin"] = float(np.max(norm) / (np.median(norm)+1e-8))  # peak prominence
    return {
        "layer_names": [k for k,_ in ordered],
        "drift_raw": profile.tolist(),
        "drift_norm": norm.tolist(),
        "structural_features": feat,
        "per_image": [{l: per_image[n][l] for l in layers} for n in range(len(per_image))],
        "first_psnr": first_psnr,
    }, (ordered, profile, norm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=19)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--quick", action="store_true", help="Only 1 seed, fewer epsilons")
    args = ap.parse_args()

    images = TEST_IMAGES[:args.images]
    epsilons = [1e-6,1e-5,1.5e-5,2e-5,2.5e-5,3e-5,5e-5,1e-4] if args.quick else EPSILONS
    n_seeds = 1 if args.quick else N_SEEDS

    print(f"N={len(images)}  epsilons={[f'{e:.0e}' for e in epsilons]}  seeds={n_seeds}")

    # Load model
    print("\nLoading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_hook_targets(pipe.unet)
    hooker = UNetFeatureHooker(pipe.unet)
    hooker.register(targets)
    prompt_embeds = encode_empty_prompt(pipe)
    n_w = sum(1 for n,p in pipe.unet.named_parameters() if is_weight_param(n,p))
    print(f"  Hook: {len(targets)} layers, perturbable: {n_w}")

    original_state = copy.deepcopy(pipe.unet.state_dict())

    # =====================================================================
    # Baseline (full set, no perturbation)
    # =====================================================================
    print("\n=== Baseline (epsilon=0) ===")
    base_r, base_data = diagnose(pipe, prompt_embeds, hooker, targets, images, args.steps)
    bf = base_r["structural_features"]
    print(f"  Peak: {bf['peak_layer']} pos={bf['peak_position']:.3f} margin={bf['peak_margin']:.1f}x")
    print(f"  PSNR(image0)={base_r['first_psnr']:.1f} dB")
    print(f"  Range=[{bf['drift_range'][0]:.4f},{bf['drift_range'][1]:.4f}]")

    # =====================================================================
    # Bootstrap noise floor (from baseline per-image profiles, zero GPU)
    # =====================================================================
    print("\n=== Bootstrap Noise Floor (B=100, from per-image data) ===")
    per_image_profiles = []
    for img_data in base_r["per_image"]:
        ordered = sorted(img_data.items(), key=lambda x: targets.index(x[0]))
        profile = np.array([v for _, v in ordered])
        dmin, dmax = profile.min(), profile.max()
        norm = (profile - dmin) / (dmax - dmin) if dmax > dmin else profile.copy()
        per_image_profiles.append(norm)

    boot_d_total, boot_d_peak, boot_d_shape, boot_d_mag = [], [], [], []
    rng = np.random.RandomState(0)
    for b in tqdm(range(N_BOOT), desc="    bootstrap", leave=False):
        idx_a = rng.choice(len(per_image_profiles), len(per_image_profiles), replace=True)
        idx_b = rng.choice(len(per_image_profiles), len(per_image_profiles), replace=True)
        prof_a = np.mean([per_image_profiles[i] for i in idx_a], axis=0)
        prof_b = np.mean([per_image_profiles[i] for i in idx_b], axis=0)
        for p in [prof_a, prof_b]:
            dmin, dmax = p.min(), p.max()
            if dmax > dmin: p[:] = (p - dmin) / (dmax - dmin)
        f_a = extract_features(prof_a); f_b = extract_features(prof_b)
        dd = decomposed_distance(prof_a, prof_b, f_a, f_b)
        boot_d_total.append(dd["D_total"]); boot_d_peak.append(dd["D_peak"])
        boot_d_shape.append(dd["D_shape"]); boot_d_mag.append(dd["D_mag"])
    if boot_d_total:
        nf = {
            "D_total_median": float(np.median(boot_d_total)),
            "D_total_p95": float(np.percentile(boot_d_total, 95)),
            "D_peak_median": float(np.median(boot_d_peak)),
            "D_shape_median": float(np.median(boot_d_shape)),
            "D_mag_median": float(np.median(boot_d_mag)),
        }
        print(f"  Noise floor D_total: median={nf['D_total_median']:.6f}  p95={nf['D_total_p95']:.6f}")
        print(f"  D_peak: {nf['D_peak_median']:.6f}  D_shape: {nf['D_shape_median']:.6f}  D_mag: {nf['D_mag_median']:.6f}")
    else:
        nf = None

    # =====================================================================
    # Perturbed variants (multi-seed per epsilon)
    # =====================================================================
    results = []  # list of {epsilon, seed, D_s, D_decomposed, psnr, ...}
    for eps in epsilons:
        for seed in range(n_seeds):
            label = f"eps{eps:.0e}_s{seed}"
            print(f"\n  {label}")

            pipe.unet.load_state_dict(copy.deepcopy(original_state))
            pipe.unet = perturb_weights(pipe.unet, eps, seed=42+seed*100+int(eps*1e10))

            hooker.remove(); hooker = UNetFeatureHooker(pipe.unet); hooker.register(targets)

            r, _ = diagnose(pipe, prompt_embeds, hooker, targets, images, args.steps)

            if r is None:
                results.append({"epsilon": float(eps), "seed": seed, "status": "crashed"})
                print(f"    CRASHED")
                continue

            pf = r["structural_features"]
            ds = structural_distance(bf, pf)
            dd = decomposed_distance(base_r["drift_norm"], r["drift_norm"], bf, pf)

            # Diagnose NaN
            has_nan_raw = bool(np.any(np.isnan(r["drift_raw"])))
            n_zero = int(np.sum(np.array(r["drift_raw"]) < 1e-8))

            entry = {
                "epsilon": float(eps), "seed": seed,
                "D_s": float(ds),
                "D_decomposed": {k: float(v) for k,v in dd.items()},
                "PSNR": r["first_psnr"],
                "peak_layer": pf["peak_layer"],
                "peak_pos": pf["peak_position"],
                "peak_margin": pf.get("peak_margin", 0),
                "peak_preserved": pf["peak_layer"] == bf["peak_layer"],
                "has_nan": has_nan_raw,
                "n_near_zero": n_zero,
                "drift_range_ratio": float(pf["drift_range"][1]/max(pf["drift_range"][0],1e-8)),
            }
            results.append(entry)
            print(f"    D_s={ds:.6f}  D_p={dd['D_peak']:.6f} D_sh={dd['D_shape']:.6f} D_m={dd['D_mag']:.6f}"
                  f"  PSNR={entry['PSNR']:.1f}  {'NaN!' if has_nan_raw else ''}")

    hooker.remove(); del pipe; torch.cuda.empty_cache()

    # =====================================================================
    # Summary
    # =====================================================================
    ref_inter = 0.249
    nf_total = nf["D_total_median"] if nf else 0.0
    nf_p95 = nf["D_total_p95"] if nf else 0.0

    print(f"\n{'='*70}")
    print("C1 v3: Weight Perturbation Dose-Response")
    print(f"{'='*70}")
    print(f"Noise floor (bootstrap, N={len(images)}): D_s median={nf_total:.6f}  p95={nf_p95:.6f}")
    print(f"Min inter-arch D_s = {ref_inter:.4f}")
    print(f"\n{'eps':>9s}  {'D_s':>8s}  {'D_peak':>8s} {'D_shape':>8s} {'D_mag':>8s}  {'PSNR':>7s}  {'peak':>10s}")
    print("-"*75)

    for eps in epsilons:
        eps_results = [r for r in results if abs(r.get("epsilon",0)-eps)<1e-12]
        if not eps_results or eps_results[0].get("status")=="crashed":
            print(f"{eps:9.0e}  {'CRASHED':>8s}")
            continue
        # Average over seeds for stable reporting
        ds_vals = [r["D_s"] for r in eps_results if "D_s" in r]
        dp_vals = [r["D_decomposed"]["D_peak"] for r in eps_results if "D_decomposed" in r]
        dsh_vals = [r["D_decomposed"]["D_shape"] for r in eps_results if "D_decomposed" in r]
        dm_vals = [r["D_decomposed"]["D_mag"] for r in eps_results if "D_decomposed" in r]
        psnr_vals = [r["PSNR"] for r in eps_results if r.get("PSNR") is not None]
        peak_ok = eps_results[0].get("peak_preserved", False)
        ds_mean = np.mean(ds_vals); psnr_mean = np.mean(psnr_vals) if psnr_vals else 0
        print(f"{eps:9.0e}  {ds_mean:8.6f}  {np.mean(dp_vals):8.6f} {np.mean(dsh_vals):8.6f} "
              f"{np.mean(dm_vals):8.6f}  {psnr_mean:6.1f}  {'✓' if peak_ok else '✗':>10s}")

    # C1 check
    stable = [eps for eps in epsilons
              if any(r.get("D_s",1)<ref_inter for r in results if abs(r.get("epsilon",0)-eps)<1e-12)]
    print(f"\nC1: stable regime epsilon <= {stable[-1]:.0e}" if stable else "\nC1: NO stable regime")
    if stable:
        max_ds = max(r["D_s"] for eps in stable
                     for r in results if abs(r.get("epsilon",0)-eps)<1e-12 and "D_s" in r)
        print(f"  Max D_s(stable) = {max_ds:.6f}  ({max_ds/nf_total:.1f}x noise floor)" if nf_total>0 else f"  Max D_s = {max_ds:.6f}")

    # Save
    summary = {
        "protocol": {"images": len(images), "steps": args.steps, "epsilons": epsilons,
                     "seeds_per_epsilon": n_seeds, "bootstrap_B": N_BOOT,
                     "perturbation": "per-layer relative, weight matrices only, bias/LN excluded"},
        "noise_floor_bootstrap": nf,
        "baseline": {"peak": bf["peak_layer"], "peak_pos": bf["peak_position"],
                     "peak_margin": bf["peak_margin"], "psnr": base_r["first_psnr"],
                     "features": bf},
        "results": results,
        "reference": {"min_inter_arch_Ds": ref_inter, "pair": "SD 1.5 vs SDXL"},
    }
    sp = OUT_DIR / "weight_perturbation_v3_summary.json"
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nSaved to {sp}")


if __name__ == "__main__":
    main()
