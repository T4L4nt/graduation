"""
P0b v4: C1 weight perturbation — D_s v2 metric (continuous-only, soft peak matching),
        per-image aggregation, UNet output norm curve.

v4 fixes over v3:
  1. D_s v2: peak_count removed from composite distance
     D_total = sqrt(D_peak_pos^2 + D_shape^2 + D_mag^2)
     D_mag = L2(concentration, spread)
     D_shape = 1 - Spearman rho on normalized profiles
     D_peak_pos = |peak_position_a - peak_position_b|
     D_peak_struct = soft peak matching distance (separate diagnostic, not in D_total)
  2. Per-image aggregation: extract features per-image first, then mean
  3. UNet output norm per epsilon (confirms PSNR degradation mechanism)
  4. Bootstrap noise floor with v2 metric
"""

import copy, json, sys
from pathlib import Path
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEVICE = "cuda"; DTYPE = torch.float16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
COCO_VAL = sorted((PROJECT_ROOT / "data" / "coco_val").glob("*.jpg"))

EPSILONS = [1e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 5e-5, 1e-4, 3e-4, 1e-3]
N_SEEDS = 3
N_BOOT = 100

# -------- hooks --------
def discover_targets(unet):
    t = []
    for n, m in unet.named_modules():
        p = n.split(".")
        if "resnets" in p:
            idx = p.index("resnets")
            if len(p) == idx+2 and p[-1].isdigit(): t.append(n)
        if "transformer_blocks" in p:
            idx = p.index("transformer_blocks")
            if len(p) == idx+2 and p[-1]=="0": t.append(n)
    return sorted(t)

class Hooker:
    def __init__(s, unet): s.u = unet; s.f = {}; s.h = []
    def _fn(s, n):
        def fn(m,i,o): s.f[n] = o.detach().float().cpu()
        return fn
    def reg(s, t):
        s.rm()
        for n, m in s.u.named_modules():
            if n in t: s.h.append(m.register_forward_hook(s._fn(n)))
    def rm(s):
        for h in s.h: h.remove()
        s.h.clear(); s.f.clear()

# -------- perturbation --------
def is_wt(n, p):
    if "bias" in n: return False
    if "norm" in n.lower(): return False
    if p.ndim < 2: return False
    return True

def perturb(unet, eps, seed=42):
    torch.manual_seed(seed)
    u2 = copy.deepcopy(unet)
    with torch.no_grad():
        for n, p in u2.named_parameters():
            if not is_wt(n, p): continue
            w = p.data.view(-1).float(); wn = w.norm().item()
            noise = torch.randn_like(w) * eps * max(wn, 1e-8)
            p.data.copy_((w+noise).view(p.shape).to(p.dtype))
    return u2

# -------- DDIM --------
def enc_empty(pipe):
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad(): return pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

def ddim_inv(pipe, lat, pe, N):
    s = pipe.scheduler; s.set_timesteps(N, device=DEVICE)
    ts = s.timesteps; z = lat.clone(); ext = ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]; npred=pipe.unet(z,tc,encoder_hidden_states=pe).sample
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc=(1-ac).sqrt(); sn=(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon(pipe, noise, pe, N):
    s=pipe.scheduler; s.set_timesteps(N, device=DEVICE); z=noise.clone()
    with torch.no_grad():
        for t in s.timesteps:
            npred=pipe.unet(z,t,encoder_hidden_states=pe).sample
            z=s.step(npred,t,z).prev_sample
    return z

def load_enc(pipe, path, sz=512):
    img = Image.open(path).convert("RGB").resize((sz,sz))
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    t = 2*t-1
    with torch.no_grad(): l=pipe.vae.encode(t).latent_dist.sample()
    return l*pipe.vae.config.scaling_factor

# -------- D_s v2: continuous-only metric --------
def gini(x):
    x=np.sort(np.asarray(x,dtype=np.float64)); n=len(x); s=np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0

def soft_peak_distance(profile_a, profile_b, prom_thresh=0.1):
    """Continuous peak structure distance via optimal assignment.
    0 = identical peak structures; 1 = completely different."""
    pa, pb = np.asarray(profile_a), np.asarray(profile_b)
    peaks_a, props_a = find_peaks(pa, prominence=prom_thresh)
    peaks_b, props_b = find_peaks(pb, prominence=prom_thresh)
    prom_a = props_a['prominences'] if len(peaks_a)>0 else np.array([])
    prom_b = props_b['prominences'] if len(peaks_b)>0 else np.array([])
    pos_a = peaks_a / len(pa) if len(peaks_a)>0 else np.array([])
    pos_b = peaks_b / len(pb) if len(peaks_b)>0 else np.array([])
    na, nb = len(peaks_a), len(peaks_b)
    if na==0 and nb==0: return 0.0
    mx = max(na, nb)
    cost = np.ones((mx, mx))
    for i in range(na):
        for j in range(nb):
            pd = abs(pos_a[i]-pos_b[j])
            pr = min(prom_a[i],prom_b[j])/max(prom_a[i],prom_b[j],1e-8)
            cost[i,j] = pd + (1.0-pr)
    ri, cj = linear_sum_assignment(cost)
    mc = 0.0
    for i,j in zip(ri,cj):
        if i<na and j<nb: mc += cost[i,j]
        elif i<na: mc += min(prom_a[i], 1.0)
        elif j<nb: mc += min(prom_b[j], 1.0)
    return float(mc / max(mx, 1))

def structural_features_v2(profile_raw):
    """Per-image continuous features. profile_raw is NOT normalized."""
    p = np.asarray(profile_raw, dtype=np.float64)
    L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax>dmin else p.copy()
    pp = float(np.argmax(pn))/L
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    # Peak prominence of the top peak (continuous measure of peak quality)
    peaks, props = find_peaks(pn, prominence=0.1)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks)>0 else 0.0
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "top_prominence": top_prom, "n_peaks": len(peaks)}

def distance_v2(feat_a, feat_b, profile_a=None, profile_b=None):
    """Continuous-only D_s v2.
    D_total = sqrt(D_pp^2 + D_shape^2 + D_mag^2)
    D_peak_struct reported separately if profiles provided.
    """
    d_pp = abs(feat_a["peak_position"] - feat_b["peak_position"])
    d_mag = np.linalg.norm([
        feat_a["concentration"] - feat_b["concentration"],
        feat_a["spread"] - feat_b["spread"]
    ])
    d_shape = 0.0
    if profile_a is not None and profile_b is not None:
        rho, _ = spearmanr(np.asarray(profile_a), np.asarray(profile_b))
        d_shape = 1.0 - float(rho) if not np.isnan(rho) else 1.0
    d_total = float(np.linalg.norm([d_pp, d_shape, d_mag]))
    d_peak_struct = None
    if profile_a is not None and profile_b is not None:
        d_peak_struct = soft_peak_distance(profile_a, profile_b)
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_shape": d_shape,
            "D_mag": d_mag, "D_peak_struct": d_peak_struct}

# -------- Single-image diagnostic --------
def diagnose_single(pipe, pe, hooker, targets, img_path):
    """Run inversion-reconstruction on one image, return per-layer drift + UNet output norm."""
    latent = load_enc(pipe, img_path)
    # Inversion
    z_inv = ddim_inv(pipe, latent, pe, 50)
    # UNet output norm at turnaround
    hooker.f.clear()
    with torch.no_grad():
        unet_out = pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    unet_norm = float(torch.norm(unet_out).item())
    inv_f = {k: v.clone() for k,v in hooker.f.items()}
    # Reconstruction
    z_recon = ddim_recon(pipe, z_inv, pe, 50)
    hooker.f.clear()
    with torch.no_grad():
        pipe.unet(z_recon, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    recon_f = {k: v.clone() for k,v in hooker.f.items()}
    # Per-layer drift
    ld = {}
    for ln in targets:
        if ln in inv_f and ln in recon_f:
            ld[ln] = float(torch.norm(inv_f[ln]-recon_f[ln], p=2).item())
    # PSNR
    t_rec = pipe.vae.decode(z_recon/pipe.vae.config.scaling_factor).sample
    t_ref = pipe.vae.decode(latent/pipe.vae.config.scaling_factor).sample
    mse = float((t_rec-t_ref).pow(2).mean().item())
    psnr = float(10*np.log10(4.0/max(mse,1e-12)))
    return ld, unet_norm, psnr

# -------- Main --------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=19)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    images = COCO_VAL[:args.images]
    epsilons = [1e-6,1e-5,2e-5,5e-5,1e-4,3e-4,1e-3] if args.quick else EPSILONS
    n_seeds = 1 if args.quick else args.seeds

    print(f"N={len(images)} epsilons={[f'{e:.0e}' for e in epsilons]} seeds={n_seeds}")

    # Load
    print("\nLoading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_targets(pipe.unet)
    hooker = Hooker(pipe.unet); hooker.reg(targets)
    pe = enc_empty(pipe)
    orig = copy.deepcopy(pipe.unet.state_dict())
    print(f"  {len(targets)} layers")

    # ===== Baseline =====
    print("\n=== Baseline ===")
    per_img_raw = []; unet_norms_b = []; psnrs_b = []
    for imp in tqdm(images, desc="  baseline"):
        ld, un, psnr = diagnose_single(pipe, pe, hooker, targets, imp)
        per_img_raw.append(ld); unet_norms_b.append(un); psnrs_b.append(psnr)
    # Per-image features
    base_per_img = []
    for ld in per_img_raw:
        ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
        profile = np.array([v for _,v in ordered])
        base_per_img.append({"profile": profile, "features": structural_features_v2(profile)})
    # Aggregate: mean of per-image features (v2 aggregation)
    bf = {}
    for k in ["peak_position","concentration","spread","top_prominence","n_peaks"]:
        bf[k] = float(np.mean([pi["features"][k] for pi in base_per_img]))
    # Mean profile for shape distance computation
    base_mean_profile = np.mean([pi["profile"] for pi in base_per_img], axis=0)
    dmin, dmax = base_mean_profile.min(), base_mean_profile.max()
    base_mean_norm = (base_mean_profile-dmin)/(dmax-dmin) if dmax>dmin else base_mean_profile.copy()
    bf_unet_norm = float(np.mean(unet_norms_b))
    bf_psnr = float(np.mean(psnrs_b))
    print(f"  peak_pos={bf['peak_position']:.4f} conc={bf['concentration']:.4f} sp={bf['spread']:.4f}")
    print(f"  UNet out norm={bf_unet_norm:.1f} PSNR={bf_psnr:.1f}")

    # ===== Bootstrap noise floor (v2 metric, per-image aggregation) =====
    print(f"\n=== Bootstrap Noise Floor (B={N_BOOT}) ===")
    boot = {"D_total":[],"D_peak_pos":[],"D_shape":[],"D_mag":[],"D_peak_struct":[]}
    rng = np.random.RandomState(0)
    # Per-image individual profiles for bootstrapping
    ind_profiles = [pi["profile"] for pi in base_per_img]
    for b in tqdm(range(N_BOOT), desc="  bootstrap", leave=False):
        ia = rng.choice(len(ind_profiles), len(ind_profiles), replace=True)
        ib = rng.choice(len(ind_profiles), len(ind_profiles), replace=True)
        pa = np.mean([ind_profiles[i] for i in ia], axis=0)
        pb = np.mean([ind_profiles[i] for i in ib], axis=0)
        dma, dmia = pa.min(), pa.max()
        dmb, dmib = pb.min(), pb.max()
        pna = (pa-dma)/(dmia-dma) if dmia>dma else pa.copy()
        pnb = (pb-dmb)/(dmib-dmb) if dmib>dmb else pb.copy()
        fa = structural_features_v2(pa); fb = structural_features_v2(pb)
        dd = distance_v2(fa, fb, pna, pnb)
        for k in boot: boot[k].append(dd[k])
    nf = {f"noise_{k}": float(np.median(boot[k])) for k in boot}
    nf["noise_D_total_p95"] = float(np.percentile(boot["D_total"], 95))
    print(f"  D_total: median={nf['noise_D_total']:.6f} p95={nf['noise_D_total_p95']:.6f}")
    print(f"  D_peak_pos={nf['noise_D_peak_pos']:.6f} D_shape={nf['noise_D_shape']:.6f}")
    print(f"  D_mag={nf['noise_D_mag']:.6f} D_peak_struct={nf['noise_D_peak_struct']:.6f}")

    # ===== Perturbed variants =====
    all_results = []
    for eps in epsilons:
        for seed in range(n_seeds):
            label = f"eps{eps:.0e}_s{seed}"
            print(f"\n  {label}")
            pipe.unet.load_state_dict(copy.deepcopy(orig))
            pipe.unet = perturb(pipe.unet, eps, seed=42+seed*100+int(eps*1e10))
            hooker.rm(); hooker = Hooker(pipe.unet); hooker.reg(targets)
            per_img = []; uns = []; psnrs = []
            for imp in tqdm(images, desc=f"    {label}", leave=False):
                try:
                    ld, un, psnr = diagnose_single(pipe, pe, hooker, targets, imp)
                    per_img.append(ld); uns.append(un); psnrs.append(psnr)
                except Exception as e:
                    print(f"    err {imp.name}: {e}")
            if not per_img:
                all_results.append({"epsilon": float(eps), "seed": seed, "status": "crashed"})
                continue
            # Per-image features
            pi_feats = []
            for ld in per_img:
                ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
                profile = np.array([v for _,v in ordered])
                pi_feats.append({"profile": profile, "features": structural_features_v2(profile)})
            # Aggregate per-image features
            pf = {}
            for k in ["peak_position","concentration","spread","top_prominence","n_peaks"]:
                pf[k] = float(np.mean([pi["features"][k] for pi in pi_feats]))
            pert_mean_profile = np.mean([pi["profile"] for pi in pi_feats], axis=0)
            dmin, dmax = pert_mean_profile.min(), pert_mean_profile.max()
            pert_mean_norm = (pert_mean_profile-dmin)/(dmax-dmin) if dmax>dmin else pert_mean_profile.copy()
            dd = distance_v2(bf, pf, base_mean_norm, pert_mean_norm)
            has_nan = bool(np.any(np.isnan(pert_mean_profile)))
            entry = {"epsilon": float(eps), "seed": seed,
                     "D_total": float(dd["D_total"]), "D_peak_pos": float(dd["D_peak_pos"]),
                     "D_shape": float(dd["D_shape"]), "D_mag": float(dd["D_mag"]),
                     "D_peak_struct": float(dd["D_peak_struct"]) if dd["D_peak_struct"] is not None else None,
                     "UNet_norm": float(np.mean(uns)), "PSNR": float(np.mean(psnrs)),
                     "peak_pos": pf["peak_position"], "conc": pf["concentration"],
                     "spread": pf["spread"], "n_peaks": pf["n_peaks"],
                     "peak_preserved": int(np.argmax(pert_mean_norm)/len(targets)) == int(np.argmax(base_mean_norm)/len(targets)),
                     "has_nan": has_nan}
            all_results.append(entry)
            print(f"    D_total={dd['D_total']:.6f} D_pp={dd['D_peak_pos']:.6f} D_sh={dd['D_shape']:.6f}"
                  f" D_mag={dd['D_mag']:.6f} D_ps={dd.get('D_peak_struct',0):.6f}"
                  f"  UNet={entry['UNet_norm']:.1f} PSNR={entry['PSNR']:.1f}")

    hooker.rm(); del pipe; torch.cuda.empty_cache()

    # ===== Summary =====
    ref_inter = 0.249
    print(f"\n{'='*75}")
    print("C1 v4: D_s v2 (continuous-only, per-image aggregation)")
    print(f"{'='*75}")
    print(f"Noise floor D_total: median={nf['noise_D_total']:.6f} p95={nf['noise_D_total_p95']:.6f}")
    print(f"  D_peak_pos={nf['noise_D_peak_pos']:.6f} D_shape={nf['noise_D_shape']:.6f}")
    print(f"  D_mag={nf['noise_D_mag']:.6f} D_peak_struct={nf['noise_D_peak_struct']:.6f}")
    print(f"Baseline: pp={bf['peak_position']:.4f} UNet={bf_unet_norm:.1f} PSNR={bf_psnr:.1f}")
    print(f"\n{'eps':>9s}  {'D_total':>10s}  {'D_pp':>8s}  {'D_sh':>8s}  {'D_mag':>8s}  {'D_ps':>8s}  {'UNet':>8s}  {'PSNR':>7s}")
    print("-"*85)

    for eps in epsilons:
        v = [r for r in all_results if abs(r.get("epsilon",0)-eps)<1e-12 and r.get("status")!="crashed"]
        if not v:
            print(f"{eps:9.1e}  {'CRASHED':>10s}")
            continue
        dm_total = np.mean([r["D_total"] for r in v])
        dm_pp = np.mean([r["D_peak_pos"] for r in v])
        dm_sh = np.mean([r["D_shape"] for r in v])
        dm_mag = np.mean([r["D_mag"] for r in v])
        dm_ps = np.mean([r["D_peak_struct"] for r in v if r.get("D_peak_struct") is not None])
        dm_un = np.mean([r["UNet_norm"] for r in v])
        dm_pn = np.mean([r["PSNR"] for r in v])
        print(f"{eps:9.1e}  {dm_total:10.6f}  {dm_pp:8.6f}  {dm_sh:8.6f}  {dm_mag:8.6f}  {dm_ps:8.6f}  {dm_un:8.1f}  {dm_pn:6.1f}")

    # Stable regime
    stable = []
    for eps in epsilons:
        v = [r for r in all_results if abs(r.get("epsilon",0)-eps)<1e-12 and r.get("status")!="crashed"]
        if v:
            dm = np.mean([r["D_total"] for r in v])
            if dm < nf["noise_D_total_p95"]:
                stable.append((eps, "below_noise"))
            elif dm < ref_inter:
                stable.append((eps, "below_inter_arch"))

    print(f"\nStable regime (D_total < noise floor p95={nf['noise_D_total_p95']:.6f}):")
    for eps, tag in stable:
        print(f"  ε={eps:.1e} {'(BELOW NOISE FLOOR)' if tag=='below_noise' else ''}")

    summary = {
        "protocol": {"images": len(images), "epsilons": epsilons, "seeds": n_seeds,
                     "bootstrap_B": N_BOOT,
                     "metric": "D_s v2: continuous only (peak_pos, shape, mag), per-image aggregation"},
        "noise_floor": nf,
        "baseline": {"features": bf, "UNet_norm": bf_unet_norm, "PSNR": bf_psnr},
        "results": all_results,
        "reference": {"min_inter_arch": ref_inter, "pair": "SD 1.5 vs SDXL"},
    }
    sp = OUT_DIR / "weight_perturbation_v4_summary.json"
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nSaved to {sp}")


if __name__ == "__main__":
    main()
