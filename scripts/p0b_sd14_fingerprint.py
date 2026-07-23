"""
P0b: SD 1.4 real fingerprint — the C1 closure experiment.

Loads SD 1.4 UNet from cache, swaps into SD 1.5 pipeline (shared VAE/encoder),
runs full diagnostic on 19 coco_val images, v2 metric.

Pre-registered C1 decision rule:
  D_s(SD1.4, SD1.5) < noise floor p95 (0.016)
    → "checkpoint-invariant"  ✓ C1 PASS
  D_s between floor p95 (0.016) and min inter-arch (0.249)
    → "weak variation, but below cross-architecture by an order of magnitude"
  D_s approaching min inter-arch (0.249)
    → C1 in trouble, re-examine the claim scope

Usage:
  python -u scripts/p0b_sd14_fingerprint.py
"""

import copy, json, sys
from pathlib import Path
import torch, numpy as np
from PIL import Image
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
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

MODEL_15 = "runwayml/stable-diffusion-v1-5"
CACHE_14_UNET = "/home/hiaskc/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/main/unet"
COCO_VAL = sorted((PROJECT_ROOT / "data" / "coco_val").glob("*.jpg"))


# Reuse v4's hooking, DDIM, features, distance_v2 (copy inline for modularity)
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

def diagnose_single(pipe, pe, hooker, targets, imp):
    latent = load_enc(pipe, imp)
    z_inv = ddim_inv(pipe, latent, pe, 50)
    hooker.f.clear()
    with torch.no_grad(): pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    inv_f = {k: v.clone() for k,v in hooker.f.items()}
    z_recon = ddim_recon(pipe, z_inv, pe, 50)
    hooker.f.clear()
    with torch.no_grad(): pipe.unet(z_recon, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    recon_f = {k: v.clone() for k,v in hooker.f.items()}
    ld = {}
    for ln in targets:
        if ln in inv_f and ln in recon_f:
            ld[ln] = float(torch.norm(inv_f[ln]-recon_f[ln], p=2).item())
    unet_norm = float(torch.norm(inv_f.get(targets[-1], torch.zeros(1))).item())  # rough proxy
    t_rec = pipe.vae.decode(z_recon/pipe.vae.config.scaling_factor).sample
    t_ref = pipe.vae.decode(latent/pipe.vae.config.scaling_factor).sample
    mse = float((t_rec-t_ref).pow(2).mean().item())
    psnr = float(10*np.log10(4.0/max(mse,1e-12)))
    return ld, unet_norm, psnr

# v2 structural features
def gini(x):
    x=np.sort(np.asarray(x,dtype=np.float64)); n=len(x); s=np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0

def structural_features_v2(profile_raw):
    p = np.asarray(profile_raw, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax>dmin else p.copy()
    pp = float(np.argmax(pn))/L
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    peaks, props = find_peaks(pn, prominence=0.1)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks)>0 else 0.0
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "top_prominence": top_prom, "n_peaks": len(peaks)}

def distance_v2(fa, fb, pa=None, pb=None):
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"], fa["spread"]-fb["spread"]])
    d_shape = 0.0
    if pa is not None and pb is not None:
        rho,_ = spearmanr(np.asarray(pa), np.asarray(pb))
        d_shape = 1.0-float(rho) if not np.isnan(rho) else 1.0
    d_total = float(np.linalg.norm([d_pp, d_shape, d_mag]))
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_shape": d_shape, "D_mag": d_mag}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=19)
    ap.add_argument("--unet-path", type=str, default=None,
                    help="Path to custom UNet cache directory (e.g., for RV)")
    ap.add_argument("--model-name", type=str, default=None,
                    help="Model display name (e.g., 'RV')")
    args = ap.parse_args()
    images = COCO_VAL[:args.images]

    print(f"Images: {len(images)}")

    # Load SD 1.5 pipeline
    print("\nLoading SD 1.5 pipeline...")
    pipe15 = StableDiffusionPipeline.from_pretrained(MODEL_15, local_files_only=True, torch_dtype=DTYPE).to(DEVICE)
    pipe15.scheduler = DDIMScheduler.from_config(pipe15.scheduler.config)
    targets = discover_targets(pipe15.unet)
    print(f"  Targets: {len(targets)} layers")

    # --- SD 1.5 fingerprint ---
    print("\n=== SD 1.5 Fingerprint ===")
    hooker15 = Hooker(pipe15.unet); hooker15.reg(targets)
    pe15 = enc_empty(pipe15)
    per_img_15 = []
    for imp in tqdm(images, desc="  sd15"):
        ld, un, psnr = diagnose_single(pipe15, pe15, hooker15, targets, imp)
        per_img_15.append(ld)
    hooker15.rm()
    # Aggregate per-image
    pi_feats_15 = []
    pi_profiles_15 = []
    for ld in per_img_15:
        ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
        profile = np.array([v for _,v in ordered])
        pi_feats_15.append(structural_features_v2(profile))
        pi_profiles_15.append(profile)
    feat15 = {k: float(np.mean([pi[k] for pi in pi_feats_15])) for k in pi_feats_15[0]}
    mean_prof_15 = np.mean(pi_profiles_15, axis=0)
    dmin, dmax = mean_prof_15.min(), mean_prof_15.max()
    mean_norm_15 = (mean_prof_15-dmin)/(dmax-dmin) if dmax>dmin else mean_prof_15.copy()
    print(f"  pp={feat15['peak_position']:.4f} conc={feat15['concentration']:.4f} sp={feat15['spread']:.4f}")

    # --- Custom UNet fingerprint ---
    if args.unet_path:
        model_label = args.model_name or Path(args.unet_path).parent.parent.name
        print(f"\n=== {model_label} UNet (load + swap) ===")
        unet_custom = UNet2DConditionModel.from_pretrained(
            args.unet_path, local_files_only=True, torch_dtype=DTYPE).to(DEVICE)
        pipe_custom = copy.deepcopy(pipe15)
        pipe_custom.unet = unet_custom
        targets_custom = discover_targets(pipe_custom.unet)
        assert targets_custom == targets, f"Layer mismatch: expected {len(targets)}, got {len(targets_custom)}"
        hooker_custom = Hooker(pipe_custom.unet); hooker_custom.reg(targets)
        pe_custom = enc_empty(pipe_custom)
        per_img_custom = []
        for imp in tqdm(images, desc=f"  {model_label}"):
            ld, un, psnr = diagnose_single(pipe_custom, pe_custom, hooker_custom, targets, imp)
            per_img_custom.append(ld)
        hooker_custom.rm()

        pi_feats_custom = []
        pi_profiles_custom = []
        for ld in per_img_custom:
            ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
            profile = np.array([v for _,v in ordered])
            pi_feats_custom.append(structural_features_v2(profile))
            pi_profiles_custom.append(profile)
        feat_custom = {k: float(np.mean([pi[k] for pi in pi_feats_custom])) for k in pi_feats_custom[0]}
        mean_prof_custom = np.mean(pi_profiles_custom, axis=0)
        dmin, dmax = mean_prof_custom.min(), mean_prof_custom.max()
        mean_norm_custom = (mean_prof_custom-dmin)/(dmax-dmin) if dmax>dmin else mean_prof_custom.copy()
        print(f"  pp={feat_custom['peak_position']:.4f} conc={feat_custom['concentration']:.4f} sp={feat_custom['spread']:.4f}")

        del pipe_custom, unet_custom; torch.cuda.empty_cache()
        hooker15.rm(); del pipe15; torch.cuda.empty_cache()

        # --- D_s(SD1.5, custom) ---
        dd = distance_v2(feat15, feat_custom, mean_norm_15, mean_norm_custom)
        print(f"\n{'='*60}")
        print(f"D_s(SD1.5, {model_label}) — v2 metric")
        print(f"{'='*60}")
        print(f"  D_total    = {dd['D_total']:.6f}")
        print(f"  D_peak_pos = {dd['D_peak_pos']:.6f}")
        print(f"  D_shape    = {dd['D_shape']:.6f}")
        print(f"  D_mag      = {dd['D_mag']:.6f}")

        NOISE_P95 = 0.016258; INTER_ARCH_MIN = 0.249
        print(f"\nC1 Decision Rule:")
        print(f"  Noise floor p95   = {NOISE_P95:.6f}")
        print(f"  Min inter-arch    = {INTER_ARCH_MIN:.4f}")
        if dd['D_total'] < NOISE_P95:
            print(f"  ✓ CHECKPOINT-INVARIANT: D_s < noise floor")
        elif dd['D_total'] < INTER_ARCH_MIN:
            print(f"  ~ WEAK VARIATION")
        else:
            print(f"  ✗ C1 TROUBLE")

        summary = {
            "models": {"sd15": MODEL_15, "custom": str(args.unet_path)},
            "images": len(images),
            "sd15_features": feat15,
            "custom_features": feat_custom,
            "D_s": dd,
            "decision": {"D_total": dd["D_total"], "noise_floor_p95": NOISE_P95,
                         "min_inter_arch_Ds": INTER_ARCH_MIN,
                         "verdict": "checkpoint_invariant" if dd["D_total"] < NOISE_P95 else
                                    "weak_variation" if dd["D_total"] < INTER_ARCH_MIN else "trouble"}
        }
        label = args.model_name or "custom"
        sp = OUT_DIR / f"{label.lower()}_fingerprint.json"
        with open(sp, "w") as f: json.dump(summary, f, indent=2)
        print(f"\nSaved to {sp}")
        return

    # Original code path: SD 1.4 fingerprint
    hooker14 = Hooker(pipe14.unet); hooker14.reg(targets)
    pe14 = enc_empty(pipe14)
    per_img_14 = []
    for imp in tqdm(images, desc="  sd14"):
        ld, un, psnr = diagnose_single(pipe14, pe14, hooker14, targets, imp)
        per_img_14.append(ld)
    hooker14.rm()

    pi_feats_14 = []
    pi_profiles_14 = []
    for ld in per_img_14:
        ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
        profile = np.array([v for _,v in ordered])
        pi_feats_14.append(structural_features_v2(profile))
        pi_profiles_14.append(profile)
    feat14 = {k: float(np.mean([pi[k] for pi in pi_feats_14])) for k in pi_feats_14[0]}
    mean_prof_14 = np.mean(pi_profiles_14, axis=0)
    dmin, dmax = mean_prof_14.min(), mean_prof_14.max()
    mean_norm_14 = (mean_prof_14-dmin)/(dmax-dmin) if dmax>dmin else mean_prof_14.copy()
    print(f"  pp={feat14['peak_position']:.4f} conc={feat14['concentration']:.4f} sp={feat14['spread']:.4f}")

    del pipe14, pipe15, unet14; torch.cuda.empty_cache()

    # --- D_s(SD1.4, SD1.5) ---
    dd = distance_v2(feat15, feat14, mean_norm_15, mean_norm_14)
    print(f"\n{'='*60}")
    print(f"D_s(SD1.4, SD1.5) — v2 metric")
    print(f"{'='*60}")
    print(f"  D_total    = {dd['D_total']:.6f}")
    print(f"  D_peak_pos = {dd['D_peak_pos']:.6f}")
    print(f"  D_shape    = {dd['D_shape']:.6f}")
    print(f"  D_mag      = {dd['D_mag']:.6f}")

    NOISE_P95 = 0.016258
    INTER_ARCH_MIN = 0.249
    print(f"\nC1 Decision Rule:")
    print(f"  Noise floor p95   = {NOISE_P95:.6f}")
    print(f"  Min inter-arch    = {INTER_ARCH_MIN:.4f}")
    if dd['D_total'] < NOISE_P95:
        print(f"  ✓ CHECKPOINT-INVARIANT: D_s < noise floor")
    elif dd['D_total'] < INTER_ARCH_MIN:
        ratio = dd['D_total'] / INTER_ARCH_MIN
        print(f"  ~ WEAK VARIATION: noise_floor < D_s < inter_arch_min (ratio={ratio:.3f})")
    else:
        print(f"  ✗ C1 TROUBLE: D_s >= inter_arch_min")

    # Save
    summary = {
        "models": {"sd14": "CompVis/stable-diffusion-v1-4 (UNet only)", "sd15": MODEL_ID},
        "images": len(images),
        "sd15_features": feat15,
        "sd14_features": feat14,
        "D_s": dd,
        "decision": {
            "D_total": dd["D_total"],
            "noise_floor_p95": NOISE_P95,
            "min_inter_arch_Ds": INTER_ARCH_MIN,
            "verdict": "checkpoint_invariant" if dd["D_total"] < NOISE_P95 else
                       "weak_variation" if dd["D_total"] < INTER_ARCH_MIN else "trouble"
        }
    }
    sp = OUT_DIR / "sd14_fingerprint.json"
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nSaved to {sp}")


if __name__ == "__main__":
    main()
