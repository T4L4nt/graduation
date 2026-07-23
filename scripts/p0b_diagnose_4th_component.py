"""
Quick diagnostic: identify the 4th D_s component causing D_s=1.0
despite D_peak≈0, D_shape≈0, D_mag≈0.

Hypothesis: peak_count changes from 2→1 (or similar) when a small
ripple crosses the prominence=0.1 threshold, causing the 4-feature
Euclidean distance to jump from ~0.005 to ~1.0 on that dimension.

Runs baseline + 5 suspicious (epsilon, seed) combos on 1 image,
extracts raw drift profiles, identifies which feature(s) cause the jump.
"""

import copy, json, sys
from pathlib import Path
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda"; DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-v1-5"
TEST_IMG = sorted(Path("data/coco_val").glob("*.jpg"))[0]
OUT_DIR = Path("outputs/p0b_cross_checkpoint")
NUM_STEPS = 50

# Suspicious combos from v3 results
SUSPICIOUS = [
    (3e-5, 0, "cliff_seed0"),
    (3e-5, 1, "cliff_seed1"),
    (3e-5, 2, "cliff_seed2"),
    (5e-5, 0, "clean_seed0"),
    (1e-4, 0, "all_broken"),
]


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
    def __init__(self, unet): self.unet = unet; self.features = {}; self.handles = []
    def _hook_fn(self, name):
        def fn(m, i, o): self.features[name] = o.detach().float().cpu()
        return fn
    def register(self, targets):
        self.remove()
        for name, module in self.unet.named_modules():
            if name in targets:
                self.handles.append(module.register_forward_hook(self._hook_fn(name)))
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear(); self.features.clear()

def is_weight_param(name, param):
    if "bias" in name: return False
    if "norm" in name.lower(): return False
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
            noise = torch.randn_like(w_flat) * epsilon * max(w_norm, 1e-8)
            param.data.copy_((w_flat + noise).view(param.shape).to(param.dtype))
    return unet_pert

def encode_empty(pipe):
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad(): return pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

def ddim_inv(pipe, latents, pe, N):
    s = pipe.scheduler; s.set_timesteps(N, device=DEVICE)
    ts = s.timesteps; z = latents.clone(); ext = ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn = ext[i],ext[i-1]; npred = pipe.unet(z,tc,encoder_hidden_states=pe).sample
            ac,an = s.alphas_cumprod[tc], s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc=(1-ac).sqrt(); sn=(1-an).sqrt()
            z = c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon(pipe, noise, pe, N):
    s=pipe.scheduler; s.set_timesteps(N, device=DEVICE); z=noise.clone()
    with torch.no_grad():
        for t in s.timesteps:
            npred=pipe.unet(z,t,encoder_hidden_states=pe).sample
            z=s.step(npred,t,z).prev_sample
    return z

def load_encode(pipe, path, size=512):
    img = Image.open(path).convert("RGB").resize((size,size))
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    t=2*t-1
    with torch.no_grad():
        l=pipe.vae.encode(t).latent_dist.sample()
        return l*pipe.vae.config.scaling_factor

def diagnose_single(pipe, pe, hooker, targets):
    latent = load_encode(pipe, TEST_IMG)
    z_inv = ddim_inv(pipe, latent, pe, NUM_STEPS)
    hooker.features.clear()
    with torch.no_grad(): pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    inv_f = {k:v.clone() for k,v in hooker.features.items()}
    z_recon = ddim_recon(pipe, z_inv, pe, NUM_STEPS)
    hooker.features.clear()
    with torch.no_grad(): pipe.unet(z_recon, pipe.scheduler.timesteps[0], encoder_hidden_states=pe).sample
    recon_f = {k:v.clone() for k,v in hooker.features.items()}
    ld = {}
    unet_out_inv = None; unet_out_recon = None
    for ln in targets:
        if ln in inv_f and ln in recon_f:
            drift = torch.norm(inv_f[ln]-recon_f[ln], p=2).item()
            ld[ln] = drift
        # Capture UNet output norm from the final timestep (last layer)
    ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
    profile = np.array([v for _,v in ordered])
    dmin,dmax = profile.min(), profile.max()
    norm = (profile-dmin)/(dmax-dmin) if dmax>dmin else profile.copy()

    # Detailed peak analysis
    peaks, props = find_peaks(norm, prominence=0.1)
    peak_vals = norm[peaks]
    peak_layers = [ordered[p][0] for p in peaks]
    print(f"    Peaks found: {len(peaks)} at {peak_layers}")
    print(f"    Peak prominences: {props['prominences'] if 'prominences' in props else 'N/A'}")
    print(f"    Peak values: {['%.4f'%v for v in peak_vals]}")
    print(f"    Max value: {norm.max():.4f} at layer {ordered[np.argmax(norm)][0]}")

    # Compute all 4 structural features manually
    L = len(norm)
    pp = float(np.argmax(norm))/L
    pc = int(np.sum(norm[peaks] > 0.5))
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(norm)[-k:]
    conc = float(np.sum(norm[top])/np.sum(norm)) if np.sum(norm)>0 else 0.0
    # Gini
    x = np.sort(norm)
    n=len(x); s=np.sum(x)
    gini_v = float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0

    print(f"    Features: pp={pp:.4f} pc={pc} conc={conc:.4f} gini={gini_v:.4f}")
    print(f"    Prominence of peak[0]: {props['prominences'][0]:.4f}" if len(peaks)>0 and 'prominences' in props else "    No peaks")

    return {"layer_names": [k for k,_ in ordered], "profile_raw": profile.tolist(),
            "profile_norm": norm.tolist(), "features": {"peak_position": pp, "peak_count": pc,
            "concentration": conc, "spread": gini_v}, "peaks": len(peaks), "peak_values": peak_vals.tolist()}


def structural_distance(fa, fb):
    keys = ["peak_position","peak_count","concentration","spread"]
    return float(np.linalg.norm(np.array([fa[k] for k in keys]) - np.array([fb[k] for k in keys])))

def decomposed_distance(pa, pb, fa, fb):
    d_peak = abs(fa["peak_position"] - fb["peak_position"])
    from scipy.stats import spearmanr
    rho,_ = spearmanr(pa, pb)
    d_shape = 1.0-float(rho) if not np.isnan(rho) else 1.0
    d_peak_cnt = abs(fa["peak_count"] - fb["peak_count"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"], fa["spread"]-fb["spread"]])
    # Show ALL components including peak_count
    d_total_4 = float(np.linalg.norm([d_peak, d_peak_cnt, d_shape, d_mag]))
    return {"D_peak_pos": d_peak, "D_peak_cnt": d_peak_cnt, "D_shape": d_shape, "D_mag": d_mag, "D_total_4comp": d_total_4}


print("Loading SD 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
targets = discover_hook_targets(pipe.unet)
hooker = UNetFeatureHooker(pipe.unet); hooker.register(targets)
pe = encode_empty(pipe)
original_state = copy.deepcopy(pipe.unet.state_dict())
print(f"  {len(targets)} layers")

# Baseline
print("\n=== BASELINE ===")
b = diagnose_single(pipe, pe, hooker, targets)
bf = b["features"]

# Analyze each suspicious combo
results = {}
for eps, seed, label in SUSPICIOUS:
    print(f"\n=== {label} (eps={eps:.1e}, seed={seed}) ===")
    pipe.unet.load_state_dict(copy.deepcopy(original_state))
    pipe.unet = perturb_weights(pipe.unet, eps, seed=42+seed*100+int(eps*1e10))
    hooker.remove(); hooker = UNetFeatureHooker(pipe.unet); hooker.register(targets)
    r = diagnose_single(pipe, pe, hooker, targets)
    ds = structural_distance(bf, r["features"])
    dd = decomposed_distance(b["profile_norm"], r["profile_norm"], bf, r["features"])
    print(f"    D_s(4-feature) = {ds:.6f}")
    print(f"    D_peak_pos={dd['D_peak_pos']:.6f}  D_peak_cnt={dd['D_peak_cnt']:.1f}  D_shape={dd['D_shape']:.6f}  D_mag={dd['D_mag']:.6f}")
    results[label] = {"result": r, "D_s": ds, "D_decomposed": dd}
    # Save profiles for later plotting
    results[label]["profiles"] = {"baseline_norm": b["profile_norm"], "baseline_raw": b["profile_raw"],
                                   "pert_norm": r["profile_norm"], "pert_raw": r["profile_raw"]}

hooker.remove(); del pipe; torch.cuda.empty_cache()

# Save for plotting
with open(OUT_DIR / "4th_component_diagnosis.json", "w") as f:
    json.dump({k: {"D_s": v["D_s"], "D_decomposed": {kk: float(vv) for kk,vv in v["D_decomposed"].items()},
                   "profiles": {kk: vv for kk,vv in v["profiles"].items()},
                   "baseline_features": bf}
              for k,v in results.items()}, f, indent=2)

print("\nDiagnosis saved.")
PYEOF