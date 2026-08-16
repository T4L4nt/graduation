"""
FLUX P-multi@104 — Definition 1 multi-timestep protocol, flow-matching adaptation.

Flow forward process: x_t = (1-t)·x_0 + t·ε  (t ∈ [0,1], ε ~ N(0,I))
  Reference at t: z_ref = (1-t)·z_0_packed + t·ε_seed → transformer(z_ref, t) → f_ref
  Recon at t:     z_recon from reconstruction trajectory at matched t → f_recon
  Drift: mean over K of L2(f_ref_hidden - f_recon_hidden), K = {0,1,2,24,25,26,47,48,49}/50
Hooks: 57 blocks (joint_0..18 + single_0..37), hidden (image token) features.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from flux_common import load_flux_pipeline, FluxFeatureExtractor
from layer_order import natural_key, layer_hash

DEVICE = "cuda"
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/phase9_flux_fp16")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50; SEED = 42

print(f"FLUX P-multi@104: {len(IMAGES)} images, {NUM_STEPS} Euler flow steps")
pipe = load_flux_pipeline(device=DEVICE, dtype=torch.float16, offload_t5=True)

def get_K(n):
    if n<=6: return list(range(n))
    k=[0,1,2,n//2-1,n//2,n//2+1,n-3,n-2,n-1]
    return sorted(set(max(0,min(n-1,i)) for i in k))
K = get_K(NUM_STEPS)
print(f"K = {K}")

extractor = FluxFeatureExtractor(pipe.transformer)
extractor.register_hooks()

def snapshot_hidden():
    """Return {block_name: hidden_features} for current hook state."""
    feats = extractor.get_features()
    return {name: f["hidden"] for name, f in feats.items() if "hidden" in f}

# Prompt embedding (empty prompt)
with torch.no_grad():
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(prompt="", prompt_2="", device=DEVICE)

per_img = defaultdict(dict)
failed = []

for img_path in tqdm(IMAGES, desc="FLUX P-multi@104"):
    nm = img_path.stem
    try:
        img = Image.open(str(img_path)).convert("RGB")
        with torch.no_grad():
            image_tensor = pipe.image_processor.preprocess(img, height=img.height, width=img.width)
            image_tensor = image_tensor.to(DEVICE, dtype=pipe.vae.dtype)
            z0_raw = pipe.vae.encode(image_tensor).latent_dist.sample(
                torch.Generator(device=DEVICE).manual_seed(SEED))
            z0_raw = (z0_raw - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        bs, nc, lh, lw = z0_raw.shape
        ph, pw = 2*(lh//2), 2*(lw//2)
        z0_raw = z0_raw[:, :, :ph, :pw]
        z0 = pipe._pack_latents(z0_raw, bs, nc, ph, pw)
        latent_image_ids = pipe._prepare_latent_image_ids(
            bs, ph//2, pw//2, DEVICE, pipe.transformer.dtype)

        guidance = torch.full([bs], 1.0, device=DEVICE, dtype=torch.float32) \
            if pipe.transformer.config.guidance_embeds else None

        # Scheduler-consistent sigma grid with dynamic shifting (mu from FluxPipeline)
        scheduler = pipe.scheduler
        seq_len = ph * pw // 4  # image tokens after packing
        base_shift, max_shift = 0.5, 1.15
        base_seq, max_seq = 256, 4096
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        mu = m * seq_len + b
        scheduler.set_timesteps(NUM_STEPS, device=DEVICE, mu=mu)
        sigmas = scheduler.sigmas          # len 51, descending, in [0,1]
        dt_vals = sigmas[1:] - sigmas[:-1] # negative (descending)

        def fwd(z, t):
            tt = torch.full((bs,), float(t), device=DEVICE, dtype=pipe.transformer.dtype)
            with torch.no_grad():
                return pipe.transformer(
                    hidden_states=z, encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled, timestep=tt,
                    img_ids=latent_image_ids, txt_ids=text_ids,
                    guidance=guidance, return_dict=True).sample

        # Reference features at K steps: flow-forward interpolation
        torch.manual_seed(SEED)
        eps_ref = torch.randn_like(z0)
        ref_feats = {}
        for kidx in K:
            sig = float(sigmas[kidx])
            z_ref = (1-sig)*z0 + sig*eps_ref
            fwd(z_ref, sigmas[kidx])
            ref_feats[kidx] = snapshot_hidden()

        # Euler inversion (forward flow): z += dt*v, dt<0 (sigmas descending)
        z_t = z0.clone()
        for i in range(NUM_STEPS):
            v = fwd(z_t, sigmas[i])
            z_t = z_t + dt_vals[i]*v

        # Euler reconstruction (backward flow): z -= dt*v
        z_r = z_t.clone()
        traj_recon = {}
        for i in range(NUM_STEPS-1, -1, -1):
            v = fwd(z_r, sigmas[i])
            z_r = z_r - dt_vals[i]*v
            if i in K:
                traj_recon[i] = z_r.clone()

        # Compare at K steps
        lds = defaultdict(list)
        for kidx in K:
            sig = float(sigmas[kidx])
            # Reference re-run (fresh hook state)
            z_ref = (1-sig)*z0 + sig*eps_ref
            fwd(z_ref, sigmas[kidx])
            rf = snapshot_hidden()
            z_recon = traj_recon[kidx]
            fwd(z_recon, sigmas[kidx])
            rc = snapshot_hidden()
            for ln in rf:
                if ln in rc:
                    lds[ln].append(float(torch.norm(rf[ln].float()-rc[ln].float(), p=2).item()))

        for ln in lds:
            if lds[ln]:
                per_img[ln][nm] = float(np.mean(lds[ln]))
        del z0, z_t, z_r, eps_ref
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAIL {nm}: {e}")
        failed.append(nm)

extractor.remove_hooks()
del pipe; torch.cuda.empty_cache()

# Aggregate
canonical = sorted(per_img.keys(), key=natural_key)
profile = [float(np.mean(list(per_img[ln].values()))) for ln in canonical]
p=np.asarray(profile,dtype=np.float64); L=len(p)
pn=(p-p.min())/(p.max()-p.min()) if p.max()>p.min() else p
idx=int(np.argmax(pn)); pp=idx/L
kk=max(1,int(np.ceil(0.2*L)))
conc=float(np.sum(pn[np.argsort(pn)[-kk:]])/np.sum(pn))
def gini(x):
    x=np.sort(np.asarray(x,dtype=np.float64))
    n=len(x); s=np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0
sp=float(gini(pn))
from scipy.signal import find_peaks
peaks,_=find_peaks(pn,prominence=0.1)

print(f"\nFLUX P-multi@104:")
print(f"  pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} n_peaks={len(peaks)}")
print(f"  peak={canonical[idx]} (idx={idx}/{L})")
print(f"  Failed: {len(failed)} images")
print(f"  Old P-tT: pp=0.368 (s2)  [loose check: alarm if >2x deviation]")

summary = {
    "protocol_id": "P-multi-v1", "n_images": len(IMAGES)-len(failed),
    "steps": NUM_STEPS, "K_indices": list(K), "seed": SEED,
    "failed_images": failed,
    "canonical_layers": list(canonical),
    "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical[idx]},
    "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    "layer_list_hash": layer_hash("FLUX", list(canonical)),
}
out = OUT_DIR/"flux_p_multi_104.json"
with open(out,"w") as f: json.dump(summary,f,indent=2)
print(f"Saved: {out}")
