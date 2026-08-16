"""
SD3.5 P-multi@104 — Definition 1 multi-timestep protocol (Euler flow).

Rectified flow forward: z_t = (1-σ_t)·x_0 + σ_t·ε  (σ from FlowMatchEuler scheduler)
  Reference at t: z_ref = (1-σ)·z_0 + σ·ε → transformer(z_ref, t) → f_ref
  Recon at t:     z_recon from Euler reconstruction trajectory at matched t → f_recon
  Drift: mean over K of L2(f_ref - f_recon), K = {0,1,2,24,25,26,47,48,49}/50
Hooks: 24 JointTransformerBlocks, image hidden_states = output[1].
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from layer_order import natural_key, layer_hash

DEVICE = "cuda"; DTYPE = torch.float16
MODEL_PATH = "/home/hiaskc/.cache/huggingface/hub/models/stabilityai--stable-diffusion-3.5-medium/snapshots/master"
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/sd35_phase1")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50; SEED = 42

print(f"SD3.5 P-multi@104: {len(IMAGES)} images, {NUM_STEPS} Euler flow steps")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, local_files_only=True,
    tokenizer_3=None, text_encoder_3=None)
pipe.enable_model_cpu_offload()
print(f"[MEM] after load+offload: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved", flush=True)

# Empty prompt encoding (zero-padded T5 placeholder, from sd35_phase1_diagnostics.py)
def encode_empty(pipe):
    device = pipe._execution_device
    joint_dim = pipe.transformer.config.joint_attention_dim
    tok1 = pipe.tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out1 = pipe.text_encoder(tok1["input_ids"].to(device), output_hidden_states=True)
    h1 = out1.hidden_states[-2].to(dtype=pipe.text_encoder.dtype)
    p1 = out1.text_embeds.to(dtype=pipe.text_encoder.dtype)
    tok2 = pipe.tokenizer_2("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out2 = pipe.text_encoder_2(tok2["input_ids"].to(device), output_hidden_states=True)
    h2 = out2.hidden_states[-2].to(dtype=pipe.text_encoder_2.dtype)
    p2 = out2.text_embeds.to(dtype=pipe.text_encoder_2.dtype)
    clip = torch.cat([h1, h2], dim=-1)
    clip = torch.nn.functional.pad(clip, (0, joint_dim - clip.shape[-1]))
    t5 = torch.zeros(1, 256, joint_dim, device=device, dtype=pipe.transformer.dtype)
    enc = torch.cat([clip, t5], dim=1).to(pipe.transformer.dtype)
    pooled = torch.cat([p1, p2], dim=-1).to(pipe.transformer.dtype)
    return enc, pooled

enc, pooled = encode_empty(pipe)
print(f"[MEM] after encode_empty: {torch.cuda.memory_allocated()/1e9:.2f}GB alloc, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved", flush=True)

class BlockHooker:
    def __init__(s):
        s.f={}; s.h=[]
        for i, block in enumerate(pipe.transformer.transformer_blocks):
            def fn(module, inp, out, idx=i):
                if isinstance(out, tuple) and len(out)==2:
                    hs = out[1]
                else:
                    hs = out
                s.f[f"block_{idx}"] = hs.detach().float().cpu()
            s.h.append(block.register_forward_hook(fn))
    def clear(s): s.f.clear()
    def remove(s):
        for h in s.h: h.remove()
        s.h.clear()

hooker = BlockHooker()

def get_K(n):
    if n<=6: return list(range(n))
    k=[0,1,2,n//2-1,n//2,n//2+1,n-3,n-2,n-1]
    return sorted(set(max(0,min(n-1,i)) for i in k))
K = get_K(NUM_STEPS)
print(f"K = {K}")

def load_enc(path):
    img = Image.open(path).convert("RGB").resize((1024,1024))
    px = pipe.image_processor.preprocess(img).to("cuda", torch.float16)
    with torch.no_grad():
        lat = pipe.vae.encode(px).latent_dist.sample()
        lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return lat

_mem_count = [0]

per_img = defaultdict(dict)
failed = []

for img_path in tqdm(IMAGES, desc="SD3.5 P-multi@104"):
    nm = img_path.stem
    try:
        lat = load_enc(img_path)
        scheduler = pipe.scheduler
        scheduler.set_timesteps(NUM_STEPS, device="cuda")
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas

        @torch.no_grad()
        def fwd(z, t):
            return pipe.transformer(hidden_states=z, encoder_hidden_states=enc,
                                    pooled_projections=pooled, timestep=t,
                                    return_dict=False)[0]

        # Euler inversion (forward flow)
        torch.manual_seed(SEED)
        eps_ref = torch.randn_like(lat)
        z = lat.clone()
        for i, t in enumerate(timesteps):
            dt = sigmas[i+1] - sigmas[i]
            v = fwd(z, t.unsqueeze(0).expand(z.shape[0]))
            z = z + dt*v

        # Euler reconstruction (backward flow), record trajectory at K
        traj = {}
        for i in range(len(timesteps)-1, -1, -1):
            t = timesteps[i]
            dt = sigmas[i+1] - sigmas[i]
            v = fwd(z, t.unsqueeze(0).expand(z.shape[0]))
            z = z - dt*v
            if i in K:
                traj[i] = z.clone()

        # Compare at K
        lds = defaultdict(list)
        for kidx in K:
            sig = sigmas[kidx]
            z_ref = (1-sig)*lat + sig*eps_ref
            t = timesteps[kidx]
            fwd(z_ref, t.unsqueeze(0).expand(z_ref.shape[0]))
            rf = dict(hooker.f)
            hooker.clear()
            z_recon = traj[kidx]
            fwd(z_recon, t.unsqueeze(0).expand(z_recon.shape[0]))
            rc = dict(hooker.f)
            hooker.clear()
            for ln in rf:
                if ln in rc:
                    lds[ln].append(float(torch.norm(rf[ln]-rc[ln], p=2).item()))
            del rf, rc
            hooker.clear()

        for ln in lds:
            if lds[ln]: per_img[ln][nm] = float(np.mean(lds[ln]))
        del lat, z, eps_ref, traj
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAIL {nm}: {e}")
        failed.append(nm)

hooker.remove(); del pipe; torch.cuda.empty_cache()

canonical = [f"block_{i}" for i in range(24)]
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

print(f"\nSD3.5 P-multi@104:")
print(f"  pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} n_peaks={len(peaks)}")
print(f"  peak={canonical[idx]} (idx={idx}/{L})")
print(f"  Failed: {len(failed)}")
print(f"  Old P-t0: pp=0.917 (block_22)  [loose check: alarm if >2x]")

summary = {
    "protocol_id": "P-multi-v1", "n_images": len(IMAGES)-len(failed),
    "steps": NUM_STEPS, "K_indices": list(K), "seed": SEED,
    "failed_images": failed,
    "canonical_layers": list(canonical),
    "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical[idx]},
    "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    "layer_list_hash": layer_hash("SD3.5", list(canonical)),
}
out = OUT_DIR/"layer_drift_summary_p_multi_104.json"
with open(out,"w") as f: json.dump(summary,f,indent=2)
print(f"Saved: {out}")
