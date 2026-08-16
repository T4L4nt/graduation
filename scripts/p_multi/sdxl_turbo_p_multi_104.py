"""
SDXL-Turbo P-multi@104 — Definition 1 compliant multi-timestep protocol.

Protocol (from PROTOCOL_MANIFEST.md, P-multi-v1):
  K = relative rule {first 3, mid 3, last 3} of 50 DDIM steps = {0,1,2,24,25,26,47,48,49}
  Ref: DDPM-forward noise on original latent at t → unet(z_ref, t) → f_ref[t]
  Recon: inversion→reconstruction trajectory → unet(z_recon[k], t_k) → f_recon[t]
  Drift: mean over K of L2(f_ref - f_recon)
  Seed: torch.manual_seed(42) per image
  dtype: fp16 (VAE fp32)
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from layer_order import unet_topo_key, layer_hash

DEVICE = "cuda"; DTYPE = torch.float16
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/sdxl_phase1")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50; SEED = 42

print(f"SDXL-Turbo P-multi@104: {len(IMAGES)} images, {NUM_STEPS} DDIM steps")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.vae.to(torch.float32)

# Encode empty prompt (dual encoder)
(prompt_embeds, _, pooled, _) = pipe.encode_prompt(
    prompt="", prompt_2="", device=DEVICE, num_images_per_prompt=1,
    do_classifier_free_guidance=False)
time_ids = torch.tensor([[1024,1024,0,0,1024,1024]], device=DEVICE, dtype=DTYPE)
added_cond = {"text_embeds": pooled, "time_ids": time_ids}

# Hook targets (same discovery as phase1)
def discover_targets(unet):
    tgt = []
    for n,_ in unet.named_modules():
        p=n.split(".")
        if "resnets" in p:
            idx=p.index("resnets")
            if len(p)==idx+2 and p[-1].isdigit(): tgt.append(n)
        if "transformer_blocks" in p:
            idx=p.index("transformer_blocks")
            if len(p)==idx+2 and p[-1]=="0": tgt.append(n)
    return sorted(tgt)

targets = discover_targets(pipe.unet)
canonical = sorted(targets, key=unet_topo_key)
print(f"Hook targets: {len(targets)} layers")

class Hooker:
    def __init__(s, unet, tgt):
        s.f={}; s.h=[]
        for n in tgt:
            m=unet
            for t in n.split("."):
                try: m=getattr(m,t)
                except: m=None; break
            if m: s.h.append(m.register_forward_hook(lambda m,i,o,n=n: s._f(n,o)))
    def _f(s,n,o):
        if isinstance(o,tuple): o=o[0]
        if o.dim()==3: o=o.mean(1,keepdim=True)
        s.f[n]=o.detach().cpu()
    def clear(s): s.f.clear()
    def remove(s):
        for h in s.h: h.remove()
        s.h.clear()

hooker = Hooker(pipe.unet, targets)

def ddim_inv(lat):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=lat.clone(); ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]
            npred=pipe.unet(z,tc,encoder_hidden_states=prompt_embeds,added_cond_kwargs=added_cond).sample
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc,sn=(1-ac).sqrt(),(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon_traj(noise):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=noise.clone(); traj=[z.clone()]
    with torch.no_grad():
        for t in ts:
            npred=pipe.unet(z,t,encoder_hidden_states=prompt_embeds,added_cond_kwargs=added_cond).sample
            z=s.step(npred,t,z).prev_sample; traj.append(z.clone())
    return traj

def get_K(n):
    if n<=6: return list(range(n))
    k=[0,1,2,n//2-1,n//2,n//2+1,n-3,n-2,n-1]
    return sorted(set(max(0,min(n-1,i)) for i in k))
K = get_K(NUM_STEPS)
print(f"K = {K}")

def load_enc(path):
    img=Image.open(path).convert("RGB").resize((1024,1024),Image.LANCZOS)
    t=transforms.ToTensor()(img).unsqueeze(0).to(DEVICE,dtype=torch.float32)
    t=2*t-1
    with torch.no_grad():
        lat=pipe.vae.encode(t).latent_dist.sample()*pipe.vae.config.scaling_factor
    return lat.to(DTYPE)

per_img = defaultdict(dict)
for img_path in tqdm(IMAGES, desc="SDXL-Turbo P-multi@104"):
    nm = img_path.stem
    try:
        lat = load_enc(img_path)
        inv_lat = ddim_inv(lat)
        traj = ddim_recon_traj(inv_lat)
        s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
        ts = s.timesteps
        
        lds = defaultdict(list)
        torch.manual_seed(SEED); np.random.seed(SEED)
        for kidx in K:
            t = ts[kidx]; alpha = s.alphas_cumprod[t]
            z_ref = alpha.sqrt()*lat + (1-alpha).sqrt()*torch.randn_like(lat)
            z_recon = traj[kidx]
            hooker.clear()
            pipe.unet(z_ref, t.to(DEVICE), encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond).sample
            rf = dict(hooker.f)
            hooker.clear()
            pipe.unet(z_recon, t.to(DEVICE), encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond).sample
            rf2 = dict(hooker.f)
            for ln in targets:
                if ln in rf and ln in rf2:
                    lds[ln].append(float(torch.norm(rf[ln].float()-rf2[ln].float(),p=2).item()))
        for ln in targets:
            if lds[ln]: per_img[ln][nm] = float(np.mean(lds[ln]))
    except Exception as e:
        print(f"  FAIL {nm}: {e}")

hooker.remove(); del pipe; torch.cuda.empty_cache()

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

# Assertions
assert not np.isnan(profile).any(), "NaN in profile"
assert idx == int(pp*L) or abs(idx - round(pp*L)) <= 1, "pp*L not valid index"

print(f"\nSDXL-Turbo P-multi@104:")
print(f"  pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} n_peaks={len(peaks)}")
print(f"  peak={canonical[idx]} (idx={idx}/{L})")
print(f"  SDXL base 对照: pp=0.429 (M.R1), conc=0.567, sp=0.570")

summary = {
    "protocol_id": "P-multi-v1", "n_images": len(IMAGES), "steps": NUM_STEPS,
    "K_indices": list(K), "seed": SEED,
    "canonical_layers": list(canonical),
    "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical[idx]},
    "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    "layer_list_hash": layer_hash("SDXL", list(targets)),
}
out = OUT_DIR/"sdxl_turbo_p_multi_104.json"
with open(out,"w") as f: json.dump(summary,f,indent=2)
print(f"Saved: {out}")
