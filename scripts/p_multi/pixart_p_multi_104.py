"""
PixArt-Σ P-multi@104 — Definition 1 multi-timestep protocol (DDIM).

Reference at t: z_ref = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε  (DDPM forward)
Recon at t:     z_recon from DDIM reconstruction trajectory at matched t
Drift: mean over K of L2(f_ref - f_recon), K = {0,1,2,24,25,26,47,48,49}/50
Hooks: 28 transformer blocks (transformer_blocks.0..27), raw output hidden_states.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import PixArtSigmaPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from layer_order import natural_key, layer_hash

DEVICE = "cuda"; DTYPE = torch.float16
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/p0b_cross_checkpoint")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50; SEED = 42

print(f"PixArt-Σ P-multi@104: {len(IMAGES)} images, {NUM_STEPS} DDIM steps")
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Encode empty prompt
(prompt_embeds, prompt_attn, _, _) = pipe.encode_prompt(
    prompt="", do_classifier_free_guidance=False, num_images_per_prompt=1, device=DEVICE)

# Hook targets: transformer_blocks.0..27
targets = [f"transformer_blocks.{i}" for i in range(28)]
canonical = sorted(targets, key=natural_key)
print(f"Hook targets: {len(targets)} transformer blocks")

class Hooker:
    def __init__(s, model, tgt):
        s.f={}; s.h=[]
        for n in tgt:
            m=model
            for t in n.split("."):
                try: m=getattr(m,t)
                except: m=None; break
            if m: s.h.append(m.register_forward_hook(lambda m,i,o,n=n: s._f(n,o)))
    def _f(s,n,o):
        if isinstance(o,tuple): o=o[0]
        s.f[n]=o.detach().float().cpu()
    def clear(s): s.f.clear()
    def remove(s):
        for h in s.h: h.remove()
        s.h.clear()

hooker = Hooker(pipe.transformer, targets)

def forward(z, t_int):
    t_tensor = torch.tensor([t_int], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        out = pipe.transformer(
            hidden_states=z, encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attn, timestep=t_tensor,
            return_dict=False)[0]
    if out.shape[1] >= 8:  # PixArt learn_sigma=True
        out = out[:, :4]
    return out

def ddim_inv(lat):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=lat.clone(); ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]
            npred=forward(z,tc)
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc,sn=(1-ac).sqrt(),(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon_traj(noise):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=noise.clone(); traj=[z.clone()]
    with torch.no_grad():
        for t in ts:
            npred=forward(z,t.item())
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
    t=transforms.ToTensor()(img).unsqueeze(0).to(DEVICE,dtype=DTYPE)
    t=2*t-1
    with torch.no_grad():
        lat=pipe.vae.encode(t).latent_dist.sample()*pipe.vae.config.scaling_factor
    return lat

per_img = defaultdict(dict)
failed = []
for img_path in tqdm(IMAGES, desc="PixArt P-multi@104"):
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
            forward(z_ref, t.item())
            rf = dict(hooker.f)
            hooker.clear()
            forward(z_recon, t.item())
            rf2 = dict(hooker.f)
            for ln in targets:
                if ln in rf and ln in rf2:
                    lds[ln].append(float(torch.norm(rf[ln]-rf2[ln],p=2).item()))
        for ln in targets:
            if lds[ln]: per_img[ln][nm] = float(np.mean(lds[ln]))
    except Exception as e:
        print(f"  FAIL {nm}: {e}")
        failed.append(nm)

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

print(f"\nPixArt-Σ P-multi@104:")
print(f"  pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} n_peaks={len(peaks)}")
print(f"  peak={canonical[idx]} (idx={idx}/{L})")
print(f"  Failed: {len(failed)}")
print(f"  Old P-t0: pp=0.964 (T27, censored)  [loose check: alarm if >2x]")

summary = {
    "protocol_id": "P-multi-v1", "n_images": len(IMAGES)-len(failed),
    "steps": NUM_STEPS, "K_indices": list(K), "seed": SEED,
    "failed_images": failed,
    "canonical_layers": list(canonical),
    "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical[idx]},
    "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    "layer_list_hash": layer_hash("PixArt-Sigma", list(targets)),
}
out = OUT_DIR/"pixart_p_multi_104.json"
with open(out,"w") as f: json.dump(summary,f,indent=2)
print(f"Saved: {out}")
