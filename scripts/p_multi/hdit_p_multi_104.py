"""
HunyuanDiT P-multi@104 — Definition 1 multi-timestep protocol.

Noise schedule: v-prediction DDIM.
  v = √ᾱ·ε − √(1−ᾱ)·x_0  ⇔  x_0 = √ᾱ·z − √(1−ᾱ)·v,  ε = √(1−ᾱ)·z + √ᾱ·v
Reference at t: z_ref = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε  (DDPM forward, same as UNet case)
Comparison: recon trajectory z_recon[k] at matched t.
Drift: mean over K of L2(f_ref − f_recon), K = relative {0,1,2,mid-1,mid,mid+1,n-3,n-2,n-1}.

Hooks: 40 transformer blocks (blocks.0..blocks.39), raw output hidden_states.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import HunyuanDiTPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from layer_order import natural_key, layer_hash

DEVICE = "cuda"; DTYPE = torch.float16
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/dit_phase1")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50; SEED = 42

print(f"H-DiT P-multi@104: {len(IMAGES)} images, {NUM_STEPS} DDIM v_pred steps")
pipe = HunyuanDiTPipeline.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, prediction_type="v_prediction")
pipe.vae.to(torch.float32)

# Encode empty prompt (dual encoder: CLIP + T5)
prompt_embeds, _, prompt_attn, _ = pipe.encode_prompt(
    prompt="", device=DEVICE, dtype=DTYPE, num_images_per_prompt=1, do_classifier_free_guidance=False)
prompt_embeds_2, _, prompt_attn_2, _ = pipe.encode_prompt(
    prompt="", device=DEVICE, dtype=DTYPE, num_images_per_prompt=1,
    do_classifier_free_guidance=False, text_encoder_index=1)

# Hook targets: blocks.0..blocks.39
targets = [f"blocks.{i}" for i in range(40)]
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
    out = pipe.transformer(
        hidden_states=z,
        encoder_hidden_states=prompt_embeds,
        text_embedding_mask=prompt_attn,
        encoder_hidden_states_t5=prompt_embeds_2,
        text_embedding_mask_t5=prompt_attn_2,
        timestep=t_tensor, return_dict=False)[0]
    return out[:, :4] if out.shape[1] >= 8 else out

def v_to_eps_x0(v, z, alpha):
    """v-pred → eps, x0"""
    eps = (1-alpha).sqrt()*z + alpha.sqrt()*v
    x0 = alpha.sqrt()*z - (1-alpha).sqrt()*v
    return eps, x0

def ddim_inv(lat):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=lat.clone(); ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]
            v=forward(z,tc)
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            eps,x0 = v_to_eps_x0(v,z,ac)
            z = an.sqrt()*x0 + (1-an).sqrt()*eps
    return z

def ddim_recon_traj(noise):
    s=pipe.scheduler; s.set_timesteps(NUM_STEPS,device=DEVICE); ts=s.timesteps
    z=noise.clone(); traj=[z.clone()]
    with torch.no_grad():
        for t in ts:
            v=forward(z,t.item())
            eps,x0 = v_to_eps_x0(v,z,s.alphas_cumprod[t])
            z = s.step(v, t, z).prev_sample
            traj.append(z.clone())
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
failed_imgs = []
for img_path in tqdm(IMAGES, desc="H-DiT P-multi@104"):
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
        failed_imgs.append(nm)
        continue

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

# Filter layers with data (skip layers missing from failed images)
valid_layers = [ln for ln in canonical if len(per_img[ln]) > 0]
profile = [float(np.mean(list(per_img[ln].values()))) for ln in valid_layers]
canonical = valid_layers
if failed_imgs:
    print(f"  Skipped {len(failed_imgs)} failed images: {failed_imgs[:5]}...")
assert not np.isnan(profile).any(), "NaN in profile"

print(f"\nH-DiT P-multi@104:")
print(f"  pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} n_peaks={len(peaks)}")
print(f"  peak={canonical[idx]} (idx={idx}/{L})")
print(f"  Old P-t0: pp=0.500 conc=0.604 sp=0.592  [loose check: alarm if >2×]")

summary = {
    "protocol_id": "P-multi-v1", "n_images": len(IMAGES), "steps": NUM_STEPS,
    "K_indices": list(K), "seed": SEED,
    "canonical_layers": list(canonical),
    "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical[idx]},
    "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    "layer_list_hash": layer_hash("H-DiT", list(targets)),
}
out = OUT_DIR/"layer_drift_summary_p_multi_104.json"
with open(out,"w") as f: json.dump(summary,f,indent=2)
print(f"Saved: {out}")
