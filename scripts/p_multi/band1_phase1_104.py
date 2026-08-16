"""
Band 1 Phase1@104 — 4 SD1.5 variants.
Protocol: P-multi (matched to SD1.5 phase1@104).
Variants: SD1.4, RealisticVision, LCM-LoRA, RandText
"""
import json, sys, numpy as np, torch
from pathlib import Path; from collections import defaultdict
from PIL import Image; from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Talant/graduation/scripts"))
try:
    from layer_order import unet_topo_key, layer_hash
except:
    import re
    def unet_topo_key(name):
        parts = name.split(".")
        if name.startswith("down_blocks"): bt, bi = 0, int(parts[1])
        elif name.startswith("mid_block"): bt, bi = 1, 0
        else: bt, bi = 2, int(parts[1])
        if "resnets" in parts: st, si = (0 if bt != 1 else 1), parts.index("resnets"); sn = int(parts[si+1])
        elif "attentions" in parts: st, si = (1 if bt != 1 else 0), parts.index("attentions"); sn = int(parts[si+1])
        else: st, sn = 2, 0
        tn = 0
        if "transformer_blocks" in parts: tn = int(parts[parts.index("transformer_blocks")+1])
        return (bt, bi, st, sn, tn)
    def layer_hash(arch, names): return "n/a"

DEVICE = "cuda"; DTYPE = torch.float16; NUM_STEPS = 50; SEED = 42
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/p0b_cross_checkpoint/band1_phase1")

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

def ddim_inv(pipe, lat, pe, st):
    s=pipe.scheduler; s.set_timesteps(st,device=DEVICE); ts=s.timesteps
    z=lat.clone(); ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]
            npred=pipe.unet(z,tc,encoder_hidden_states=pe).sample
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc,sn=(1-ac).sqrt(),(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon_traj(pipe, noise, pe, st):
    s=pipe.scheduler; s.set_timesteps(st,device=DEVICE); ts=s.timesteps
    z=noise.clone(); traj=[z.clone()]
    with torch.no_grad():
        for t in ts:
            npred=pipe.unet(z,t,encoder_hidden_states=pe).sample
            z=s.step(npred,t,z).prev_sample; traj.append(z.clone())
    return traj

def get_K(n):
    if n<=6: return list(range(n))
    k=[0,1,2,n//2-1,n//2,n//2+1,n-3,n-2,n-1]
    return sorted(set(max(0,min(n-1,i)) for i in k))

K = get_K(NUM_STEPS)

def load_enc(pipe, path):
    img=Image.open(path).convert("RGB").resize((512,512))
    t=transforms.ToTensor()(img).unsqueeze(0).to(DEVICE,dtype=DTYPE)
    t=2*t-1
    with torch.no_grad():
        lat=pipe.vae.encode(t).latent_dist.sample()*pipe.vae.config.scaling_factor
    return lat

def measure_variant(name, pipe, pe, images):
    targets = discover_targets(pipe.unet)
    canonical = sorted(targets, key=unet_topo_key)
    h = Hooker(pipe.unet, targets)
    
    per_img = defaultdict(dict)
    for img_path in tqdm(images, desc=name):
        nm = img_path.stem
        try:
            lat = load_enc(pipe, img_path)
            inv_lat = ddim_inv(pipe, lat, pe, NUM_STEPS)
            traj = ddim_recon_traj(pipe, inv_lat, pe, NUM_STEPS)
            s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
            ts = s.timesteps
            
            lds = defaultdict(list)
            torch.manual_seed(SEED); np.random.seed(SEED)
            for kidx in K:
                t = ts[kidx]; alpha = s.alphas_cumprod[t]
                z_ref = alpha.sqrt()*lat + (1-alpha).sqrt()*torch.randn_like(lat)
                z_recon = traj[kidx]
                h.clear()
                pipe.unet(z_ref, t.to(DEVICE), encoder_hidden_states=pe).sample
                rf = dict(h.f)
                h.clear()
                pipe.unet(z_recon, t.to(DEVICE), encoder_hidden_states=pe).sample
                rf2 = dict(h.f)
                for ln in targets:
                    if ln in rf and ln in rf2:
                        lds[ln].append(float(torch.norm(rf[ln].float()-rf2[ln].float(),p=2).item()))
            for ln in targets:
                if lds[ln]: per_img[ln][nm] = float(np.mean(lds[ln]))
        except Exception as e:
            print(f"  FAIL {nm}: {e}")
    
    h.remove()
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
    
    summary = {
        "variant": name, "protocol_id": "P-multi-v1", "n_images": len(images),
        "steps": NUM_STEPS, "K_indices": list(K), "seed": SEED,
        "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                     "L": L, "peak_layer": canonical[idx]},
        "canonical_layers": list(canonical), "profile": {ln:float(profile[i]) for i,ln in enumerate(canonical)},
        "per_image": {ln:dict(per_img[ln]) for ln in canonical},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}_p_multi_104.json"
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"  {name}: pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} peak={canonical[idx]} → {out}")
    return summary

# ── Variant definitions ──
variants = {}
# SD1.4
print("Loading SD1.4...")
p14 = StableDiffusionPipeline.from_pretrained("/home/hiaskc/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/main", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
p14.scheduler = DDIMScheduler.from_config(p14.scheduler.config)
variants["SD1.4"] = (p14, p14.encode_prompt("", DEVICE, 1, False)[0])

# SD1.5 baseline (reuse pipeline from SD1.4 with SD1.5 weights)
print("Loading SD1.5 baseline...")
p15 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
p15.scheduler = DDIMScheduler.from_config(p15.scheduler.config)
# Keep empty prompt embeds
pe_base = p15.encode_prompt("", DEVICE, 1, False)[0]

# RV
print("Loading RV...")
p_rv = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
p_rv.scheduler = DDIMScheduler.from_config(p_rv.scheduler.config)
rv_unet_path = "/home/hiaskc/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21/unet"
from diffusers import UNet2DConditionModel
rv_unet = UNet2DConditionModel.from_pretrained(rv_unet_path, torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
p_rv.unet = rv_unet
variants["RealisticVision"] = (p_rv, p_rv.encode_prompt("", DEVICE, 1, False)[0])

# LCM LoRA
print("Loading LCM...")
p_lcm = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
p_lcm.scheduler = DDIMScheduler.from_config(p_lcm.scheduler.config)
p_lcm.load_lora_weights("/home/hiaskc/.cache/huggingface/hub/models--latent-consistency--lcm-lora-sdv1-5/snapshots/cf2fced511dbe7e26c8d1d397e728fbab875db4b")
variants["LCM-LoRA"] = (p_lcm, p_lcm.encode_prompt("", DEVICE, 1, False)[0])

# RandText: randomly shuffled prompt embeddings (done inline below)

# Measure
results = {}
for name, (pipe, pe) in variants.items():
    print(f"\n=== {name} ===")
    results[name] = measure_variant(name, pipe, pe, IMAGES)
    del pipe; torch.cuda.empty_cache()

# RandText: use SD1.5 baseline but with random text embeddings
print("\n=== RandText ===")
torch.manual_seed(42)
rand_emb = torch.randn_like(pe_base) * pe_base.std()  # same std, random direction
results["RandText"] = measure_variant("RandText", p15, rand_emb, IMAGES)
del p15, p14; torch.cuda.empty_cache()

print("\nDone. All 4 variants measured.")
for name, r in results.items():
    f = r["features"]
    print(f"  {name:20s}: pp={f['peak_position']:.4f} conc={f['concentration']:.4f} sp={f['spread']:.4f}")
