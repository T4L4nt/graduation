"""
Random-initialization P-multi rerun (5 seeds).
SD1.5 UNet with Kaiming-initialized weights, P-multi protocol, coco_val (19 images).
Reference: trained SD1.5 P-multi pp=0.684 (U2.R0).
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from layer_order import unet_topo_key

DEVICE = "cuda"; DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT = Path("/home/hiaskc/Talant/graduation/outputs/p0b_cross_checkpoint/random_init_p_multi_104.json")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50
SEEDS = [0, 1, 2, 3, 4]

print(f"Random-init P-multi: {len(IMAGES)} images, {NUM_STEPS} steps, {len(SEEDS)} seeds")

# Load pipeline once
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

def discover_targets(unet):
    tgt = []
    for n, _ in unet.named_modules():
        p = n.split(".")
        if "resnets" in p:
            idx = p.index("resnets")
            if len(p) == idx+2 and p[-1].isdigit(): tgt.append(n)
        if "transformer_blocks" in p:
            idx = p.index("transformer_blocks")
            if len(p) == idx+2 and p[-1] == "0": tgt.append(n)
    return sorted(tgt)

class Hooker:
    def __init__(s, unet, tgt):
        s.f = {}; s.h = []
        for n in tgt:
            m = unet
            for t in n.split("."):
                try: m = getattr(m, t)
                except: m = None; break
            if m: s.h.append(m.register_forward_hook(lambda m,i,o,n=n: s._f(n,o)))
    def _f(s, n, o):
        if isinstance(o, tuple): o = o[0]
        if o.dim() == 3: o = o.mean(1, keepdim=True)
        s.f[n] = o.detach().cpu()
    def clear(s): s.f.clear()
    def remove(s):
        for h in s.h: h.remove()
        s.h.clear()

def get_K(n):
    k = [0,1,2,n//2-1,n//2,n//2+1,n-3,n-2,n-1]
    return sorted(set(max(0,min(n-1,i)) for i in k))
K = get_K(NUM_STEPS)

def ddim_inv(unet, lat):
    s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
    ts = s.timesteps; z = lat.clone(); ext = ts.tolist() + [0]
    with torch.no_grad():
        for i in range(len(ext)-1, 0, -1):
            tc, tn = ext[i], ext[i-1]
            npred = unet(z, tc, encoder_hidden_states=prompt_embeds).sample
            ac, an = s.alphas_cumprod[tc], s.alphas_cumprod[tn]
            c1 = (an/ac).sqrt(); sc, sn = (1-ac).sqrt(), (1-an).sqrt()
            z = c1*z + (sn - c1*sc)*npred
    return z

def ddim_recon_traj(unet, noise):
    s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
    ts = s.timesteps; z = noise.clone(); traj = [z.clone()]
    with torch.no_grad():
        for t in ts:
            npred = unet(z, t, encoder_hidden_states=prompt_embeds).sample
            z = s.step(npred, t, z).prev_sample
            traj.append(z.clone())
    return traj

def load_enc(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    t = 2*t - 1
    with torch.no_grad():
        lat = pipe.vae.encode(t).latent_dist.sample() * pipe.vae.config.scaling_factor
    return lat

def randomize_unet(seed):
    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_config(pipe.unet.config).to(DEVICE, dtype=DTYPE)
    def init_fn(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, a=np.sqrt(5))
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            if getattr(m, 'weight', None) is not None:
                torch.nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.zeros_(m.bias)
    unet.apply(init_fn)
    return unet

# Pre-encode all images once
print("Pre-encoding images...")
latents = {img_path.stem: load_enc(img_path) for img_path in tqdm(IMAGES, desc="  encode")}

results = {}
for seed in SEEDS:
    print(f"\n=== seed {seed} ===")
    unet = randomize_unet(seed)
    targets = discover_targets(unet)
    canonical = sorted(targets, key=unet_topo_key)
    hooker = Hooker(unet, targets)

    per_img = defaultdict(dict)
    for img_path in tqdm(IMAGES, desc=f"  seed {seed}"):
        nm = img_path.stem
        lat = latents[nm]
        inv_lat = ddim_inv(unet, lat)
        traj = ddim_recon_traj(unet, inv_lat)
        s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
        ts = s.timesteps
        lds = defaultdict(list)
        torch.manual_seed(42); np.random.seed(42)
        for kidx in K:
            t = ts[kidx]; alpha = s.alphas_cumprod[t]
            z_ref = alpha.sqrt()*lat + (1-alpha).sqrt()*torch.randn_like(lat)
            z_rec = traj[kidx]
            hooker.clear()
            with torch.no_grad():
                unet(z_ref, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
            rf = dict(hooker.f)
            hooker.clear()
            with torch.no_grad():
                unet(z_rec, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
            rc = dict(hooker.f)
            for ln in targets:
                if ln in rf and ln in rc:
                    lds[ln].append(float(torch.norm(rf[ln].float()-rc[ln].float(), p=2).item()))
        for ln in targets:
            if lds[ln]: per_img[ln][nm] = float(np.mean(lds[ln]))

    hooker.remove()
    profile = [float(np.mean(list(per_img[ln].values()))) for ln in canonical]
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    pn = (p-p.min())/(p.max()-p.min()) if p.max() > p.min() else p
    idx = int(pn.argmax()); pp = idx/L
    kk = max(1, int(np.ceil(0.2*L)))
    conc = float(np.sum(pn[np.argsort(pn)[-kk:]])/np.sum(pn))
    def gini(x):
        x = np.sort(np.asarray(x, dtype=np.float64))
        n = len(x); s = np.sum(x)
        return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s > 0 else 0.0
    sp = float(gini(pn))
    results[str(seed)] = {"pp": pp, "conc": conc, "sp": sp,
                          "peak_layer": canonical[idx], "L": L}
    print(f"  seed {seed}: pp={pp:.4f} conc={conc:.4f} sp={sp:.4f} peak={canonical[idx]}")
    del unet; torch.cuda.empty_cache()

summary = {
    "protocol_id": "P-multi-v1",
    "n_images": len(IMAGES), "steps": NUM_STEPS, "K_indices": list(K),
    "seeds": SEEDS,
    "reference_trained": {"pp": 0.6842, "conc": 0.5830, "sp": 0.5876,
                          "peak_layer": "up_blocks.2.resnets.0"},
    "results": results,
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT}")
