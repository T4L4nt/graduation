"""
SD1.5 phase1@104 — Definition 1 compliant protocol.

Protocol manifest (extracted from phase1_diagnostics.py):
  Model:    runwayml/stable-diffusion-v1-5, fp16
  Sampler:  DDIM, 50 steps, eta=0, empty prompt, guidance_scale=1.0
  K:        {0,1,2, n//2-1,n//2,n//2+1, n-3,n-2,n-1} = 9 timestep indices
  Seeding:  seed=42 per image (DDPM ref noise uses this seed)
  Ref:      for each t in K: DDPM_noise(original_latent, t, seed) → unet(z_ref, t) → f_ref
  Recon:    inversion then reconstruction, record latents[K], unet(z_recon[k], t_k) → f_recon
  Drift:    mean_{k in K} L2(f_ref[t_k] - f_recon[t_k])
  Hook:     discover_hook_targets() → resnets.N (block output) + transformer_blocks.0 (attention output)
            attention: [B,N,C] → mean(dim=1) → [B,1,C] for comparison
  Aggregation: per-layer drift averaged over K → d_l(x); cross-image mean → aggregate
  Norm:      min-max normalization per model (across layers)
  Features:  extract_v2(profile, canonical_layers)

Output: outputs/phase1/layer_drift_summary_104img.json
"""

import json, sys, csv
from pathlib import Path
from collections import defaultdict
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
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

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUT_DIR = Path("/home/hiaskc/Talant/graduation/outputs/phase1")
DATA_DIR = Path("/home/hiaskc/Talant/graduation/data/coco_val100")
IMAGES = sorted(DATA_DIR.glob("coco_*.jpg"))
NUM_STEPS = 50
SEEDS = [42]

# ── Pipeline ──
print(f"SD1.5 phase1@104: {len(IMAGES)} images, {NUM_STEPS} DDIM steps, seeds={SEEDS}")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

# ── Hook discovery (from phase1_diagnostics.py) ──
def discover_hook_targets(unet):
    targets = []
    for name, _ in unet.named_modules():
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

targets = discover_hook_targets(pipe.unet)
canonical_targets = sorted(targets, key=unet_topo_key)
print(f"  Hook targets: {len(targets)} layers, canonical peak order verified")

class UNetFeatureHooker:
    def __init__(self, unet):
        self.features = {}; self.handles = []
        for name in targets:
            mod = unet
            for t in name.split("."):
                try: mod = getattr(mod, t)
                except: mod = None; break
            if mod is not None:
                self.handles.append(mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._hook_fn(n, out)))
    def _hook_fn(self, name, output):
        if isinstance(output, tuple): output = output[0]
        if output.dim() == 3: output = output.mean(dim=1, keepdim=True)
        self.features[name] = output.detach().cpu()
    def clear(self): self.features.clear()
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

hooker = UNetFeatureHooker(pipe.unet)

# ── DDIM inversion ──
def ddim_inversion(latent, emb, steps):
    s = pipe.scheduler; s.set_timesteps(steps, device=DEVICE)
    ts = s.timesteps; z = latent.clone(); ext = ts.tolist() + [0]
    with torch.no_grad():
        for i in range(len(ext)-1, 0, -1):
            tc, tn = ext[i], ext[i-1]
            npred = pipe.unet(z, tc, encoder_hidden_states=emb).sample
            ac, an = s.alphas_cumprod[tc], s.alphas_cumprod[tn]
            c1 = (an/ac).sqrt(); sc, sn = (1-ac).sqrt(), (1-an).sqrt()
            z = c1*z + (sn - c1*sc)*npred
    return z

# ── Reconstruction with latent trajectory ──
def ddim_reconstruction_trajectory(noise, emb, steps):
    s = pipe.scheduler; s.set_timesteps(steps, device=DEVICE)
    ts = s.timesteps; z = noise.clone()
    traj = [z.clone()]
    with torch.no_grad():
        for t in ts:
            npred = pipe.unet(z, t, encoder_hidden_states=emb).sample
            z = s.step(npred, t, z).prev_sample
            traj.append(z.clone())
    return traj

# ── Key timestep indices (from phase1_diagnostics.py) ──
def get_key_indices(n_steps):
    if n_steps <= 6: return list(range(n_steps))
    k = [0, 1, 2, n_steps//2-1, n_steps//2, n_steps//2+1, n_steps-3, n_steps-2, n_steps-1]
    return sorted(set(max(0, min(n_steps-1, i)) for i in k))

key_indices = get_key_indices(NUM_STEPS)
print(f"  K = {len(key_indices)} timestep indices: {key_indices}")

# ── Image loading ──
def load_and_encode(path):
    img = Image.open(path).convert("RGB").resize((512, 512))
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    t = 2*t - 1
    with torch.no_grad():
        lat = pipe.vae.encode(t).latent_dist.sample() * pipe.vae.config.scaling_factor
    return lat

# ── Measure ──
per_image = defaultdict(dict)

for img_path in tqdm(IMAGES, desc="SD1.5 phase1@104"):
    img_name = img_path.stem
    try:
        latent = load_and_encode(img_path)

        # Inversion
        inv_latent = ddim_inversion(latent, prompt_embeds, NUM_STEPS)

        # Reconstruction trajectory
        recon_traj = ddim_reconstruction_trajectory(inv_latent, prompt_embeds, NUM_STEPS)

        s = pipe.scheduler; s.set_timesteps(NUM_STEPS, device=DEVICE)
        timesteps = s.timesteps

        layer_drifts = defaultdict(list)

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            for kidx in key_indices:
                t = timesteps[kidx]
                alpha = s.alphas_cumprod[t]
                noise_ref = torch.randn_like(latent)
                z_ref = alpha.sqrt()*latent + (1-alpha).sqrt()*noise_ref
                z_recon = recon_traj[kidx]

                hooker.clear()
                pipe.unet(z_ref, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
                ref_f = dict(hooker.features)

                hooker.clear()
                pipe.unet(z_recon, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
                recon_f = dict(hooker.features)

                for ln in targets:
                    if ln in ref_f and ln in recon_f:
                        l2 = float(torch.norm(ref_f[ln].float() - recon_f[ln].float(), p=2).item())
                        layer_drifts[ln].append(l2)

        # Per-layer mean across K and seeds
        for ln in targets:
            if layer_drifts[ln]:
                per_image[ln][img_name] = float(np.mean(layer_drifts[ln]))
    except Exception as e:
        print(f"  FAIL {img_name}: {e}")

hooker.remove()
del pipe; torch.cuda.empty_cache()

# ── Aggregate (canonical order) ──
profile = [float(np.mean(list(per_image[ln].values()))) for ln in canonical_targets]

# Features
from scipy.signal import find_peaks
def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x) - (n+1)*s)/(n*s)) if s > 0 else 0.0

p = np.asarray(profile, dtype=np.float64); L = len(p)
dmin, dmax = p.min(), p.max()
pn = (p-dmin)/(dmax-dmin)
idx = int(np.argmax(pn)); pp = idx/L
k = max(1, int(np.ceil(0.2*L)))
conc = float(np.sum(pn[np.argsort(pn)[-k:]])/np.sum(pn))
sp = float(gini(pn))
peaks, _ = find_peaks(pn, prominence=0.1)

print(f"\n{'='*60}")
print(f"SD1.5 phase1@104: {len(IMAGES)} images")
print(f"  pp={pp:.4f}  conc={conc:.4f}  sp={sp:.4f}")
print(f"  peak={canonical_targets[idx]} (idx={idx}/{L})")
print(f"  n_peaks={len(peaks)}")

# Compare with old phase1@19
print(f"\n  Old phase1@19:  pp=0.6842  conc=0.670  sp=0.641")
print(f"  New phase1@104: pp={pp:.4f}  conc={conc:.4f}  sp={sp:.4f}")
print(f"  Δ:              pp={pp-0.6842:+.4f}  conc={conc-0.670:+.4f}  sp={sp-0.641:+.4f}")

# Save
summary = {
    "protocol": "Definition 1: multi-timestep L2(f_ref - f_recon), K=9 indices, DDIM 50-step",
    "n_images": len(IMAGES), "steps": NUM_STEPS, "seeds": SEEDS,
    "K_indices": list(key_indices),
    "canonical_layers": list(canonical_targets),
    "profile": {ln: float(profile[i]) for i, ln in enumerate(canonical_targets)},
    "features": {"peak_position": pp, "concentration": conc, "spread": sp,
                 "n_peaks": int(len(peaks)), "L": L, "peak_layer": canonical_targets[idx]},
    "per_image": {ln: dict(per_image[ln]) for ln in canonical_targets},
    "layer_list_hash": layer_hash("SD1.5", list(targets)),
}
out = OUT_DIR / "layer_drift_summary_104img.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
