#!/usr/bin/env python
"""
C1 Boundary Spectrum — 100-image remeasurement.
===============================================

Remeasures all 4 intra-architecture variants on 104 coco_val100 images:
  SD1.4  — continued training (CompVis checkpoint)
  LCM    — LCM LoRA distillation
  RandText — random text encoder replacement
  RV     — Realistic Vision full fine-tune

Uses canonical UNet layer ordering and v2 metric (D_total, no peak_count).
SD 1.5 baseline loaded from existing unified_100img measurement.

Usage:
    python scripts/c1_boundary_100img.py              # all 4 variants
    python scripts/c1_boundary_100img.py --model SD14 # single variant
    python scripts/c1_boundary_100img.py --images 20  # quick test

Output: outputs/c1_boundary_100img/{variant}_drift.json
        outputs/c1_boundary_100img/boundary_spectrum.json
"""

import argparse, copy, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "coco_val100"
OUT_DIR = PROJECT_ROOT / "outputs" / "c1_boundary_100img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"; DTYPE = torch.float16

# ─── v2 features ───

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s>0 else 0.0

def extract_v2(profile, layer_names):
    p = np.asarray(profile, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax>dmin else p.copy()
    pp = float(np.argmax(pn))/L
    k = max(1,int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    peaks, props = find_peaks(pn, prominence=0.1)
    n_peaks = len(peaks)
    top_prom = float(props['prominences'][np.argmax(pn[peaks])]) if len(peaks)>0 else 0.0
    peak_layer = layer_names[int(np.argmax(pn))]
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "n_peaks": n_peaks, "top_prominence": top_prom,
            "L": L, "peak_layer": peak_layer}

def d_v2(fa, fb):
    d_pp = abs(fa["peak_position"]-fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"],
                            fa["spread"]-fb["spread"]])
    d_total = float(np.linalg.norm([d_pp, d_mag]))
    return {"D_total": d_total, "D_peak_pos": d_pp, "D_mag": d_mag}

# ─── Canonical UNet ordering ───

def _unet_sort_key(name):
    parts = name.split(".")
    section_order = {"down_blocks":0,"mid_block":1,"up_blocks":2}
    sec = section_order.get(parts[0],99)
    blk_idx=0; type_ord=0; sub_idx=0
    for i,p in enumerate(parts):
        if p in ("down_blocks","mid_block","up_blocks"):
            if i+1<len(parts) and parts[i+1].isdigit(): blk_idx=int(parts[i+1])
        elif p=="resnets":
            type_ord=0
            if i+1<len(parts) and parts[i+1].isdigit(): sub_idx=int(parts[i+1])
        elif p=="attentions":
            type_ord=1
            if i+1<len(parts) and parts[i+1].isdigit(): sub_idx=int(parts[i+1])
    return (sec, blk_idx, type_ord, sub_idx)

def discover_unet_targets(unet):
    targets = []
    for n,m in unet.named_modules():
        parts = n.split(".")
        if "resnets" in parts:
            idx = parts.index("resnets")
            if len(parts)==idx+2 and parts[-1].isdigit(): targets.append(n)
        if "transformer_blocks" in parts:
            idx = parts.index("transformer_blocks")
            if len(parts)==idx+2 and parts[-1]=="0": targets.append(n)
    return sorted(targets, key=_unet_sort_key)

# ─── Hooks + DDIM ───

class FeatureExtractor:
    def __init__(s, model, targets):
        s.model=model; s.targets=set(targets); s.features={}; s.handles=[]
    def _fn(s, n):
        def fn(m,i,o): s.features[n]= (o[0] if isinstance(o,tuple) else o).detach().float().cpu()
        return fn
    def register(s):
        s.remove()
        for n,m in s.model.named_modules():
            if n in s.targets: s.handles.append(m.register_forward_hook(s._fn(n)))
    def remove(s):
        for h in s.handles: h.remove()
        s.handles.clear(); s.features.clear()

def ddim_inv(pipe, latent, pe, steps):
    s=pipe.scheduler; s.set_timesteps(steps,device=DEVICE)
    ts=s.timesteps; z=latent.clone(); ext=ts.tolist()+[0]
    with torch.no_grad():
        for i in range(len(ext)-1,0,-1):
            tc,tn=ext[i],ext[i-1]
            npred=pipe.unet(z,tc,encoder_hidden_states=pe).sample
            ac,an=s.alphas_cumprod[tc],s.alphas_cumprod[tn]
            c1=(an/ac).sqrt(); sc=(1-ac).sqrt(); sn=(1-an).sqrt()
            z=c1*z+(sn-c1*sc)*npred
    return z

def ddim_recon(pipe, noise, pe, steps):
    s=pipe.scheduler; s.set_timesteps(steps,device=DEVICE); z=noise.clone()
    with torch.no_grad():
        for t in s.timesteps:
            npred=pipe.unet(z,t,encoder_hidden_states=pe).sample
            z=s.step(npred,t,z).prev_sample
    return z

# ─── Baseline (SD 1.5) load from existing ───

def load_sd15_baseline():
    """Load SD1.5 baseline features + per-image drift from unified_100img."""
    with open("outputs/unified_100img/SD1.5_drift.json") as f:
        d = json.load(f)
    return d["features"], d["layer_order"], d["per_image"]

# ─── Measurement functions ───

def measure_variant(pipe, extractor, targets, images, pe, label, steps=50):
    """Generic: run inversion-reconstruction on `images`, return per-image drift."""
    per_image = defaultdict(dict)
    for img_path in tqdm(images, desc=f"  {label}"):
        img = Image.open(img_path).convert("RGB").resize((512,512), Image.LANCZOS)
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
        t = 2*t-1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor

        extractor.features.clear()
        z_inv = ddim_inv(pipe, latent, pe, steps)
        inv_f = {k:v.clone() for k,v in extractor.features.items()}
        extractor.features.clear()
        _ = ddim_recon(pipe, z_inv, pe, steps)
        recon_f = {k:v.clone() for k,v in extractor.features.items()}

        for ln in targets:
            if ln in inv_f and ln in recon_f:
                drift = float(torch.norm(inv_f[ln]-recon_f[ln], p=2).item())
                per_image[ln][img_path.name] = drift
    return dict(per_image)


def measure_sd14(images, steps=50):
    """SD 1.4: swap CompVis UNet into SD1.5 pipeline."""
    from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    unet_path = "/home/hiaskc/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/main/unet"
    print("  Loading SD1.5 pipeline + SD1.4 UNet...")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=DTYPE)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE,
                                                     unet=unet).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad(): pe = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

    return measure_variant(pipe, extractor, targets, images, pe, "SD1.4", steps), targets


def measure_lcm(images, steps=50):
    """LCM LoRA: apply LCM LoRA to SD1.5 pipeline."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler, LCMScheduler
    lora_id = "latent-consistency/lcm-lora-sdv1-5"
    model_id = "runwayml/stable-diffusion-v1-5"
    print("  Loading SD1.5 + LCM LoRA...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.load_lora_weights(lora_id, weight_name="pytorch_lora_weights.safetensors")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # use DDIM for inversion
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad(): pe = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

    return measure_variant(pipe, extractor, targets, images, pe, "LCM", steps), targets


def measure_randtext(images, steps=50):
    """RandText: replace text encoder output with random Gaussian embeddings."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    print("  Loading SD1.5 for RandText...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    # Generate random embedding matching CLIP output shape
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    token_len = ti.input_ids.shape[1]
    hidden_dim = pipe.text_encoder.config.hidden_size
    rng = np.random.RandomState(42)
    random_emb = torch.from_numpy(rng.randn(1, token_len, hidden_dim).astype(np.float32)).to(DEVICE, dtype=DTYPE)
    pe = random_emb

    return measure_variant(pipe, extractor, targets, images, pe, "RandText", steps), targets


def measure_rv(images, steps=50):
    """Realistic Vision: swap RV UNet into SD1.5 pipeline."""
    from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
    model_id = "runwayml/stable-diffusion-v1-5"
    unet_path = "/home/hiaskc/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21/unet"
    print("  Loading SD1.5 pipeline + RV UNet...")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=DTYPE)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE,
                                                     unet=unet).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    targets = discover_unet_targets(pipe.unet)
    extractor = FeatureExtractor(pipe.unet, targets)
    extractor.register()
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad(): pe = pipe.text_encoder(ti.input_ids.to(DEVICE))[0]

    return measure_variant(pipe, extractor, targets, images, pe, "RV", steps), targets


# ─── Registry ───

VARIANTS = {
    "SD14":     {"fn": measure_sd14,     "label": "SD 1.4 (continued training)"},
    "LCM":      {"fn": measure_lcm,      "label": "LCM LoRA (distillation)"},
    "RandText": {"fn": measure_randtext, "label": "Random Text Encoder"},
    "RV":       {"fn": measure_rv,       "label": "Realistic Vision (full fine-tune)"},
}

# ─── Main ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Single variant to measure")
    ap.add_argument("--images", type=int, default=104)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    images = sorted(DATA_DIR.glob("coco_*.jpg"))[:args.images]
    print(f"C1 Boundary Spectrum — 100-image Remeasurement")
    print(f"  Images: {len(images)}, Steps: {args.steps}")

    # Load SD1.5 baseline
    sd15_feat, sd15_order, _ = load_sd15_baseline()
    print(f"  SD1.5 baseline: pp={sd15_feat['peak_position']:.4f} peak={sd15_feat['peak_layer']}")

    variants_to_run = [args.model] if args.model else list(VARIANTS.keys())
    results = {}
    has_nan = False  # track any NaN issues

    for name in variants_to_run:
        info = VARIANTS[name]
        out_path = OUT_DIR / f"{name}_drift.json"
        if out_path.exists():
            print(f"\n[{name}] Already exists, skip")
            with open(out_path) as f: results[name] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"[{name}] {info['label']}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            per_image, targets = info["fn"](images, steps=args.steps)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time()-t0

        profile = np.array([np.mean(list(per_image[ln].values())) for ln in targets])
        features = extract_v2(profile, targets)
        dd = d_v2(sd15_feat, features)

        # Check for NaN
        if np.any(np.isnan(profile)):
            print(f"  WARNING: NaN in profile!")
            has_nan = True

        print(f"  Done in {elapsed:.0f}s. pp={features['peak_position']:.4f} ({features['peak_layer']})")
        print(f"  D_total={dd['D_total']:.6f} D_pp={dd['D_peak_pos']:.6f} D_mag={dd['D_mag']:.6f}")

        result = {"variant": name, "label": info["label"],
                  "n_images": len(images), "n_steps": args.steps,
                  "n_layers": len(targets), "layer_order": targets,
                  "features": features, "D_vs_sd15": dd,
                  "per_image": {ln: dict(vals) for ln, vals in per_image.items()}}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        results[name] = result

    # ─── Boundary spectrum summary ───
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("C1 Boundary Spectrum (100-image, canonical ordering)")
        print(f"{'='*60}")
        print(f"{'Variant':>12s}  {'D_total':>9s}  {'D_pp':>7s}  {'D_mag':>7s}  {'Peak Pos':>9s}")
        print("-"*55)
        for name in VARIANTS:
            if name in results:
                r = results[name]
                dd = r["D_vs_sd15"]
                pp = r["features"]["peak_position"]
                print(f"{name:>12s}  {dd['D_total']:9.6f}  {dd['D_peak_pos']:7.5f}  {dd['D_mag']:7.5f}  {pp:9.4f}")

        # Noise floor from bootstrap
        with open("outputs/unified_100img/cross_arch_v2_matrix.json") as f:
            m = json.load(f)
        nf = m.get("noise_floor", {})
        noise_median = nf.get("median", 0.0017)
        noise_p95 = nf.get("p95", 0.0046)

        # Order
        entries = sorted([(name, results[name]["D_vs_sd15"]["D_total"]) for name in results],
                        key=lambda x: x[1])
        spectrum = " < ".join([f"{n}({v:.4f})" for n,v in entries])
        max_intra = max(v for _,v in entries)
        min_inter = min(dd["D_total"] for dd in m.get("pairwise", {}).values()) if m.get("pairwise") else 0.142

        print(f"\n  Boundary spectrum: {spectrum}")
        print(f"  Noise floor: median={noise_median:.6f} p95={noise_p95:.6f}")
        print(f"  Max intra-arch: {max_intra:.4f}")
        print(f"  Min inter-arch: {min_inter:.4f}")
        print(f"  Gap ratio: {min_inter/max_intra:.1f}x")

        summary = {
            "protocol": {"images": len(images), "steps": args.steps,
                         "ordering": "canonical (resnets before attentions per block)"},
            "noise_floor": nf,
            "sd15_baseline": sd15_feat,
            "variants": {name: results[name]["features"] for name in results},
            "distances": {name: results[name]["D_vs_sd15"] for name in results},
            "boundary_spectrum": spectrum,
            "max_intra_arch": max_intra,
            "gap_ratio": min_inter/max_intra,
        }
        with open(OUT_DIR / "boundary_spectrum.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved to {OUT_DIR}/boundary_spectrum.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
