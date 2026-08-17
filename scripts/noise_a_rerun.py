"""
Noise A deep-dive rerun (per preregistered spec).
Conditions: orig / cut_a / noise_a, 19 coco_val images, seed 42, empty prompt, 50 DDIM steps.
Saves per-image PSNR/SSIM + per-layer drift profiles (P-multi-compatible K) to canonical store.
Analysis: paired t (noise vs orig), TOST (noise vs cut, 0.5 dB), peak drift drop %, cross-image std, Cohen's d.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from phase7_skip_intervention import (
    SkipIntervention, UNetFeatureHooker, ddim_inversion,
    load_and_encode, DEVICE, DTYPE,
)
from phase7_skip_noise_intervention import NoiseIntervention

PROJECT = Path("/home/hiaskc/Talant/graduation")
DATA_DIR = PROJECT / "data/coco_val"
OUT = PROJECT / "outputs/phase7_skip_intervention/noise_a_rerun.json"
IMAGES = sorted(DATA_DIR.glob("*.jpg"))[:19]
NUM_STEPS = 50
SEED = 42

print(f"Noise A rerun: {len(IMAGES)} images x 3 conditions")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, local_files_only=True).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

def get_K(n):
    k = [0, 1, 2, n//2-1, n//2, n//2+1, n-3, n-2, n-1]
    return sorted(set(max(0, min(n-1, i)) for i in k))

def measure_one(latent, intervention_ctx=None):
    """Run inversion+recon (under intervention), return (avg_drifts, recon_pixels)."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    K = get_K(len(timesteps))

    def _body():
        inv_latent = ddim_inversion(pipe, latent, prompt_embeds, NUM_STEPS)
        z = inv_latent.clone()
        recon_latents = [z.clone()]
        with torch.no_grad():
            for t in timesteps:
                noise_pred = pipe.unet(z, t, encoder_hidden_states=prompt_embeds).sample
                z = scheduler.step(noise_pred, t, z).prev_sample
                recon_latents.append(z.clone())
        # Feature comparison at K
        hooker = UNetFeatureHooker(pipe.unet)
        per_layer = defaultdict(list)
        torch.manual_seed(SEED); np.random.seed(SEED)
        with torch.no_grad():
            for idx in K:
                t = timesteps[idx]
                alpha = scheduler.alphas_cumprod[t]
                z_ref = alpha.sqrt()*latent + (1-alpha).sqrt()*torch.randn_like(latent)
                z_rec = recon_latents[idx]
                hooker.clear()
                pipe.unet(z_ref, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
                rf = hooker.features.copy()
                hooker.clear()
                pipe.unet(z_rec, t.to(DEVICE), encoder_hidden_states=prompt_embeds).sample
                rc = hooker.features.copy()
                for ln in rf:
                    if ln in rc:
                        per_layer[ln].append(torch.norm(rf[ln].float()-rc[ln].float(), p=2).item())
        hooker.remove()
        avg = {k: float(np.mean(v)) for k, v in per_layer.items()}
        return avg, z  # final reconstruction latent

    if intervention_ctx is not None:
        with intervention_ctx:
            return _body()
    return _body()

def psnr_ssim(pipe, orig_img_tensor, recon_latent):
    with torch.no_grad():
        recon_img = pipe.vae.decode(recon_latent / pipe.vae.config.scaling_factor).sample
    recon_01 = (recon_img / 2 + 0.5).clamp(0, 1)
    orig_01 = (orig_img_tensor / 2 + 0.5).clamp(0, 1)
    mse = (recon_01 - orig_01).pow(2).mean().item()
    psnr = 20*np.log10(1.0) - 10*np.log10(mse) if mse > 0 else 100.0
    # SSIM via skimage on CPU
    from skimage.metrics import structural_similarity as ssim
    o = orig_01.cpu().squeeze(0).permute(1,2,0).numpy()
    r = recon_01.cpu().squeeze(0).permute(1,2,0).numpy()
    s = ssim(o, r, channel_axis=2, data_range=1.0)
    return float(psnr), float(s)

results = {c: {"psnr": {}, "ssim": {}, "drift": {}} for c in ["orig", "cut_a", "noise_a"]}

for img_path in tqdm(IMAGES, desc="Noise A rerun"):
    nm = img_path.stem
    latent, img_tensor = load_and_encode(pipe, img_path)

    # orig
    avg, recon = measure_one(latent)
    psnr, ssim = psnr_ssim(pipe, img_tensor, recon)
    results["orig"]["psnr"][nm] = psnr
    results["orig"]["ssim"][nm] = ssim
    results["orig"]["drift"][nm] = avg

    # cut_a (peak skip zeroed)
    ctx = SkipIntervention(pipe.unet, [2])
    avg, recon = measure_one(latent, ctx)
    psnr, ssim = psnr_ssim(pipe, img_tensor, recon)
    results["cut_a"]["psnr"][nm] = psnr
    results["cut_a"]["ssim"][nm] = ssim
    results["cut_a"]["drift"][nm] = avg

    # noise_a (peak skip replaced with per-tensor-statistics Gaussian)
    ctx = NoiseIntervention(pipe.unet, [2])
    avg, recon = measure_one(latent, ctx)
    psnr, ssim = psnr_ssim(pipe, img_tensor, recon)
    results["noise_a"]["psnr"][nm] = psnr
    results["noise_a"]["ssim"][nm] = ssim
    results["noise_a"]["drift"][nm] = avg

    torch.cuda.empty_cache()

del pipe; torch.cuda.empty_cache()

# ── Analysis ──
from scipy import stats
print("\n=== PSNR ===")
stems = [p.stem for p in IMAGES]
ps = {c: np.array([results[c]["psnr"][n] for n in stems]) for c in results}
ss = {c: np.array([results[c]["ssim"][n] for n in stems]) for c in results}
for c in ["orig", "cut_a", "noise_a"]:
    print(f"  {c:>7s}: PSNR={ps[c].mean():.2f} ± {ps[c].std(ddof=1):.2f}  SSIM={ss[c].mean():.4f} ± {ss[c].std(ddof=1):.4f}")

t_no, p_no = stats.ttest_rel(ps["noise_a"], ps["orig"])
d_no = (ps["noise_a"].mean() - ps["orig"].mean()) / np.sqrt((ps["noise_a"].var(ddof=1)+ps["orig"].var(ddof=1))/2)
print(f"\n  paired t (noise vs orig): t={t_no:.2f}, p={p_no:.4f}, Cohen's d={d_no:.2f}")

# TOST noise vs cut (equivalence bound 0.5 dB)
diff = ps["noise_a"] - ps["cut_a"]
t1 = (diff.mean() - 0.5) / (diff.std(ddof=1)/np.sqrt(len(diff)))
t2 = (diff.mean() + 0.5) / (diff.std(ddof=1)/np.sqrt(len(diff)))
p_tost = max(stats.t.cdf(t1, len(diff)-1), 1 - stats.t.cdf(t2, len(diff)-1))
print(f"  TOST (noise vs cut, Δ=0.5 dB): mean diff={diff.mean():+.2f}, p={p_tost:.4f} "
      f"({'equivalent' if p_tost < 0.05 else 'NOT equivalent'})")

# Peak-layer drift drop
peak = "up_blocks.2.resnets.0"
dr = {c: np.array([results[c]["drift"][n][peak] for n in stems]) for c in results}
print(f"\n=== Peak-layer drift ({peak}) ===")
for c in ["orig", "cut_a", "noise_a"]:
    print(f"  {c:>7s}: {dr[c].mean():.0f} ± {dr[c].std(ddof=1):.0f}")
print(f"  cut_a drop:  {(1 - dr['cut_a'].mean()/dr['orig'].mean())*100:+.1f}%")
print(f"  noise_a drop: {(1 - dr['noise_a'].mean()/dr['orig'].mean())*100:+.1f}%")

summary = {
    "protocol_id": "P-multi-compatible (K=9, DDPM-forward ref, seed 42)",
    "n_images": len(IMAGES), "steps": NUM_STEPS,
    "conditions": ["orig", "cut_a", "noise_a"],
    "interventions": {
        "cut_a": "SkipIntervention, up_blocks[2] zeroed",
        "noise_a": "NoiseIntervention, up_blocks[2] replaced by per-tensor-statistics Gaussian",
    },
    "psnr_per_image": {c: results[c]["psnr"] for c in results},
    "ssim_per_image": {c: results[c]["ssim"] for c in results},
    "drift_per_image": {c: results[c]["drift"] for c in results},
    "analysis": {
        "psnr_mean": {c: float(ps[c].mean()) for c in ps},
        "psnr_std": {c: float(ps[c].std(ddof=1)) for c in ps},
        "ssim_mean": {c: float(ss[c].mean()) for c in ss},
        "paired_t_noise_vs_orig": {"t": float(t_no), "p": float(p_no), "cohens_d": float(d_no)},
        "tost_noise_vs_cut_0p5db": {"mean_diff": float(diff.mean()), "p": float(p_tost)},
        "peak_layer": peak,
        "peak_drift_mean": {c: float(dr[c].mean()) for c in dr},
        "peak_drift_std": {c: float(dr[c].std(ddof=1)) for c in dr},
        "peak_drop_pct": {"cut_a": float((1-dr['cut_a'].mean()/dr['orig'].mean())*100),
                          "noise_a": float((1-dr['noise_a'].mean()/dr['orig'].mean())*100)},
    },
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUT}")
