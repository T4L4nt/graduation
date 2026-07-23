"""
P0b #2: Conflict metric bake-off for skip connection causal role.

Computes 3 candidate conflict metrics at each skip connection,
correlates each with the ablation effect (ΔPSNR from Cut A / Cut B / partial α).

Metrics:
  1. L2_pairing: ||s - u||_2 / ||u||_2  (original Conflict C)
  2. Cosine_mismatch: 1 - cos(s, u)
  3. L2_norm_ratio: ||s|| / ||u|| (simpler than L2_pairing)

SD 1.5 has 5 skip connections (down_blocks.i → up_blocks[3-i]):
  - down_blocks.0 → up_blocks.3 (low drift, Cut B target)
  - down_blocks.1 → up_blocks.2 (high drift, Cut A target)
  - down_blocks.2 → up_blocks.1
  - down_blocks.3 → up_blocks.0 (mid_block → up_blocks.0)

Diagnoses each skip on 3 images, saves metrics for downstream analysis.
"""

import copy, json, sys, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEVICE = "cuda"; DTYPE = torch.float16
OUT = Path("outputs/p0b_cross_checkpoint"); OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMAGES = sorted((Path("data/coco_val")).glob("*.jpg"))[:3]

# SD 1.5 skip connections: (down_block, up_block_index)
SKIPS = [
    ("down_blocks.0", 3),  # → up_blocks.3
    ("down_blocks.1", 2),  # → up_blocks.2 (Cut A peak)
    ("down_blocks.2", 1),  # → up_blocks.1
    ("mid_block", 0),      # → up_blocks.0
]


def enc_empty(pipe):
    ti = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
    with torch.no_grad():
        return pipe.text_encoder(ti.input_ids.to(DEVICE))[0]


def ddim_inv(pipe, lat, pe, N):
    s = pipe.scheduler; s.set_timesteps(N, device=DEVICE)
    ts = s.timesteps; z = lat.clone(); ext = ts.tolist() + [0]
    with torch.no_grad():
        for i in range(len(ext)-1, 0, -1):
            tc, tn = ext[i], ext[i-1]
            npred = pipe.unet(z, tc, encoder_hidden_states=pe).sample
            ac, an = s.alphas_cumprod[tc], s.alphas_cumprod[tn]
            c1 = (an/ac).sqrt(); sc = (1-ac).sqrt(); sn = (1-an).sqrt()
            z = c1*z + (sn-c1*sc)*npred
    return z


def load_enc(pipe, path, sz=512):
    img = Image.open(path).convert("RGB").resize((sz, sz))
    t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    t = 2*t - 1
    with torch.no_grad():
        l = pipe.vae.encode(t).latent_dist.sample()
    return l * pipe.vae.config.scaling_factor


def hook_skip_features(unet, down_name, up_idx):
    """Hook the skip feature (from down_block output) and the up_block input (before adding skip).

    SD 1.5 UNet structure: each down/up block has multiple resnets and attentions.
    The skip connects the output of the LAST resnet in down_block[N] to the
    LAST resnet in up_block[3-N] (via residual concatenation).

    We hook:
    - skip_src: the output of down_block[N] (post final processing)
    - skip_dst_pre: the input to the up_block's final resnet BEFORE skip addition
    """
    features = {}
    handles = []

    # Hook the output of the down block (skip source)
    # In SD 1.5 UNet, up_blocks receive skip connections from corresponding down_blocks
    # The skip is concatenated with the upsampled features

    # Find the final resnet in the specified down_block
    for name, module in unet.named_modules():
        # Hook the last resnet in the down block (output carries skip signal)
        if name.startswith(down_name) and name.endswith("resnets.1") and "attentions" not in name:
            def make_fn(n):
                def fn(m, i, o):
                    features[f"{n}_out"] = o.detach().float().cpu()
                return fn
            handles.append(module.register_forward_hook(make_fn(name)))

    # Hook the up_block input before skip
    up_name = f"up_blocks.{up_idx}"
    for name, module in unet.named_modules():
        if name.startswith(up_name) and name.endswith("resnets.2") and "attentions" not in name:
            def make_fn2(n):
                def fn(m, i, o):
                    features[f"{n}_pre"] = i[0].detach().float().cpu()
                return fn
            handles.append(module.register_forward_hook(make_fn2(name)))

    return features, handles


def compute_conflict_metrics(skip_feat, up_feat):
    """Compute 3 candidate conflict metrics with spatial mean pooling for alignment."""
    # Spatial mean pool to get per-channel vectors
    s = skip_feat.float().mean(dim=[2, 3]).reshape(-1) if skip_feat.dim() == 4 else skip_feat.float().reshape(-1)
    u = up_feat.float().mean(dim=[2, 3]).reshape(-1) if up_feat.dim() == 4 else up_feat.float().reshape(-1)

    # 1. L2 pairing (original C)
    l2_pairing = float(torch.norm(s - u).item()) / max(torch.norm(u).item(), 1e-8)

    # 2. Cosine mismatch
    cos_sim = float(torch.dot(s, u).item()) / max(torch.norm(s).item() * torch.norm(u).item(), 1e-8)
    cosine_mismatch = 1.0 - cos_sim

    # 3. L2 norm ratio
    l2_norm_ratio = float(torch.norm(s).item()) / max(torch.norm(u).item(), 1e-8)

    return {"L2_pairing": l2_pairing, "Cosine_mismatch": cosine_mismatch, "L2_norm_ratio": l2_norm_ratio}


def main():
    print("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, local_files_only=True, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pe = enc_empty(pipe)

    all_skip_metrics = {f"skip_{db}_{ui}": {"L2_pairing": [], "Cosine_mismatch": [], "L2_norm_ratio": []}
                        for db, ui in SKIPS}

    for img_path in IMAGES:
        print(f"\n  Image: {img_path.name}")
        latent = load_enc(pipe, img_path)
        z_inv = ddim_inv(pipe, latent, pe, 50)

        # For each skip, hook and measure conflict at turnaround
        for down_name, up_idx in SKIPS:
            feats, handles = hook_skip_features(pipe.unet, down_name, up_idx)

            # Run one forward pass to capture features
            with torch.no_grad():
                pipe.unet(z_inv, pipe.scheduler.timesteps[0], encoder_hidden_states=pe)

            # Extract skip and up features
            db_short = down_name.split(".")[-2] + down_name.split(".")[-1]  # e.g. "d0"
            skip_key = f"{down_name}.resnets.1_out"
            up_key = f"up_blocks.{up_idx}.resnets.2_pre"

            if skip_key in feats and up_key in feats:
                metrics = compute_conflict_metrics(feats[skip_key], feats[up_key])
                label = f"skip_{db_short}_{up_idx}"
                for metric_name, val in metrics.items():
                    all_skip_metrics[label][metric_name].append(val)
                print(f"    {label}: L2={metrics['L2_pairing']:.4f} Cos={metrics['Cosine_mismatch']:.4f} NormRatio={metrics['L2_norm_ratio']:.4f}")
            else:
                print(f"    {label}: MISSING features (skip_key={skip_key in feats}, up_key={up_key in feats})")
                print(f"    Available: {list(feats.keys())[:5]}")

            for h in handles:
                h.remove()

    # Aggregate
    print(f"\n{'='*60}")
    print("Conflict Metric Summary (mean over images)")
    print(f"{'='*60}")

    # Reference ablation effects from Phase 7c:
    # Cut A (skip_d1_2): +2.20 dB
    # Cut B (skip_d0_3): -0.11 dB
    ablation_ref = {
        "skip_d0_3": -0.11,   # Cut B
        "skip_d1_2": +2.20,   # Cut A
    }

    print(f"{'Skip':>12s}  {'L2_pair':>10s}  {'Cos_mis':>10s}  {'NRatio':>10s}  {'ΔPSNR':>10s}")
    print("-" * 60)

    summary = {}
    for label, metrics_dict in all_skip_metrics.items():
        l2_mean = float(np.mean(metrics_dict["L2_pairing"])) if metrics_dict["L2_pairing"] else 0
        cos_mean = float(np.mean(metrics_dict["Cosine_mismatch"])) if metrics_dict["Cosine_mismatch"] else 0
        nr_mean = float(np.mean(metrics_dict["L2_norm_ratio"])) if metrics_dict["L2_norm_ratio"] else 0
        dpsnr = ablation_ref.get(label, None)
        dpsnr_str = f"{dpsnr:+.2f}" if dpsnr is not None else "N/A"
        print(f"{label:>12s}  {l2_mean:10.4f}  {cos_mean:10.4f}  {nr_mean:10.4f}  {dpsnr_str:>10s}")
        summary[label] = {"L2_pairing": l2_mean, "Cosine_mismatch": cos_mean, "L2_norm_ratio": nr_mean, "delta_psnr": dpsnr}

    # If we have at least 2 dpsnr values, compute correlation
    dpsnr_pairs = [(k, v) for k, v in summary.items() if v["delta_psnr"] is not None]
    if len(dpsnr_pairs) >= 2:
        print(f"\n  Metric × ΔPSNR Spearman ρ (n={len(dpsnr_pairs)}):")
        for metric in ["L2_pairing", "Cosine_mismatch", "L2_norm_ratio"]:
            from scipy.stats import spearmanr
            m_vals = [v[metric] for _, v in dpsnr_pairs]
            d_vals = [v["delta_psnr"] for _, v in dpsnr_pairs]
            rho, p = spearmanr(m_vals, d_vals)
            print(f"    {metric:>15s}: ρ={rho:+.4f} p={p:.4f}")

    with open(OUT / "conflict_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}/conflict_metrics.json")

    del pipe; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
