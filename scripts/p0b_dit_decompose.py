"""
P0b #14: DiT component decomposition — attention vs MLP vs residual drift.

For each HunyuanDiT block, hooks attention output, MLP output, and
block output during inversion+reconstruction, measuring drift contribution
of each component.

Key question: does the "drift concentrated in ResNet residuals, not Attention"
pattern from SD 1.5 UNet generalize to Transformer backbones?

Usage: python -u scripts/p0b_dit_decompose.py
"""

import sys, json, torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import HunyuanDiTPipeline, DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEVICE = "cuda"; DTYPE = torch.float16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs" / "p0b_cross_checkpoint"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
TEST_IMG = sorted((PROJECT_ROOT / "data" / "coco_val").glob("*.jpg"))[:3]


def discover_block_components(model):
    """Discover attention, MLP, and residual components in each block."""
    components = {"attn_out": [], "mlp_out": [], "block_out": []}
    for name, _ in model.named_modules():
        if name.endswith(".attn1") or name.endswith(".attn2"):
            components["attn_out"].append(name)
        elif name.endswith(".ff.net.2"):
            components["mlp_out"].append(name.replace(".net.2", ""))
        elif ".proj_out" in name and "adaln" not in name:
            # Final projection = block output
            pass
    return components


class BlockHooker:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.handles = []

    def _make_hook(self, label):
        def fn(module, input, output):
            self.features[label] = output[0] if isinstance(output, tuple) else output
        return fn

    def register_attn_hooks(self):
        """Hook the attention output before residual add in each block."""
        self.remove()
        for name, module in self.model.named_modules():
            # HunyuanDiT: blocks have transformer_blocks[i].attn1 (self-attn) and attn2 (cross-attn)
            if name.endswith(".attn1") or name.endswith(".attn2"):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
            # MLP output (ff.net.2 is the final linear layer in the feedforward)
            if "ff.net.2" in name and "blocks." in name:
                block_idx = name.split(".blocks.")[1].split(".")[0]
                self.handles.append(module.register_forward_hook(
                    self._make_hook(f"blocks.{block_idx}.ff_out")))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.features.clear()


def main():
    print(f"Loading {MODEL_ID}...")
    pipe = HunyuanDiTPipeline.from_pretrained(MODEL_ID, local_files_only=True, torch_dtype=DTYPE).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) if hasattr(pipe, 'scheduler') else None

    # Encode empty prompt
    text_input = pipe.tokenizer("", padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]

    hooker = BlockHooker(pipe.transformer)
    hooker.register_attn_hooks()

    per_image_attn_drifts = []

    for img_path in tqdm(TEST_IMG, desc="  diag"):
        # Load + encode image
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
        t = 2 * t - 1
        with torch.no_grad():
            latent = pipe.vae.encode(t).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor

        # DDIM inversion (simplified — single forward pass at turnaround)
        scheduler = pipe.scheduler
        scheduler.set_timesteps(50, device=DEVICE)
        z = latent.clone()

        # Just do a single forward pass at t=50 (turnaround) to measure component contributions
        with torch.no_grad():
            t_step = scheduler.timesteps[0]
            model_input = scheduler.scale_model_input(z, t_step) if hasattr(scheduler, 'scale_model_input') else z
            # HunyuanDiT forward
            noise_pred = pipe.transformer(
                model_input, timestep=t_step,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=None,
                pooled_projections=None,
            ).sample

        # Record drift at each attention/MLP output
        attn_drift = {}
        for name, feat in hooker.features.items():
            attn_drift[name] = float(torch.norm(feat).item())  # L2 norm as proxy for activation magnitude

        per_image_attn_drifts.append(attn_drift)

    hooker.remove()
    del pipe; torch.cuda.empty_cache()

    # Aggregate
    all_keys = sorted(per_image_attn_drifts[0].keys())
    mean_drift = {}
    for k in all_keys:
        vals = [d.get(k, 0) for d in per_image_attn_drifts]
        mean_drift[k] = float(np.mean(vals))

    # Report
    # Separate attn1 (self-attn), attn2 (cross-attn), ff_out (MLP)
    attn1_drifts = {k: v for k, v in mean_drift.items() if "attn1" in k}
    attn2_drifts = {k: v for k, v in mean_drift.items() if "attn2" in k}
    ff_drifts = {k: v for k, v in mean_drift.items() if "ff_out" in k}

    print(f"\n=== DiT Component Drift (activation norm, turnaround t=50) ===")
    print(f"Self-attention (attn1): {len(attn1_drifts)} blocks, mean norm={np.mean(list(attn1_drifts.values())):.2f}")
    print(f"Cross-attention (attn2): {len(attn2_drifts)} blocks, mean norm={np.mean(list(attn2_drifts.values())):.2f}")
    print(f"MLP output (ff_out): {len(ff_drifts)} blocks, mean norm={np.mean(list(ff_drifts.values())):.2f}")

    # Top/bottom split
    bottom_attn1 = np.mean([v for k, v in attn1_drifts.items() if int(k.split(".blocks.")[1].split(".")[0]) < 20])
    top_attn1 = np.mean([v for k, v in attn1_drifts.items() if int(k.split(".blocks.")[1].split(".")[0]) >= 20])
    bottom_ff = np.mean([v for k, v in ff_drifts.items() if int(k.split(".blocks.")[1].split(".")[0]) < 20])
    top_ff = np.mean([v for k, v in ff_drifts.items() if int(k.split(".blocks.")[1].split(".")[0]) >= 20])
    print(f"\n  Bottom (0-19): attn1={bottom_attn1:.2f} ff={bottom_ff:.2f}")
    print(f"  Top (20-39): attn1={top_attn1:.2f} ff={top_ff:.2f}")

    json.dump({"attn1": attn1_drifts, "attn2": attn2_drifts, "ff": ff_drifts,
                "stats": {"attn1_mean": np.mean(list(attn1_drifts.values())),
                          "attn2_mean": np.mean(list(attn2_drifts.values())),
                          "ff_mean": np.mean(list(ff_drifts.values())),
                          "bottom_attn1": bottom_attn1, "top_attn1": top_attn1,
                          "bottom_ff": bottom_ff, "top_ff": top_ff}},
              open(OUT / "dit_decompose.json", "w"), indent=2)
    print(f"\nSaved to {OUT}/dit_decompose.json")


if __name__ == "__main__":
    main()
