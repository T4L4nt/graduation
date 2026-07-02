"""Batch watercolor stylization for all COCO images using PixArt-α."""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import PixArtAlphaPipeline
from tqdm import tqdm


def main():
    input_dir = Path("data/coco_val")
    output_dir = Path("outputs/pixart_watercolor")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = "watercolor painting, soft flowing washes, artistic brushstrokes, vibrant translucent colors"
    strength = 0.5
    steps = 25
    seed = 42
    target_size = (1024, 1024)
    device = "cuda"
    dtype = torch.float16

    print("Loading PixArt-α pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    scheduler = pipe.scheduler
    vae_scale = pipe.vae.config.scaling_factor
    out_ch = pipe.transformer.config.out_channels

    # Pre-compute prompt embeds (shared across all images)
    print("Encoding prompt...")
    (prompt_embeds, prompt_attention_mask,
     neg_embeds, neg_mask) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        negative_prompt="blurry, low quality, distorted, ugly",
        device=device,
    )
    prompt_embeds_cfg = torch.cat([neg_embeds, prompt_embeds], dim=0)
    prompt_mask_cfg = torch.cat([neg_mask, prompt_attention_mask], dim=0)

    resolution_cfg = torch.tensor([[1024, 1024]]).repeat(2, 1).to(device, dtype=dtype)
    aspect_ratio_cfg = torch.tensor([[1.0]]).repeat(2, 1).to(device, dtype=dtype)

    images = sorted(input_dir.glob("*.jpg"))
    print(f"Processing {len(images)} images...")

    for img_path in tqdm(images):
        out_path = output_dir / f"{img_path.stem}_watercolor.png"
        if out_path.exists():
            continue  # skip already done

        image = Image.open(img_path).convert("RGB")
        orig_size = image.size
        image_resized = image.resize(target_size, Image.LANCZOS)

        img_tensor = torch.from_numpy(
            (np.array(image_resized).astype(np.float32) / 127.5) - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            latents = pipe.vae.encode(img_tensor).latent_dist.sample()
            latents = latents * vae_scale

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)

        scheduler.set_timesteps(steps, device=device)
        timesteps = scheduler.timesteps
        start_step = max(int(len(timesteps) * strength), 1)
        t_start = timesteps[start_step]

        noisy_latents = scheduler.add_noise(latents, noise, t_start.unsqueeze(0))

        latents_noisy = noisy_latents
        with torch.no_grad():
            for t in timesteps[start_step:]:
                latent_input = torch.cat([latents_noisy] * 2)
                latent_input = scheduler.scale_model_input(latent_input, t)
                t_broadcast = t.unsqueeze(0).expand(2)

                noise_pred = pipe.transformer(
                    latent_input,
                    encoder_hidden_states=prompt_embeds_cfg,
                    encoder_attention_mask=prompt_mask_cfg,
                    timestep=t_broadcast,
                    added_cond_kwargs={"resolution": resolution_cfg, "aspect_ratio": aspect_ratio_cfg},
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                if out_ch // 2 == latents_noisy.shape[1]:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                latents_noisy = scheduler.step(noise_pred, t, latents_noisy, return_dict=False)[0]

        latents_noisy = latents_noisy / vae_scale
        with torch.no_grad():
            decoded = pipe.vae.decode(latents_noisy).sample

        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        decoded = (decoded * 255).astype("uint8")
        result = Image.fromarray(decoded)
        result = result.resize(orig_size, Image.LANCZOS)
        result.save(out_path)

    print(f"Done. Saved to {output_dir}/")


if __name__ == "__main__":
    main()
