"""
PixArt-α 水彩风格化：手动 img2img，对 COCO 图片做水彩风格迁移。
"""
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import PixArtAlphaPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/pixart_watercolor.png")
    parser.add_argument("--prompt", default="watercolor painting, soft washes, flowing colors, artistic")
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16
    target_size = (1024, 1024)

    print("Loading PixArt-α pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    # Load and preprocess image
    print(f"Loading image: {args.input}")
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size
    image_resized = image.resize(target_size, Image.LANCZOS)

    # Convert to tensor [-1, 1]
    img_tensor = torch.from_numpy(
        (np.array(image_resized).astype(np.float32) / 127.5) - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

    # Encode with VAE
    print("Encoding image to latents...")
    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # Setup noise
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)

    # Determine starting timestep
    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps
    start_step = int(len(timesteps) * args.strength)
    start_step = max(start_step, 1)
    t_start = timesteps[start_step]

    noisy_latents = scheduler.add_noise(latents, noise, t_start.unsqueeze(0))

    # Prepare prompt embeds
    print(f"Running img2img with prompt: '{args.prompt}'")
    print(f"  strength={args.strength}, steps={args.steps}, start_step={start_step}")
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=args.prompt,
        do_classifier_free_guidance=True,
        negative_prompt="blurry, low quality, distorted, ugly",
        device=device,
    )

    # Concatenate negative and positive for CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    # Prepare micro-conditions (required when sample_size == 128)
    batch_size = 1
    resolution = torch.tensor([[1024, 1024]]).repeat(batch_size, 1).to(device, dtype=dtype)
    aspect_ratio = torch.tensor([[1.0]]).repeat(batch_size, 1).to(device, dtype=dtype)
    resolution_cfg = torch.cat([resolution, resolution], dim=0)
    aspect_ratio_cfg = torch.cat([aspect_ratio, aspect_ratio], dim=0)

    # Denoising loop
    latents_noisy = noisy_latents
    with torch.no_grad():
        for i, t in enumerate(timesteps[start_step:]):
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            t_broadcast = t.unsqueeze(0).expand(latent_model_input.shape[0])

            noise_pred = pipe.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t_broadcast,
                added_cond_kwargs={"resolution": resolution_cfg, "aspect_ratio": aspect_ratio_cfg},
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # PixArt predicts 8 channels (noise + variance); keep only noise
            if pipe.transformer.config.out_channels // 2 == latents_noisy.shape[1]:
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            latents_noisy = scheduler.step(noise_pred, t, latents_noisy, return_dict=False)[0]

    # Decode
    print("Decoding...")
    latents_noisy = latents_noisy / pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded = pipe.vae.decode(latents_noisy).sample

    # Convert back to PIL
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    decoded = (decoded * 255).astype("uint8")
    result = Image.fromarray(decoded)
    result = result.resize(orig_size, Image.LANCZOS)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
