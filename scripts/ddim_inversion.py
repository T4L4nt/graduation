"""
DDIM Inversion → Reconstruction 最简验证脚本
第一阶段基线实验：复现内容漂移现象
"""

import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms
import lpips

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_model(model_id: str = "runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_image(path: str, size: tuple[int, int] = (512, 512), dtype=DTYPE):
    img = Image.open(path).convert("RGB").resize(size)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE, dtype=dtype)
    return img_tensor, img


def ddim_inversion(
    pipe, latents, prompt_embeds, num_steps: int = 50, guidance_scale: float = 1.0
):
    """DDIM 反演：潜变量 z_0 → z_T。

    从干净潜变量（t≈0）出发，按 timesteps 从小到大反向递推，
    逐步向潜变量注入噪声，最终到达最高 timestep（t=T）。
    """
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps  # 从大到小: [T, T-Δ, ..., t_min]

    uncond_embeds = None
    if guidance_scale > 1.0:
        uncond_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    z = latents.clone()
    # 在 timesteps 末尾补 0，用于第一次反演步（从 t=0 到 t_min）
    extended_ts = timesteps.tolist() + [0]

    with torch.no_grad():
        for i in range(len(extended_ts) - 1, 0, -1):
            t_cur = extended_ts[i]    # 较干净的 timestep（从 0 开始）
            t_next = extended_ts[i - 1]  # 较噪的 timestep

            latent_model_input = (
                torch.cat([z] * 2) if guidance_scale > 1.0 else z
            )
            emb = (
                torch.cat([uncond_embeds, prompt_embeds])
                if guidance_scale > 1.0
                else prompt_embeds
            )
            noise_pred = pipe.unet(
                latent_model_input, t_cur, encoder_hidden_states=emb
            ).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            alpha_cur = scheduler.alphas_cumprod[t_cur]
            alpha_next = scheduler.alphas_cumprod[t_next]

            coeff1 = (alpha_next / alpha_cur).sqrt()
            sigma_cur = (1 - alpha_cur).sqrt()
            sigma_next = (1 - alpha_next).sqrt()
            coeff2 = sigma_next - coeff1 * sigma_cur

            z = coeff1 * z + coeff2 * noise_pred

    return z


def ddim_reconstruction(
    pipe, noise, prompt_embeds, num_steps: int = 50, guidance_scale: float = 1.0
):
    """DDIM 重建：噪声 z_T → 潜变量 z_0"""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=DEVICE)
    timesteps = scheduler.timesteps

    uncond_embeds = None
    if guidance_scale > 1.0:
        uncond_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]

    z = noise.clone()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([z] * 2) if guidance_scale > 1.0 else z
            )
            emb = (
                torch.cat([uncond_embeds, prompt_embeds])
                if guidance_scale > 1.0
                else prompt_embeds
            )
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=emb
            ).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            z = scheduler.step(noise_pred, t, z).prev_sample

    return z


def compute_metrics(original, reconstructed):
    original = original.float()
    reconstructed = reconstructed.float()

    mse = torch.nn.functional.mse_loss(original, reconstructed)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))

    return {"PSNR": psnr.item()}


def main():
    parser = argparse.ArgumentParser(description="DDIM Inversion → Reconstruction")
    parser.add_argument(
        "--image", type=str, default="data/face2.jpg", help="输入图片路径"
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="文本提示词（CFG=1 时可为空）"
    )
    parser.add_argument("--steps", type=int, default=50, help="反演/重建步数")
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG 强度")
    parser.add_argument(
        "--model", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument("--out", type=str, default="data/recon.png", help="输出路径")
    parser.add_argument("--lpips", action="store_true", help="启用 LPIPS 计算")
    args = parser.parse_args()

    print(f"[设备] {DEVICE}")
    print(f"[图片] {args.image}")
    print(f"[步数] {args.steps}")

    # 加载模型
    print("[1/5] 加载模型...")
    pipe = load_model(args.model)

    # 加载图片 & 编码为潜变量
    print("[2/5] 编码图片...")
    img_tensor, pil_img = load_image(args.image, dtype=DTYPE)
    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # 文本嵌入
    print("[3/5] 文本嵌入...")
    prompt_embeds = pipe.encode_prompt(
        args.prompt or "", DEVICE, 1, False
    )[0]

    # DDIM 反演
    print(f"[4/5] DDIM 反演 ({args.steps} 步)...")
    noise = ddim_inversion(
        pipe, latents, prompt_embeds, args.steps, args.guidance
    )

    # DDIM 重建
    print(f"[5/5] DDIM 重建 ({args.steps} 步)...")
    recon_latents = ddim_reconstruction(
        pipe, noise, prompt_embeds, args.steps, args.guidance
    )

    # 解码重建潜变量
    with torch.no_grad():
        recon_tensor = pipe.vae.decode(
            recon_latents / pipe.vae.config.scaling_factor
        ).sample

    # 保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    recon_img = transforms.ToPILImage()(
        (recon_tensor.squeeze(0) / 2 + 0.5).clamp(0, 1)
    )
    recon_img.save(args.out)
    print(f"[保存] {args.out}")

    # 指标
    img_resized = transforms.ToTensor()(pil_img).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    metrics = compute_metrics(img_resized, recon_tensor.clamp(-1, 1))
    print(f"[指标] PSNR: {metrics['PSNR']:.2f} dB")

    if args.lpips:
        lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
        lpips_val = lpips_fn(img_resized, recon_tensor.clamp(-1, 1)).item()
        print(f"[指标] LPIPS: {lpips_val:.4f}")


if __name__ == "__main__":
    main()
