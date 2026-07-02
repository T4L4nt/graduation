"""
Phase 4 SOTA 对比：ControlNet (Canny edge)

ControlNet 是条件生成方法，非反演-重建。用 Canny 边缘图作为结构约束，
在风格 prompt 引导下生成保持原图结构的风格化图像。

对比维度：
- 结构保持：原图 vs ControlNet 输出的 PSNR/SSIM/LPIPS
- 风格注入：CLIP 方向相似度
- 与我们的方法对比：反演-重建 vs 条件生成的根本差异

用法:
  python scripts/phase4_controlnet.py --images data/coco_val/coco_000000000139.jpg
  python scripts/phase4_controlnet.py --n-images 3  # coco_val 前 3 张
"""

import argparse, json, sys, time, os
from pathlib import Path

import torch, numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import lpips

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import DEVICE, DTYPE, compute_metrics, decode_latent, save_recon_img
from phase3_prep import CLIPFeatureExtractor, build_style_cross_attn_tokens

OUT_DIR = Path("outputs/phase4_sota/controlnet")
STYLE_TEXTS = [
    "an oil painting in impressionist style",
    "a watercolor sketch with soft brushstrokes",
    "a neon-lit cyberpunk scene",
]


def load_controlnet_pipe():
    """Load ControlNet + SD 1.5 pipeline."""
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def extract_canny_edges(img_path: str, low=100, high=200):
    """Extract Canny edge map from image. Returns PIL Image."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges)


def run_controlnet(pipe, img_path, edge_img, style_text, strength,
                   num_steps=50, seed=42):
    """Generate stylized image via ControlNet + style prompt."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=style_text,
            image=edge_img,
            num_inference_steps=num_steps,
            controlnet_conditioning_scale=strength,
            generator=generator,
        ).images[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 1.0])
    args = parser.parse_args()

    # Determine images
    if args.images:
        images = args.images
    else:
        coco = sorted(Path("data/coco_val").glob("*.jpg"))
        images = [str(p) for p in coco[:args.n_images]]

    print(f"[ControlNet] {len(images)} images, {args.steps} steps")
    print(f"[Strengths] {args.strengths}")

    # Load pipeline
    print("[0] Loading ControlNet pipeline...")
    pipe = load_controlnet_pipe()
    extractor = CLIPFeatureExtractor()
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for img_path in images:
        if not os.path.exists(img_path):
            continue
        img_name = Path(img_path).stem
        print(f"\n{'='*50}\n{img_name}")

        # Load original image as tensor for metrics
        img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
        from torchvision import transforms
        orig_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)
        orig_tensor = 2 * orig_tensor - 1  # [-1, 1]

        # Extract Canny edges
        edge_img = extract_canny_edges(img_path)
        v_orig = extractor.encode_image(img_path)

        for style_text in STYLE_TEXTS:
            style_key = style_text.split()[-2]  # rough short name
            for strength in args.strengths:
                tag = f"cn_{style_key}_s{strength:.1f}"
                t0 = time.perf_counter()
                result_pil = run_controlnet(
                    pipe, img_path, edge_img, style_text, strength, args.steps
                )
                elapsed = time.perf_counter() - t0

                # Convert result to tensor for metrics (resize to 512x512)
                result_pil_512 = result_pil.resize((512, 512), Image.LANCZOS)
                result_tensor = transforms.ToTensor()(result_pil_512).unsqueeze(0).to(DEVICE)
                result_tensor = 2 * result_tensor - 1

                metrics = compute_metrics(orig_tensor, result_tensor, lpips_fn)

                # CLIP metrics
                v_result = extractor.encode_image_from_tensor(result_tensor)
                v_style = extractor.encode_text(style_text)
                clip_style = float((v_result * v_style).sum())
                clip_content = float((v_result * v_orig).sum())

                row = {
                    "image": img_name, "style": style_key,
                    "strength": strength, **metrics,
                    "CLIP_style": clip_style, "CLIP_content": clip_content,
                    "time_s": elapsed,
                }
                all_results.append(row)
                print(f"  {tag}: PSNR={metrics['PSNR']:.2f} LPIPS={metrics['LPIPS']:.3f}  "
                      f"CLIP_s={clip_style:.3f} CLIP_c={clip_content:.3f}  "
                      f"({elapsed:.1f}s)")

                # Save image
                save_recon_img(result_tensor, OUT_DIR, img_name, args.steps, tag)

                del result_tensor
                torch.cuda.empty_cache()

    # Save metrics
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("ControlNet Summary (best CLIP_style per image)")
    for img_name in sorted(set(r["image"] for r in all_results)):
        img_results = [r for r in all_results if r["image"] == img_name]
        best = max(img_results, key=lambda r: r["CLIP_style"])
        print(f"  {img_name}: best={best['style']}_s{best['strength']:.1f}  "
              f"PSNR={best['PSNR']:.2f}  CLIP_s={best['CLIP_style']:.3f}")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
