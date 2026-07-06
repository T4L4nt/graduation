"""
Fill missing EDICT and NTI results for 4 coco_val images.

Images already covered: 139, 285, 632, 724, 776, 785, 802, 872, 885,
                         1000, 1353, 1490, 1532, 1584, 1675
Missing: 1818, 2153, 2261, 2532
"""

import json, time, sys
from pathlib import Path

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ / "scripts"))

from phase2_common import (
    DEVICE, DTYPE, MODEL_ID,
    load_pipeline, load_image, decode_latent,
    ddim_inversion, ddim_reconstruction,
    ddim_inversion_with_features, ddim_reconstruction_with_correction,
    FeatureCorrector,
    compute_metrics, save_recon_img, get_top_drift_layers,
    DEFAULT_TEST_IMAGES,
)

# Phase 2 EDICT imports
from phase2_edict import (
    edict_inversion, edict_reconstruction,
    EDICT_P, EDICT_CORR_LAMBDA, OUT_DIR as EDICT_OUT,
)

# Phase 2 NTI imports
from phase2_nti import (
    generate_caption, load_or_generate_captions,
    nti_optimize_and_reconstruct, ddim_inversion_trajectory,
    NTI_N_ITER, NTI_GUIDANCE, NTI_INITIAL_LR,
    OUT_DIR as NTI_OUT,
)

MISSING_IMAGES = [
    "data/coco_val/coco_000000001818.jpg",
    "data/coco_val/coco_000000002153.jpg",
    "data/coco_val/coco_000000002261.jpg",
    "data/coco_val/coco_000000002532.jpg",
]

NUM_STEPS = 50


def run_edict_missing():
    """Run EDICT on missing images."""
    print("=" * 60)
    print("EDICT: Running on 4 missing coco_val images")
    print("=" * 60)

    pipe = load_pipeline()
    top_layers = get_top_drift_layers(k=5)
    collector_cls = type("Collector", (), {
        "TOP_LAYERS": top_layers,
    })  # simplified

    # Load existing metrics
    metrics_path = EDICT_OUT / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_results = json.load(f)
    else:
        all_results = []

    for img_path in MISSING_IMAGES:
        img_name = Path(img_path).stem
        print(f"\n[EDICT] {img_name}")

        t0 = time.time()
        original = load_image(img_path, resize=512)

        # Prompt
        if img_name.startswith("coco"):
            prompt = "a photo of a" if "coco" in img_name else ""
            prompt = "a photograph"  # generic for coco_val
        else:
            prompt = "a photograph"

        pipe.to(DEVICE)

        # DDIM baseline
        inv_latent = ddim_inversion(pipe, original, prompt, NUM_STEPS)
        recon_ddim = ddim_reconstruction(pipe, inv_latent, prompt, NUM_STEPS)
        metrics_ddim = compute_metrics(original, recon_ddim, decode_latent(pipe, recon_ddim))
        metrics_ddim["image"] = img_name
        metrics_ddim["method"] = "DDIM"
        metrics_ddim["steps"] = NUM_STEPS
        metrics_ddim["time_s"] = time.time() - t0

        # EDICT
        t0 = time.time()
        x_T, y_T = edict_inversion(pipe, inv_latent, prompt, NUM_STEPS, EDICT_P)
        recon_edict = edict_reconstruction(pipe, x_T, y_T, prompt, NUM_STEPS, EDICT_P)
        metrics_edict = compute_metrics(original, recon_edict, decode_latent(pipe, recon_edict))
        metrics_edict["image"] = img_name
        metrics_edict["method"] = "EDICT"
        metrics_edict["steps"] = NUM_STEPS
        metrics_edict["time_s"] = time.time() - t0

        # Save recons
        save_recon_img(original, recon_ddim, EDICT_OUT / "recons", img_name, "DDIM")
        save_recon_img(original, recon_edict, EDICT_OUT / "recons", img_name, "EDICT")

        all_results.append(metrics_ddim)
        all_results.append(metrics_edict)
        print(f"  DDIM:  PSNR={metrics_ddim['PSNR']:.2f}  LPIPS={metrics_ddim['LPIPS']:.3f}")
        print(f"  EDICT: PSNR={metrics_edict['PSNR']:.2f}  LPIPS={metrics_edict['LPIPS']:.3f}")

    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} total entries to {metrics_path}")


def run_nti_missing():
    """Run NTI on missing images."""
    print("\n" + "=" * 60)
    print("NTI: Running on 4 missing coco_val images")
    print("=" * 60)

    # Generate captions first
    all_images = DEFAULT_TEST_IMAGES + MISSING_IMAGES
    captions = load_or_generate_captions(all_images)

    pipe = load_pipeline()

    # Load existing metrics
    metrics_path = NTI_OUT / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_results = json.load(f)
    else:
        all_results = []

    for img_path in MISSING_IMAGES:
        img_name = Path(img_path).stem
        caption = captions.get(img_path, "a photograph")
        print(f"\n[NTI] {img_name} | caption: {caption[:60]}")

        t0 = time.time()
        original = load_image(img_path, resize=512)
        pipe.to(DEVICE)

        # DDIM inversion to get trajectory
        inv_latent, z_T, trajectory = ddim_inversion_trajectory(pipe, original, caption, NUM_STEPS)

        # NTI reconstruction
        recon_nti = nti_optimize_and_reconstruct(
            pipe, z_T, trajectory, caption, NUM_STEPS,
            NTI_N_ITER, NTI_GUIDANCE, NTI_INITIAL_LR,
        )

        latency = time.time() - t0
        metrics_nti = compute_metrics(original, recon_nti, decode_latent(pipe, recon_nti))
        metrics_nti["image"] = img_name
        metrics_nti["method"] = "NTI"
        metrics_nti["steps"] = NUM_STEPS
        metrics_nti["time_s"] = latency

        save_recon_img(original, recon_nti, NTI_OUT / "recons", img_name, "NTI")
        all_results.append(metrics_nti)
        print(f"  NTI: PSNR={metrics_nti['PSNR']:.2f}  LPIPS={metrics_nti['LPIPS']:.3f}  ({latency:.1f}s)")

    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} total entries to {metrics_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--edict", action="store_true", help="Run EDICT on missing images")
    parser.add_argument("--nti", action="store_true", help="Run NTI on missing images")
    parser.add_argument("--all", action="store_true", help="Run both")
    args = parser.parse_args()

    if not (args.edict or args.nti or args.all):
        print("Usage: python scripts/fill_missing_edict_nti.py --edict | --nti | --all")
        sys.exit(1)

    if args.edict or args.all:
        run_edict_missing()
    if args.nti or args.all:
        run_nti_missing()
