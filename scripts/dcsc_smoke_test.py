"""
Quick smoke test for dcsc_core.py — single image, verify no crashes.
Usage: python scripts/dcsc_smoke_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import torch
from phase2_common import DEVICE, DTYPE, load_pipeline, load_image, decode_latent
from phase3_prep import CLIPFeatureExtractor, build_style_cross_attn_tokens
from dcsc_core import (
    CorrectableSubspace, DCSCStyleController, dcsc_controlled_generation
)

IMAGE = "data/coco_val/coco_000000000139.jpg"

def test_subspace():
    print("=== Test 1: CorrectableSubspace ===")
    s = CorrectableSubspace(dim=4)
    # Test empty basis
    v = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = s.project_orthogonal(v)
    print(f"  Empty basis: project_orthogonal([1,0,0,0]) = {out}")
    assert torch.allclose(out, v / v.norm()), f"Empty basis should return normalized input: {out}"

    # Add a basis vector
    grew = s.update(torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    print(f"  Added [2,0,0,0]: grew={grew}, basis_size={s._size()}")
    assert grew

    # Project: a vector in the same direction should have energy ~1 in subspace
    e = s.compute_energy_fraction(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    print(f"  Energy fraction of [1,0,0,0]: {e:.4f}")
    assert e > 0.9, f"Energy should be ~1 in same direction, got {e}"

    # Project orthogonal: a vector in basis direction should be fully removed
    out2 = s.project_orthogonal(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    print(f"  project_orthogonal([1,0,0,0]) = {out2}")
    assert out2.norm().item() < 0.1, f"Should be near-zero after projection, got norm={out2.norm().item()}"

    # Add a different direction
    grew2 = s.update(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    print(f"  Added [0,1,0,0]: grew={grew2}, basis_size={s._size()}")
    assert grew2

    # Add a linearly dependent direction (should not grow)
    grew3 = s.update(torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
    print(f"  Added [0.5,0.5,0,0] (dependent): grew={grew3}, basis_size={s._size()}")
    assert not grew3, "Linearly dependent vector should not grow basis"

    print("  [PASS] CorrectableSubspace\n")

def test_controller(model, processor):
    print("=== Test 2: DCSCStyleController (unit, no model) ===")
    dummy_v_style = torch.randn(1, 768)
    dummy_v_style = dummy_v_style / dummy_v_style.norm(dim=-1, keepdim=True)
    dummy_v_content = torch.randn(1, 768)
    dummy_v_content = dummy_v_content / dummy_v_content.norm(dim=-1, keepdim=True)

    ctrl = DCSCStyleController(
        v_style_base=dummy_v_style,
        v_content=dummy_v_content,
        lambda_0=0.5, Kp=1.0,
    )
    print(f"  Initialized: λ_0=0.5, Kp=1.0")

    # Before initialize(), compute_control should error
    try:
        ctrl.compute_control(None, None)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("  Correctly errors before initialize()")
    print("  [PASS] DCSCStyleController unit tests\n")

def test_generation():
    print("=== Test 3: dcsc_controlled_generation (1 image) ===")
    print(f"  Loading pipeline...")
    pipe = load_pipeline()
    extractor = CLIPFeatureExtractor()

    print(f"  Loading image: {IMAGE}")
    original_latent, original_tensor = load_image(pipe, IMAGE)

    # Prepare embeddings
    prompt_embeds = pipe.encode_prompt("", DEVICE, 1, False)[0]
    v_orig = extractor.encode_image(IMAGE)
    v_content = extractor.encode_text("a photo")

    # Style direction via StyleTex
    _, v_style, cos_check = extractor.compute_orthogonal_decomposition(v_orig, v_content)
    print(f"  StyleTex orthogonal check: cos(v_style, v_content) = {cos_check:.6f}")
    assert abs(cos_check) < 1e-2, f"Style should be orthogonal to content, got cos={cos_check}"

    # Get correction layers
    from phase2_common import get_top_drift_layers
    corr_layers = get_top_drift_layers(5)
    print(f"  Correction layers: {corr_layers[:3]}...")

    # Run DCSC
    print(f"  Running DCSC generation (50 steps, λ_0=0.5, Kp=1.0)...")
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

    try:
        metrics, recon, elapsed, traj = dcsc_controlled_generation(
            pipe, original_latent, original_tensor, prompt_embeds,
            num_steps=50, corr_lam=0.5, corr_layers=corr_layers,
            v_style=v_style, v_content=v_content, extractor=extractor,
            lambda_0=0.5, Kp=1.0, control_freq=5,
            style_mode="extra_token", lpips_fn=lpips_fn,
        )
        print(f"\n  Results:")
        print(f"    PSNR={metrics['PSNR']:.2f}  LPIPS={metrics['LPIPS']:.3f}")
        print(f"    CLIP_content={metrics['CLIP_content']:.4f}  CLIP_style={metrics['CLIP_style']:.4f}")
        print(f"    Elapsed: {elapsed:.1f}s")
        print(f"    Control calls: {traj['n_control_calls']}")
        print(f"    Final basis size: {traj['final_basis_size']}")
        print(f"    Final lambda: {traj['final_lambda']:.3f}")
        print(f"    Stability bound: {traj['stability_bound']:.4f}")
        print(f"    Trajectory:")
        for i, s in enumerate(traj['trajectory']['steps']):
            print(f"      step={s:3d}  λ={traj['trajectory']['lambda'][i]:.3f}  "
                  f"d={traj['trajectory']['d_content_raw'][i]:.4f}  "
                  f"basis={traj['trajectory']['basis_size'][i]}")

        # Sanity checks
        assert 0 <= traj['final_lambda'] <= 0.5, f"Lambda out of range: {traj['final_lambda']}"
        assert traj['final_basis_size'] >= 0, f"Negative basis size: {traj['final_basis_size']}"
        assert metrics['PSNR'] > 10, f"PSNR too low: {metrics['PSNR']}"
        print("\n  [PASS] DCSC generation smoke test")
    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_subspace()
    test_controller(None, None)
    if test_generation():
        print("\n" + "=" * 60)
        print("All smoke tests PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Smoke tests FAILED")
        print("=" * 60)
        sys.exit(1)
