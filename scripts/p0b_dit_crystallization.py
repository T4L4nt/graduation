"""
P0b #10: DiT-S/2 crystallization curve.
Uses the existing Phase 9 diagnostic infrastructure; just swaps in
EMA weights from each intermediate checkpoint.
"""
import json, sys, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dit_controlled_shared import (
    get_dit_s2_model, DIAG_NUM_STEPS, DEVICE, DTYPE,
    ddim_inversion_eps, ddim_reconstruction_eps, get_test_loader,
    discover_dit_hook_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "train_controlled" / "crystallization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "outputs" / "train_controlled" / "epsilon"
CKPT_STEPS = [10000, 20000, 30000, 40000, 50000]


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s > 0 else 0.0


def feat_v2(profile_raw):
    p = np.asarray(profile_raw, dtype=np.float64); L = len(p)
    dmin, dmax = p.min(), p.max()
    pn = (p-dmin)/(dmax-dmin) if dmax > dmin else p.copy()
    pp = float(np.argmax(pn))/L
    k = max(1, int(np.ceil(0.2*L)))
    top = np.argsort(pn)[-k:]
    conc = float(np.sum(pn[top])/np.sum(pn))
    sp = float(gini(pn))
    peaks, _ = find_peaks(pn, prominence=0.1)
    return {"peak_position": pp, "concentration": conc, "spread": sp, "n_peaks": len(peaks)}


def dist_v2(fa, fb):
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"], fa["spread"]-fb["spread"]])
    return {"D_total": float(np.linalg.norm([d_pp, d_mag])), "D_peak_pos": d_pp, "D_mag": d_mag}


def diagnose_dit_checkpoint(ckpt_path, test_loader):
    """Run inversion+reconstruction on all test images, return per-image drift profiles."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_state = ckpt.get("ema", ckpt.get("model"))
    model = get_dit_s2_model().to(DEVICE).to(DTYPE)
    model.load_state_dict(ema_state)
    model.eval()

    targets = discover_dit_hook_targets(model)
    per_image_drifts = []

    for batch in tqdm(test_loader, desc="    diag", leave=False):
        x0 = batch.to(DEVICE).to(DTYPE) if isinstance(batch, torch.Tensor) else batch[0].to(DEVICE).to(DTYPE)
        with torch.no_grad():
            xt, inv_features = ddim_inversion_eps(model, x0, DIAG_NUM_STEPS)
            recon, recon_features = ddim_reconstruction_eps(model, xt, DIAG_NUM_STEPS)

        ld = {}
        for ln in targets:
            if ln in inv_features and ln in recon_features:
                ld[ln] = float(torch.norm(inv_features[ln] - recon_features[ln], p=2).item())
        per_image_drifts.append(ld)

    del model; torch.cuda.empty_cache()

    # Aggregate
    pi_feats = []; pi_profs = []
    for ld in per_image_drifts:
        ordered = sorted(ld.items(), key=lambda x: targets.index(x[0]))
        profile = np.array([v for _, v in ordered])
        pi_feats.append(feat_v2(profile)); pi_profs.append(profile)
    feat_ckpt = {k: float(np.mean([pi[k] for pi in pi_feats])) for k in pi_feats[0]}

    return feat_ckpt, [k for k, _ in ordered], pi_profs


def main():
    test_loader = get_test_loader()
    results = {}

    for ckpt_step in CKPT_STEPS:
        ckpt_path = CKPT_DIR / f"checkpoint_{ckpt_step:06d}.pt"
        if not ckpt_path.exists():
            print(f"SKIP: {ckpt_path} not found")
            continue
        print(f"\nCheckpoint {ckpt_step} steps...")
        feat_ckpt, layer_names, profiles = diagnose_dit_checkpoint(ckpt_path, test_loader)
        results[ckpt_step] = {"features": feat_ckpt, "layers": layer_names,
                               "mean_profile": np.mean(profiles, axis=0).tolist()}
        print(f"  pp={feat_ckpt['peak_position']:.4f} conc={feat_ckpt['concentration']:.4f} sp={feat_ckpt['spread']:.4f}")

    # Crystallization curve: D_s(step_k, step_50k)
    ref = 50000
    ref_feat = results[ref]["features"]
    print(f"\n{'='*60}")
    print(f"Crystallization: D_s(step_k, {ref})")
    print(f"{'Step':>8s}  {'D_total':>10s}  {'D_pp':>8s}  {'D_mag':>8s}  {'pp':>8s}")
    print("-" * 50)
    for s in CKPT_STEPS:
        if s not in results: continue
        dd = dist_v2(results[s]["features"], ref_feat)
        pp = results[s]["features"]["peak_position"]
        print(f"{s:>8d}  {dd['D_total']:10.6f}  {dd['D_peak_pos']:8.6f}  {dd['D_mag']:8.6f}  {pp:8.4f}")

    json.dump({str(k): {"features": v["features"], "pairwise": {
        str(s2): dist_v2(v["features"], results[s2]["features"]) for s2 in CKPT_STEPS if s2 != k
    }} for k, v in results.items()}, open(OUT_DIR / "crystallization.json", "w"), indent=2)
    print(f"\nSaved to {OUT_DIR}/crystallization.json")


if __name__ == "__main__":
    main()
