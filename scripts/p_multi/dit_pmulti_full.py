"""
DiT-S/2 crystallization P-multi audit (complete).
P-multi: 9-K DDPM-forward fresh-noise reference vs reconstruction trajectory, per-layer L2.
Available checkpoints: eps 10k/50k (20k/30k/40k deleted); flow 10k-50k.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, "/home/hiaskc/Talant/graduation/scripts")
from dit_controlled_shared import (
    discover_dit_hook_targets, get_test_loader, DEVICE,
    DIAG_NUM_STEPS, DDPM_TIMESTEPS, BETA_START, BETA_END,
    NoiseScheduleDDPM, DiTTransformer2DModel, TransformerFeatureHooker,
)
from diffusers import DDIMScheduler

PROJECT_ROOT = Path("/home/hiaskc/Talant/graduation")
CKPT_DIRS = {
    "eps": PROJECT_ROOT / "outputs/train_controlled/epsilon",
    "flow": PROJECT_ROOT / "outputs/train_controlled/flow",
}
CKPT_STEPS = [10000, 20000, 30000, 40000, 50000]

def get_K(n):
    k = [0, 1, 2, n//2-1, n//2, n//2+1, n-3, n-2, n-1]
    return sorted(set(max(0, min(n-1, i)) for i in k))

def load_model(ckpt_path, paradigm):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("ema", ckpt.get("model", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = DiTTransformer2DModel(
        sample_size=64//8, patch_size=2, in_channels=4,
        hidden_size=384, depth=12, num_heads=6, num_classes=1000,
        learn_sigma=False, prediction_type="epsilon",
    )
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model

def diagnose_pmulti(model, loader, paradigm):
    sched = DDIMScheduler(num_train_timesteps=DDPM_TIMESTEPS,
                          beta_start=BETA_START, beta_end=BETA_END,
                          prediction_type="epsilon")
    sched.set_timesteps(DIAG_NUM_STEPS, device=DEVICE)
    timesteps = sched.timesteps
    K = get_K(DIAG_NUM_STEPS)

    targets = discover_dit_hook_targets(model)
    hooker = TransformerFeatureHooker(model)
    hooker.register()

    per_layer = defaultdict(list)
    for batch in tqdm(loader, desc=f"  {paradigm}", leave=False):
        x0 = batch[0].to(DEVICE)
        cl = torch.zeros(x0.shape[0], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            # Inversion: x0 -> xT (DDIM eps, eta=0)
            z = x0.clone()
            ext = timesteps.tolist() + [0]
            for i in range(len(ext)-1, 0, -1):
                tc, tn = ext[i], ext[i-1]
                eps_pred = model(z, timestep=torch.tensor([tc], device=DEVICE),
                                 class_labels=cl).sample
                ac, an = sched.alphas_cumprod[tc], sched.alphas_cumprod[tn]
                x0p = (z - (1-ac).sqrt()*eps_pred) / ac.sqrt().clamp(min=1e-8)
                z = an.sqrt()*x0p + (1-an).sqrt()*eps_pred

            # Reconstruction trajectory (eta=0: z_{t-1} = sqrt(a_{t-1})*x0_pred)
            traj = {}
            for i, t in enumerate(timesteps):
                eps_pred = model(z, timestep=torch.tensor([t], device=DEVICE),
                                 class_labels=cl).sample
                alpha = sched.alphas_cumprod[t]
                x0p = (z - (1-alpha).sqrt()*eps_pred) / alpha.sqrt().clamp(min=1e-8)
                z = alpha.sqrt()*x0p
                if i in K:
                    traj[i] = z.clone()

            # P-multi comparison: fresh-noise reference at matched t
            torch.manual_seed(42)
            eps_ref = torch.randn_like(x0)
            for kidx in K:
                t = timesteps[kidx]
                alpha = sched.alphas_cumprod[t]
                z_ref = alpha.sqrt()*x0 + (1-alpha).sqrt()*eps_ref
                z_rec = traj[kidx]

                hooker.features.clear()
                with torch.no_grad():
                    model(z_ref, timestep=torch.tensor([t], device=DEVICE),
                          class_labels=cl).sample
                rf = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}

                hooker.features.clear()
                with torch.no_grad():
                    model(z_rec, timestep=torch.tensor([t], device=DEVICE),
                          class_labels=cl).sample
                rc = {k: v.detach().cpu().clone() for k, v in hooker.features.items()}

                for ln in targets:
                    if ln in rf and ln in rc:
                        per_layer[ln].append(float(torch.norm(rf[ln].float()-rc[ln].float(), p=2).item()))

    hooker.remove()
    profile = {ln: float(np.mean(v)) for ln, v in per_layer.items()}
    return profile, targets

def extract_v2(profile, layer_names):
    p = np.asarray([profile[ln] for ln in layer_names], dtype=np.float64)
    L = len(p)
    pn = (p-p.min())/(p.max()-p.min()) if p.max() > p.min() else p
    idx = int(pn.argmax()); pp = idx/L
    kk = max(1, int(np.ceil(0.2*L)))
    conc = float(np.sum(pn[np.argsort(pn)[-kk:]])/np.sum(pn))
    def gini(x):
        x = np.sort(np.asarray(x, dtype=np.float64))
        n = len(x); s = np.sum(x)
        return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s > 0 else 0.0
    sp = float(gini(pn))
    return {"peak_position": pp, "concentration": conc, "spread": sp,
            "L": L, "peak_layer": layer_names[idx]}

results = {}
for paradigm in ["eps", "flow"]:
    for step in CKPT_STEPS:
        ckpt_path = CKPT_DIRS[paradigm] / f"checkpoint_{step:06d}.pt"
        if not ckpt_path.exists():
            print(f"[skip] {paradigm} {step}: checkpoint deleted (P-t0 data retained as historical)")
            continue
        print(f"\n=== {paradigm} {step} P-multi ===")
        model = load_model(ckpt_path, paradigm)
        loader = get_test_loader()
        profile, targets = diagnose_pmulti(model, loader, paradigm)
        feat = extract_v2(profile, targets)
        results[f"{paradigm}_{step}"] = feat
        print(f"  pp={feat['peak_position']:.4f} conc={feat['concentration']:.4f} "
              f"sp={feat['spread']:.4f} peak={feat['peak_layer']}")
        del model; torch.cuda.empty_cache()

out = PROJECT_ROOT / "outputs/train_controlled/crystallization/p_multi_audit.json"
with open(out, "w") as f:
    json.dump({"protocol_id": "P-multi-v1",
               "K_indices": list(get_K(DIAG_NUM_STEPS)),
               "available_checkpoints": sorted(results.keys()),
               "scope_limitation": "eps 20k/30k/40k checkpoints deleted before P-multi audit; "
                                   "P-t0 values retained in crystallization.json",
               "results": results}, f, indent=2)
print(f"\nSaved: {out}")
