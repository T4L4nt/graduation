"""
Fig.3: DiT-S/2 crystallization curve — eps vs flow dual comparison.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x); s = np.sum(x)
    return float((2*np.sum(np.arange(1,n+1)*x)-(n+1)*s)/(n*s)) if s > 0 else 0.0

def dist_v2(fa, fb):
    d_pp = abs(fa["peak_position"] - fb["peak_position"])
    d_mag = np.linalg.norm([fa["concentration"]-fb["concentration"], fa["spread"]-fb["spread"]])
    return float(np.linalg.norm([d_pp, d_mag]))

# Load eps (has pairwise distances pre-computed)
with open("outputs/train_controlled/crystallization/crystallization.json") as f:
    eps_data = json.load(f)

# Load flow (has per-step features only, no pairwise)
with open("outputs/train_controlled/crystallization/flow_crystallization.json") as f:
    flow_data = json.load(f)

steps = [10000, 20000, 30000, 40000, 50000]

# EPS: D_total vs 50k
eps_d = []; eps_pp = []
for s in steps:
    key = str(s)
    if key in eps_data:
        pw = eps_data[key]["pairwise"]
        eps_d.append(pw["50000"]["D_total"] if "50000" in pw else 0.0)
        eps_pp.append(eps_data[key]["features"]["peak_position"])

# FLOW: compute D_total vs 50k manually
flow_feat = {str(s): flow_data[str(s)] for s in steps}
flow_ref = flow_feat["50000"]
flow_d = []; flow_pp = []
for s in steps:
    if str(s) in flow_feat:
        flow_d.append(dist_v2(flow_feat[str(s)], flow_ref))
        flow_pp.append(flow_feat[str(s)]["peak_position"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Left panel: D_total convergence ---
ax1.plot(steps, eps_d, "o-", color="#d62728", linewidth=2, markersize=8, markerfacecolor="white", label="eps-prediction (DDIM)")
ax1.plot(steps, flow_d, "s--", color="#1f77b4", linewidth=2, markersize=8, markerfacecolor="white", label="flow matching (Euler)")
ax1.axhline(y=0.016, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax1.annotate("noise floor p95", (45000, 0.018), fontsize=8, color="gray")
ax1.set_xlabel("Training steps"); ax1.set_ylabel("D_total vs 50k checkpoint")
ax1.set_title("Fingerprint convergence", fontweight="bold")
ax1.legend(fontsize=9, framealpha=0.8); ax1.grid(True, alpha=0.2)
ax1.set_xlim(5000, 55000)

# --- Right panel: Peak position convergence ---
ax2.plot(steps, eps_pp, "o-", color="#d62728", linewidth=2, markersize=8, markerfacecolor="white", label="eps-prediction")
ax2.plot(steps, flow_pp, "s--", color="#1f77b4", linewidth=2, markersize=8, markerfacecolor="white", label="flow matching")
ax2.axhline(y=eps_pp[-1], color="green", linestyle=":", linewidth=1, alpha=0.5)
ax2.annotate("final peak (block 11)", (40000, eps_pp[-1]+0.005), fontsize=8, color="green")
ax2.set_xlabel("Training steps"); ax2.set_ylabel("Peak position (relative)")
ax2.set_title("Peak position crystallization", fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.8); ax2.grid(True, alpha=0.2)
ax2.set_xlim(5000, 55000); ax2.set_ylim(0.88, 0.93)

fig.suptitle("Fig. 4: DiT-S/2 Training Crystallization — eps vs flow matching", fontweight="bold")
plt.tight_layout()
fp = OUT_DIR / "fig3_crystallization.pdf"
plt.savefig(fp, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {fp.resolve()}")
