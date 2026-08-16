"""Download RV UNet for C1 boundary measurement."""
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    allow_patterns=["unet/*", "model_index.json"],
    resume_download=True,
)
print(f"Snapshot: {path}")
unet_dir = os.path.join(path, "unet")
for f in os.listdir(unet_dir):
    fpath = os.path.join(unet_dir, f)
    print(f"  {f}: {os.path.getsize(fpath)/1e9:.2f} GB")
print("DONE")
