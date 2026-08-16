"""Download SD3.5 transformer + configs for 100-image measurement."""
import os, sys
from huggingface_hub import hf_hub_download

repo = "stabilityai/stable-diffusion-3.5-large"
files = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_3/special_tokens_map.json",
    "tokenizer_3/tokenizer_config.json",
    "tokenizer_3/vocab.json",
]

for f in files:
    print(f"[{files.index(f)+1}/{len(files)}] {f} ...", flush=True)
    try:
        path = hf_hub_download(repo, f, resume_download=True)
        size = os.path.getsize(path)
        print(f"  OK ({size/1e9:.2f} GB)", flush=True)
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)

# Report disk
import subprocess
subprocess.run(["df", "-h", "/"])
print("DONE")
