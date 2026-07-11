#!/bin/bash
cd /home/hiaskc/Talant/graduation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=scripts:$PYTHONPATH
echo "=== Task C start at $(date) ==="
conda run --no-capture-output -n grad python scripts/phase8_iclr_sdxl.py --prompts 20 2>&1 | tee outputs/phase8_iclr_sdxl/run.log
echo "=== Task C done at $(date) ==="
