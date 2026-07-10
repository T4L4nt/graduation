#!/bin/bash
cd /home/hiaskc/Talant/graduation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=scripts:$PYTHONPATH
echo "=== Task B start at $(date) ==="
echo "GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
python -u scripts/phase8_iclr_editing.py --steps 20 2>&1 | tee outputs/phase8_iclr_editing/run.log
echo "=== Task B done at $(date), exit=$? ==="
