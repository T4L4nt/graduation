#!/bin/bash
cd /home/hiaskc/Talant/graduation
export PYTHONPATH=scripts:$PYTHONPATH
echo "=== Task A start at $(date) ==="
conda run --no-capture-output -n grad python scripts/phase8_iclr_cross_prompt.py --prompts 25 2>&1 | tee outputs/phase8_iclr_cross_prompt/run.log
echo "=== Task A done at $(date) ==="
